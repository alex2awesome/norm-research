"""a118: Rust API surface and crate organization.

Aspect (from aspects.json[118]):
    "Design and organize Rust crates and APIs (modules, constructors,
     macros, extension traits, feature flags, public surface) for clarity,
     interop, and idiomatic use."

The norm is composed of several well-known Rust API-design conventions
that are individually checkable from source via the tree-sitter-rust
grammar. We do NOT regex-parse Rust source — we walk the AST and aggregate
per-item booleans.

What we check on the added Rust lines per file:

  R1. **Public-item documentation.** Every item declared `pub` (struct,
      enum, trait, fn, mod, type alias, const, static) should be preceded
      by an outer doc comment (`///` or `/** */`). Idiomatic Rust (and
      `rustdoc`'s `missing_docs` clippy lint) treat undocumented public
      items as an API-surface smell.

  R2. **Constructor convention.** An `impl T` block that contains any
      `pub fn` and whose type `T` is a `pub struct`/`pub enum` with at
      least one *private* field should expose a constructor — `pub fn new`,
      `pub fn with_*`, `pub fn from_*`, `pub fn builder`, or be paired with
      `impl Default for T`. Without one, callers cannot construct the
      type at all (private fields cannot be set from outside the crate).
      This is the most concrete Rust API-organization rule.

  R3. **Macro export hygiene.** Every top-level `macro_rules! NAME` should
      either (a) be preceded by `#[macro_export]` (making it intentionally
      public), or (b) be in a non-public module path (we approximate this
      by checking the macro identifier name — leading underscore or being
      inside a `mod` whose declaration is not `pub`). Naked `macro_rules!`
      in lib.rs / mod.rs are an API-surface mistake — they leak via
      `#[macro_use]` consumers and are hard to deprecate.

  R4. **Feature-gate correctness.** `#[cfg(feature = "NAME")]` is the
      idiomatic gate. We award credit when feature gates use named
      `feature = "..."` predicates rather than reusing `#[cfg(test)]` or
      `#[cfg(not(debug_assertions))]` to gate *public* items (which is a
      known antipattern — public API surface shouldn't flip on test mode).

  R5. **`pub use` re-export organization.** A `pub use` re-export should
      either be from `crate::` / `self::` / `super::` (intra-crate
      curation — the idiomatic facade pattern), or be a renaming
      (`pub use foo::Bar as Baz`). A bare `pub use external_crate::*` at
      module root re-exports an entire foreign API surface and is the
      most common organization mistake.

Score per file = mean over (R1..R5) of (#satisfied / #applicable).
Rules that have zero applicable items in a file are excluded from that
file's mean (rather than treated as 1.0) so a file with only one rule's
worth of items gets that rule's signal undiluted.

Applies iff: the diff adds at least one `.rs` file with at least one
top-level item that triggers at least one of R1..R5. Most diffs in our
fixtures contain zero Rust — this metric ABSTAINS on those (returns None,
applies()=False).

Classification: PARTIALLY_THIN — these are real, well-defined idioms,
but "API surface organization" as a whole includes taste-level concerns
(naming, trait granularity, sealed-trait patterns, builder ergonomics)
that no static check can measure. We score the thin slice and let
downstream models combine it with other signals.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..sandbox import added_files_by_ext

ASPECT_ID = "a118"
ASPECT_NAME = "Rust API surface and crate organization"
TIER = 2
TOOLS = ["tree-sitter-rust"]
APPLIES_TO_LANGS = ["Rust"]
CLASSIFICATION = "PARTIALLY_THIN"

RS_EXTS = [".rs"]


# ---------- parser cache ----------

_PARSER = None
_PARSER_TRIED = False


def _parser():
    global _PARSER, _PARSER_TRIED
    if _PARSER_TRIED:
        return _PARSER
    _PARSER_TRIED = True
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_rust as mod
        _PARSER = Parser(Language(mod.language()))
    except Exception:
        _PARSER = None
    return _PARSER


def _text(n, src: bytes) -> str:
    return src[n.start_byte:n.end_byte].decode("utf8", errors="replace")


# ---------- AST helpers ----------

PUBLIC_ITEM_TYPES = {
    "struct_item", "enum_item", "trait_item", "function_item",
    "mod_item", "type_item", "const_item", "static_item",
    "union_item",
}


def _has_visibility_pub(item, src: bytes) -> bool:
    """A node is `pub`-visible iff it has a visibility_modifier child whose
    text starts with `pub`. (Includes `pub`, `pub(crate)`, `pub(super)`,
    `pub(in path)` — all of which expose the item beyond its parent
    module.) We treat `pub(self)` as private (legal but unusual).
    """
    for c in item.children:
        if c.type == "visibility_modifier":
            t = _text(c, src).strip()
            if t == "pub" or (t.startswith("pub(") and not t.startswith("pub(self")):
                return True
    return False


def _name_of(item, src: bytes) -> Optional[str]:
    nm = item.child_by_field_name("name")
    if nm is not None:
        return _text(nm, src)
    # fallback: first identifier/type_identifier child
    for c in item.children:
        if c.type in ("identifier", "type_identifier"):
            return _text(c, src)
    return None


def _sibling_index(item) -> int:
    """tree-sitter-python wrapper nodes don't preserve Python identity
    across `.children` accesses, so we locate siblings by byte range,
    which uniquely identifies a node within its parent's children."""
    parent = item.parent
    if parent is None:
        return -1
    sb, eb = item.start_byte, item.end_byte
    for i, c in enumerate(parent.children):
        if c.start_byte == sb and c.end_byte == eb and c.type == item.type:
            return i
    return -1


def _has_doc_comment_before(item, src: bytes, root) -> bool:
    """True if the line(s) directly preceding `item` contain a `///`
    outer doc comment or a `/** ... */` outer doc comment.

    We look at the parent's children list and find the previous sibling.
    Both `line_comment` and `block_comment` nodes are exposed by
    tree-sitter-rust; we inspect their leading characters to distinguish
    doc from regular comments.
    """
    parent = item.parent
    if parent is None:
        return False
    siblings = list(parent.children)
    idx = _sibling_index(item)
    if idx <= 0:
        return False
    # Walk backwards past attribute_items (common: doc above #[derive] above struct)
    j = idx - 1
    while j >= 0 and siblings[j].type == "attribute_item":
        j -= 1
    if j < 0:
        return False
    prev = siblings[j]
    if prev.type == "line_comment":
        t = _text(prev, src)
        # `///` outer doc OR `//!` inner doc (the latter applies to the
        # enclosing item, not the next item, so we don't credit it)
        if t.startswith("///") and not t.startswith("////"):
            return True
    if prev.type == "block_comment":
        t = _text(prev, src)
        if t.startswith("/**") and not t.startswith("/***"):
            return True
    return False


def _has_attribute(item, src: bytes, attr_predicate) -> bool:
    """True iff one of the attribute_item siblings directly above `item`
    satisfies `attr_predicate(text_of_attribute)`.
    """
    parent = item.parent
    if parent is None:
        return False
    siblings = list(parent.children)
    idx = _sibling_index(item)
    if idx < 0:
        return False
    j = idx - 1
    while j >= 0 and siblings[j].type in ("attribute_item", "line_comment",
                                          "block_comment"):
        if siblings[j].type == "attribute_item":
            t = _text(siblings[j], src)
            if attr_predicate(t):
                return True
        j -= 1
    return False


# ---------- R1: doc comments on public items ----------

def _r1_pub_items_documented(root, src: bytes) -> Tuple[int, int]:
    """Return (#documented_pub_items, #pub_items)."""
    documented = 0
    total = 0

    def walk(n):
        nonlocal documented, total
        if n.type in PUBLIC_ITEM_TYPES and _has_visibility_pub(n, src):
            total += 1
            if _has_doc_comment_before(n, src, root):
                documented += 1
        for c in n.children:
            walk(c)
    walk(root)
    return documented, total


# ---------- R2: constructors on pub structs/enums with private fields ----------

def _struct_has_private_field(item, src: bytes) -> bool:
    """True iff `struct_item` has a field_declaration_list with at least
    one field lacking a visibility_modifier (private to the crate)."""
    if item.type != "struct_item":
        return False
    fdl = item.child_by_field_name("body")
    if fdl is None:
        # named fallback
        for c in item.children:
            if c.type in ("field_declaration_list", "ordered_field_declaration_list"):
                fdl = c
                break
    if fdl is None:
        return False
    for c in fdl.children:
        if c.type == "field_declaration":
            is_pub = any(ch.type == "visibility_modifier" and
                         _text(ch, src).startswith("pub") for ch in c.children)
            if not is_pub:
                return True
    return False


CONSTRUCTOR_NAME_PREFIXES = ("new", "with_", "from_", "builder", "default",
                             "try_new", "try_from", "build")


def _impl_constructor_names(impl_item, src: bytes) -> List[str]:
    """Return the list of pub fn names defined in this impl block."""
    body = impl_item.child_by_field_name("body")
    if body is None:
        for c in impl_item.children:
            if c.type == "declaration_list":
                body = c
                break
    if body is None:
        return []
    names = []
    for c in body.children:
        if c.type == "function_item" and _has_visibility_pub(c, src):
            nm = _name_of(c, src)
            if nm:
                names.append(nm)
    return names


def _impl_type_name(impl_item, src: bytes) -> Optional[str]:
    """For `impl T { ... }` return text of T. For `impl Trait for T` return T."""
    # tree-sitter-rust field 'type' is the target type; 'trait' is trait
    t = impl_item.child_by_field_name("type")
    if t is not None:
        return _text(t, src)
    # fallback heuristic
    for c in impl_item.children:
        if c.type in ("type_identifier", "generic_type", "scoped_type_identifier"):
            return _text(c, src)
    return None


def _r2_constructors(root, src: bytes) -> Tuple[int, int]:
    """Pub structs with private fields need a constructor.

    We collect pub-struct names with at least one private field, then
    check whether the file contains either:
      - an `impl T` block exposing a fn whose name matches a constructor
        prefix, OR
      - an `impl Default for T` block.
    """
    structs_needing_ctor: List[str] = []
    impl_funcs: Dict[str, List[str]] = {}
    impl_default: set = set()

    def walk(n):
        if (n.type == "struct_item" and _has_visibility_pub(n, src)
                and _struct_has_private_field(n, src)):
            nm = _name_of(n, src)
            if nm:
                structs_needing_ctor.append(nm)
        if n.type == "impl_item":
            tname = _impl_type_name(n, src)
            trait_ = n.child_by_field_name("trait")
            if tname is not None:
                if trait_ is not None and _text(trait_, src) == "Default":
                    impl_default.add(tname)
                elif trait_ is None:
                    # inherent impl
                    impl_funcs.setdefault(tname, []).extend(
                        _impl_constructor_names(n, src))
        for c in n.children:
            walk(c)
    walk(root)

    sat = 0
    for s in structs_needing_ctor:
        funcs = impl_funcs.get(s, [])
        has_ctor = any(any(fn == p or fn.startswith(p)
                           for p in CONSTRUCTOR_NAME_PREFIXES)
                       for fn in funcs)
        if has_ctor or s in impl_default:
            sat += 1
    return sat, len(structs_needing_ctor)


# ---------- R3: macro export hygiene ----------

def _r3_macro_hygiene(root, src: bytes) -> Tuple[int, int]:
    """Top-level macro_rules!: either #[macro_export] above, or in a
    private (non-pub) mod (we approximate: only count top-level — at the
    `source_file` root — and require macro_export there)."""
    sat = 0
    total = 0
    for c in root.children:
        if c.type == "macro_definition":
            total += 1
            if _has_attribute(c, src, lambda t: "macro_export" in t):
                sat += 1
    return sat, total


# ---------- R4: feature gate correctness ----------

def _r4_feature_gates(root, src: bytes) -> Tuple[int, int]:
    """Count #[cfg(...)] attributes whose enclosed item is `pub`-visible.
    A pub item gated by a *feature* predicate is idiomatic; gating a pub
    item on `test`/`debug_assertions` (without an accompanying feature
    predicate) flips API surface on build mode, which is the antipattern.
    """
    sat = 0
    total = 0

    def walk(n):
        nonlocal sat, total
        if n.type in PUBLIC_ITEM_TYPES and _has_visibility_pub(n, src):
            # Find cfg attributes on this item
            parent = n.parent
            if parent is None:
                for c in n.children: walk(c)
                return
            siblings = list(parent.children)
            idx = _sibling_index(n)
            if idx < 0:
                for c in n.children: walk(c)
                return
            j = idx - 1
            cfg_texts: List[str] = []
            while j >= 0 and siblings[j].type in (
                    "attribute_item", "line_comment", "block_comment"):
                if siblings[j].type == "attribute_item":
                    t = _text(siblings[j], src)
                    if "cfg(" in t or "cfg_attr(" in t:
                        cfg_texts.append(t)
                j -= 1
            for t in cfg_texts:
                total += 1
                has_feature = 'feature' in t and '=' in t
                # antipattern: cfg(test) or cfg(not(test)) on a pub item
                bad = ('cfg(test)' in t.replace(" ", "") or
                       'cfg(not(test))' in t.replace(" ", "") or
                       'debug_assertions' in t)
                if has_feature and not bad:
                    sat += 1
                elif has_feature:
                    # feature predicate present — accept even if combined
                    sat += 1
                # else: not satisfied
        for c in n.children:
            walk(c)
    walk(root)
    return sat, total


# ---------- R5: pub use re-export organization ----------

def _r5_pub_use_organization(root, src: bytes) -> Tuple[int, int]:
    """For each top-level `pub use ...;`:
      OK iff path is `crate::`, `self::`, `super::`, OR contains `as `
           (renaming), OR is NOT a glob (`*`) import.
      BAD iff it's a glob re-export of an *external* (non crate/self/super)
           crate root.
    """
    sat = 0
    total = 0

    def walk(n):
        nonlocal sat, total
        if n.type == "use_declaration":
            if _has_visibility_pub(n, src):
                total += 1
                txt = _text(n, src)
                stripped = txt.replace(" ", "").replace("\n", "")
                intra = (stripped.startswith("pubusecrate::") or
                         stripped.startswith("pubuseself::") or
                         stripped.startswith("pubusesuper::"))
                has_rename = " as " in txt
                is_glob = "::*" in stripped or stripped.endswith("*;")
                if intra or has_rename or not is_glob:
                    sat += 1
        for c in n.children:
            walk(c)
    walk(root)
    return sat, total


# ---------- per-file scoring ----------

def _score_file(code: bytes) -> Tuple[Optional[float], int]:
    """Return (score, n_applicable_rules). Score is None when no rule
    fires (no applicable items at all)."""
    p = _parser()
    if p is None:
        return None, 0
    root = p.parse(code).root_node
    if root.has_error and len(code) < 50:
        # very short fragments — skip
        return None, 0

    rules: List[Tuple[int, int]] = [
        _r1_pub_items_documented(root, code),
        _r2_constructors(root, code),
        _r3_macro_hygiene(root, code),
        _r4_feature_gates(root, code),
        _r5_pub_use_organization(root, code),
    ]
    per_rule_scores: List[float] = []
    for sat, total in rules:
        if total > 0:
            per_rule_scores.append(sat / total)
    if not per_rule_scores:
        return None, 0
    return sum(per_rule_scores) / len(per_rule_scores), len(per_rule_scores)


# ---------- cheap pre-filter ----------

def _file_looks_rust(body: str) -> bool:
    """Quick test: file has at least one `fn `, `struct `, `pub `, `mod `,
    `impl `, `trait `, or `use ` token on a line. Skips trivially empty
    added-line sets."""
    if not body or len(body) < 4:
        return False
    HINTS = ("fn ", "struct ", "enum ", "trait ", "impl ", "pub ",
             "mod ", "use ", "macro_rules!", "#[")
    return any(h in body for h in HINTS)


def _candidate_files(diff_text: str) -> Dict[str, str]:
    by_path = added_files_by_ext(diff_text, RS_EXTS)
    return {p: c for p, c in by_path.items() if _file_looks_rust(c)}


# ---------- contract ----------

def applies(diff_text: str) -> bool:
    """True iff the diff adds at least one `.rs` file with Rust-shaped
    content. We do NOT pre-parse here; we rely on extension + cheap
    keyword pre-filter to keep `applies()` subprocess-free."""
    return bool(_candidate_files(diff_text))


def score(diff_text: str) -> Optional[float]:
    by_path = _candidate_files(diff_text)
    if not by_path:
        return None
    if _parser() is None:
        return None
    file_scores: List[float] = []
    for _path, content in by_path.items():
        s, _n = _score_file(content.encode("utf8", errors="replace"))
        if s is not None:
            file_scores.append(s)
    if not file_scores:
        return None
    return float(sum(file_scores) / len(file_scores))
