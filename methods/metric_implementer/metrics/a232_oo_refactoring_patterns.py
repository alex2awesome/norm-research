"""a232: Refactoring patterns (Fowler/OO) — PARTIALLY_THIN class-level detector.

The norm (Fowler-style, OO-flavored): "Apply standard refactorings (e.g.,
Remove/Introduce Parameter or Method, Replace Constructor with Factory
Method, Replace Inheritance with Delegation) to improve design without
changing behavior."

Relation to a80 (Refactoring techniques and application) and a37
(Refactoring quality and practice):

  * a37 measures the *shape* of the diff (add/remove balance, edits-existing
    fraction, anti-cosmetic churn). Whole-diff geometry. No catalog.

  * a80 detects FUNCTION-level Fowler refactorings (Extract Method, Inline
    Function, Rename, Decompose Conditional, Introduce Parameter Object).
    Python-only, operates over function defs and their call sites.

  * a232 (this metric) targets CLASS-level / OO-specific Fowler
    refactorings that a80 cannot see because a80 never looks at class
    headers, base classes, decorators, or cross-class method movement.

The five OO refactorings detected here, with their structural signatures:

  R1. REPLACE CONSTRUCTOR WITH FACTORY METHOD.
      A class that previously declared only `__init__` / a public Java
      constructor gains a new `@classmethod`/`@staticmethod` (Python) or
      `public static Foo of/from/create/newInstance(...)` (Java) method
      that returns an instance.

  R2. REPLACE INHERITANCE WITH DELEGATION.
      Python: `class Foo(Bar):` becomes `class Foo:` AND an `__init__`
      added line introduces `self.<x> = Bar(...)`.
      Java:   `class Foo extends Bar` becomes `class Foo` (no extends), AND
      a field of type Bar (`private Bar bar;` / `this.bar = new Bar(`)
      appears in the post-state.

  R3. EXTRACT CLASS.
      A new `class X:` (Python) / `class X { ... }` (Java) is added in a
      file that already had at least one class in the pre-state, AND the
      pre-existing class loses at least one method.

  R4. PULL UP / PUSH DOWN METHOD.
      The same method name `def m(...)` is removed from one class body and
      added to a different class body in the same file. (Direction —
      pull-up vs push-down — needs the base/subclass relation; we don't
      disambiguate, both are valid named refactorings.)

  R5. REPLACE CONDITIONAL WITH POLYMORPHISM.
      Pre-state contained an `if isinstance(x, A): ... elif isinstance(x, B):`
      chain (Python) or `if (x instanceof A) ... else if (x instanceof B)`
      (Java) of length >= 2; post-state removes it AND adds a new subclass
      or an overriding method.

Score: number of distinct OO refactorings detected divided by
the maximum reasonable count for a single PR (we cap at 4). A diff that
applies one of these patterns crisply scores ~0.5; multiple patterns
push it toward 1.0. A diff that edits OO files but applies none of these
named OO refactorings scores 0.0 — i.e. the OO refactoring norm is "not
demonstrated" here.

Classification: PARTIALLY_THIN.

Why not THIN:
  * Signature-only detection: a new factory method could be a brand-new
    feature, not a refactor of an existing constructor. We have no
    behaviour-equivalence check.
  * Cross-file Move Method / Move Class is not detected (the diff slice
    is one file at a time, like a80).
  * Java-side instanceof detection is coarse: we only see the diff slice,
    not full class hierarchies.
  * Delegation detection cannot verify that the new field type matches
    the old superclass behaviour; we only check the name.

Why not THICK:
  * Each of R1, R2, R3, R4 has a crisp local structural signature that a
    parser can identify. The detector is genuinely measuring "did this
    diff perform an OO-named refactoring", not text length or churn.

`applies()` requires the diff to touch at least one Python or Java file
that contains a `class` declaration with both added and removed lines —
so pre/post comparison at the class level is possible.

Subsumption check: this metric and a80 are designed not to double-count.
a80 fires on Python function-level diffs; a232 fires on Python OR Java
class-level diffs and looks at class headers, decorators, base classes,
and cross-class method movement. The two return distinct floats in our
fixture sample (verified by the runner).
"""
from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

import whatthepatch

ASPECT_ID = "a232"
ASPECT_NAME = "Refactoring patterns (Fowler/OO)"
TIER = 2
TOOLS = []  # Python stdlib `ast` + whatthepatch + tree-sitter-java
APPLIES_TO_LANGS = ["Python", "Java"]
CLASSIFICATION = "PARTIALLY_THIN"

PY_EXTS = (".py", ".pyi")
JAVA_EXTS = (".java",)

# ---------------------------------------------------------------------------
# Pre/post reconstruction from a unified diff slice
# ---------------------------------------------------------------------------


def _parse_pre_post(diff_text: str) -> Dict[str, Tuple[str, str, str]]:
    """{path: (pre_text, post_text, lang)} for Python and Java files.

    Both pre_text and post_text are the diff's visible context plus the
    removed-only or added-only lines. They are slices, not full files,
    which is fine because every detector we run is local to the slice.
    """
    idx = diff_text.find("diff --git")
    if idx == -1:
        return {}
    try:
        diffs = list(whatthepatch.parse_patch(diff_text[idx:]))
    except Exception:
        return {}
    out: Dict[str, Tuple[List[str], List[str], str]] = {}
    for d in diffs:
        if d is None:
            continue
        new_path = d.header.new_path or ""
        old_path = d.header.old_path or ""
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        if old_path.startswith("a/"):
            old_path = old_path[2:]
        path = new_path or old_path
        if not path or path == "/dev/null":
            continue
        low = path.lower()
        if any(low.endswith(e) for e in PY_EXTS):
            lang = "python"
        elif any(low.endswith(e) for e in JAVA_EXTS):
            lang = "java"
        else:
            continue
        if new_path == "/dev/null" or old_path == "/dev/null":
            continue  # pure create or pure delete: no refactor signal
        pre: List[str] = []
        post: List[str] = []
        for ch in (d.changes or []):
            if ch.line is None:
                continue
            if ch.old is not None and ch.new is not None:
                pre.append(ch.line)
                post.append(ch.line)
            elif ch.new is None and ch.old is not None:
                pre.append(ch.line)
            elif ch.old is None and ch.new is not None:
                post.append(ch.line)
        entry = out.setdefault(path, ([], [], lang))
        entry[0].extend(pre)
        entry[1].extend(post)
    return {
        p: ("\n".join(pre), "\n".join(post), lang)
        for p, (pre, post, lang) in out.items()
    }


# ---------------------------------------------------------------------------
# Python: AST-based class structure summary
# ---------------------------------------------------------------------------


def _safe_py(src: str) -> Optional[ast.AST]:
    if not src.strip():
        return None
    try:
        return ast.parse(src)
    except SyntaxError:
        # Fall back: strip until first class/def
        lines = src.split("\n")
        for i, line in enumerate(lines):
            s = line.lstrip()
            if s.startswith(("class ", "def ", "async def ")):
                try:
                    indent = len(line) - len(s)
                    fixed = "\n".join(
                        (l[indent:] if len(l) >= indent else l)
                        for l in lines[i:]
                    )
                    return ast.parse(fixed)
                except SyntaxError:
                    continue
        return None


def _decorator_names(node: ast.AST) -> Set[str]:
    out: Set[str] = set()
    for d in getattr(node, "decorator_list", []) or []:
        if isinstance(d, ast.Name):
            out.add(d.id)
        elif isinstance(d, ast.Attribute):
            out.add(d.attr)
        elif isinstance(d, ast.Call):
            f = d.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def _py_class_summary(tree: ast.AST) -> Dict[str, dict]:
    """{class_name: {bases, methods, factory_methods, instanceof_chains,
                      delegate_fields_init_to}}."""
    out: Dict[str, dict] = {}
    if tree is None:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        name = node.name
        bases: List[str] = []
        for b in node.bases:
            if isinstance(b, ast.Name):
                bases.append(b.id)
            elif isinstance(b, ast.Attribute):
                bases.append(b.attr)
        methods: Set[str] = set()
        factory_methods: List[str] = []
        instanceof_chains = 0
        delegate_targets: Set[str] = set()
        for sub in node.body:
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.add(sub.name)
                decos = _decorator_names(sub)
                if "classmethod" in decos or "staticmethod" in decos:
                    # Does it return an instance of this class?
                    is_factory = False
                    for ret in ast.walk(sub):
                        if isinstance(ret, ast.Return) and ret.value is not None:
                            v = ret.value
                            if isinstance(v, ast.Call):
                                f = v.func
                                callee = ""
                                if isinstance(f, ast.Name):
                                    callee = f.id
                                elif isinstance(f, ast.Attribute):
                                    callee = f.attr
                                if callee in ("cls", name):
                                    is_factory = True
                                    break
                    if (is_factory or sub.name.lower().startswith(
                            ("from_", "of_", "create_", "make_", "new_"))
                            or sub.name in ("of", "from", "create", "make")):
                        factory_methods.append(sub.name)
                # __init__ body: look for self.<x> = <Base>(...) — delegate
                if sub.name == "__init__":
                    for stmt in ast.walk(sub):
                        if isinstance(stmt, ast.Assign):
                            for tgt in stmt.targets:
                                if (isinstance(tgt, ast.Attribute)
                                        and isinstance(tgt.value, ast.Name)
                                        and tgt.value.id == "self"
                                        and isinstance(stmt.value, ast.Call)):
                                    f = stmt.value.func
                                    callee = ""
                                    if isinstance(f, ast.Name):
                                        callee = f.id
                                    elif isinstance(f, ast.Attribute):
                                        callee = f.attr
                                    if callee:
                                        delegate_targets.add(callee)
                # Count isinstance() chains inside the method
                for sub2 in ast.walk(sub):
                    if isinstance(sub2, ast.If):
                        chain_len = _isinstance_chain_len(sub2)
                        if chain_len >= 2:
                            instanceof_chains += 1
        out[name] = dict(
            bases=bases,
            methods=methods,
            factory_methods=factory_methods,
            instanceof_chains=instanceof_chains,
            delegate_targets=delegate_targets,
        )
    return out


def _isinstance_chain_len(if_node: ast.If) -> int:
    """How many isinstance-condition branches the if/elif chain has."""
    n = 0
    cur: Optional[ast.AST] = if_node
    seen = 0
    while isinstance(cur, ast.If) and seen < 20:
        cond = cur.test
        # Allow `isinstance(x, A)` or boolean-ored variants
        if _expr_has_isinstance(cond):
            n += 1
        if cur.orelse and len(cur.orelse) == 1 and isinstance(cur.orelse[0],
                                                              ast.If):
            cur = cur.orelse[0]
        else:
            break
        seen += 1
    return n


def _expr_has_isinstance(expr: ast.AST) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, ast.Call):
            f = node.func
            nm = ""
            if isinstance(f, ast.Name):
                nm = f.id
            elif isinstance(f, ast.Attribute):
                nm = f.attr
            if nm == "isinstance":
                return True
    return False


# ---------------------------------------------------------------------------
# Java: tree-sitter class structure summary
# ---------------------------------------------------------------------------

_JAVA_PARSER = None


def _java_parser():
    global _JAVA_PARSER
    if _JAVA_PARSER is not None:
        return _JAVA_PARSER if _JAVA_PARSER is not False else None
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_java as m
        _JAVA_PARSER = Parser(Language(m.language()))
    except Exception:
        _JAVA_PARSER = False
        return None
    return _JAVA_PARSER


def _txt(n) -> str:
    return n.text.decode("utf8", errors="replace") if n is not None else ""


def _java_class_summary(src: str) -> Dict[str, dict]:
    parser = _java_parser()
    if parser is None or not src.strip():
        return {}
    code = src.encode("utf8", errors="replace")
    # Java diff slices may not contain a top-level class wrapper.
    if (b"class " not in code and b"interface " not in code
            and b"enum " not in code and b"record " not in code):
        code = b"class __Snip {\n" + code + b"\n}\n"
    try:
        root = parser.parse(code).root_node
    except Exception:
        return {}
    out: Dict[str, dict] = {}

    def walk(n, in_class=None):
        if n.type in ("class_declaration", "interface_declaration",
                       "enum_declaration", "record_declaration"):
            name_node = n.child_by_field_name("name")
            cname = _txt(name_node) if name_node is not None else "__Anon"
            superclass = n.child_by_field_name("superclass")
            bases: List[str] = []
            if superclass is not None:
                # superclass node holds an identifier child
                base_txt = _txt(superclass).strip()
                if base_txt.startswith("extends"):
                    base_txt = base_txt[len("extends"):].strip()
                bases.append(base_txt.split("<")[0].strip())
            iface_node = n.child_by_field_name("interfaces")
            if iface_node is not None:
                for c in iface_node.children:
                    if c.type == "type_list":
                        for cc in c.children:
                            if cc.type in ("type_identifier",
                                            "generic_type",
                                            "scoped_type_identifier"):
                                bases.append(_txt(cc).split("<")[0].strip())
            body = n.child_by_field_name("body")
            methods: Set[str] = set()
            factory_methods: List[str] = []
            instanceof_chains = 0
            delegate_targets: Set[str] = set()
            has_public_ctor = False
            if body is not None:
                for ch in body.children:
                    if ch.type == "constructor_declaration":
                        has_public_ctor = True
                    elif ch.type == "method_declaration":
                        nm_node = ch.child_by_field_name("name")
                        nm = _txt(nm_node) if nm_node is not None else ""
                        methods.add(nm)
                        # static?
                        is_static = False
                        for c in ch.children:
                            if c.type == "modifiers" and "static" in _txt(c).split():
                                is_static = True
                                break
                        ret_type_node = ch.child_by_field_name("type")
                        ret_txt = _txt(ret_type_node).split("<")[0].strip()
                        if is_static:
                            # Factory if returns the enclosing type or named
                            # canonically.
                            ln = nm.lower()
                            if (ret_txt == cname
                                    or ln in ("of", "from", "create", "make",
                                             "newinstance", "valueof", "parse",
                                             "build")
                                    or any(ln.startswith(p) and len(ln) > len(p)
                                           and nm[len(p):len(p)+1].isupper()
                                           for p in ("create", "from", "of",
                                                     "make", "valueof",
                                                     "parse", "new"))):
                                factory_methods.append(nm)
                        # instanceof chains inside method body
                        body_node = ch.child_by_field_name("body")
                        if body_node is not None:
                            instanceof_chains += _java_count_instanceof_chains(
                                body_node)
                    elif ch.type == "field_declaration":
                        # Capture the field's TYPE name (delegate target).
                        type_node = ch.child_by_field_name("type")
                        if type_node is not None:
                            tname = _txt(type_node).split("<")[0].strip()
                            if tname and tname[0:1].isupper():
                                delegate_targets.add(tname)
            out[cname] = dict(
                bases=bases,
                methods=methods,
                factory_methods=factory_methods,
                instanceof_chains=instanceof_chains,
                delegate_targets=delegate_targets,
                has_public_ctor=has_public_ctor,
            )
        for c in n.children:
            walk(c, in_class)

    walk(root)
    return out


def _java_count_instanceof_chains(body_node) -> int:
    """Count if/else-if chains with >=2 instanceof checks."""
    n = 0

    def visit(n_):
        nonlocal n
        if n_.type == "if_statement":
            chain_len, deepest = 0, n_
            seen = 0
            cur = n_
            while cur is not None and cur.type == "if_statement" and seen < 20:
                cond = cur.child_by_field_name("condition")
                if cond is not None and "instanceof" in _txt(cond):
                    chain_len += 1
                else_node = cur.child_by_field_name("alternative")
                if else_node is None:
                    break
                # alternative may be a block or another if_statement
                nxt = else_node
                if else_node.type == "block":
                    inner_ifs = [c for c in else_node.children
                                 if c.type == "if_statement"]
                    if len(inner_ifs) == 1:
                        nxt = inner_ifs[0]
                if nxt.type == "if_statement":
                    cur = nxt
                else:
                    break
                seen += 1
            if chain_len >= 2:
                n += 1
                return  # don't recurse into already-counted chain
        for c in n_.children:
            visit(c)

    visit(body_node)
    return n


# ---------------------------------------------------------------------------
# OO refactoring detection (per file)
# ---------------------------------------------------------------------------


def _detect_per_file(pre_src: str, post_src: str, lang: str) -> Set[str]:
    """Return the set of detected OO-refactoring labels for this file."""
    detected: Set[str] = set()
    if lang == "python":
        pre_tree = _safe_py(pre_src)
        post_tree = _safe_py(post_src)
        if pre_tree is None and post_tree is None:
            return detected
        pre_cls = _py_class_summary(pre_tree) if pre_tree else {}
        post_cls = _py_class_summary(post_tree) if post_tree else {}
    elif lang == "java":
        pre_cls = _java_class_summary(pre_src)
        post_cls = _java_class_summary(post_src)
    else:
        return detected

    # ----- R1. Replace Constructor with Factory Method -----
    for cname, post_info in post_cls.items():
        pre_info = pre_cls.get(cname)
        if pre_info is None:
            continue
        new_factories = (set(post_info.get("factory_methods", []))
                         - set(pre_info.get("factory_methods", [])))
        if new_factories:
            # The class must already have had a ctor — i.e. an `__init__`
            # method (Python) or a public constructor (Java) in pre-state.
            if lang == "python":
                if "__init__" in pre_info.get("methods", set()):
                    detected.add("R1_factory_method")
            else:
                if pre_info.get("has_public_ctor"):
                    detected.add("R1_factory_method")

    # ----- R2. Replace Inheritance with Delegation -----
    for cname, post_info in post_cls.items():
        pre_info = pre_cls.get(cname)
        if pre_info is None:
            continue
        pre_bases = set(pre_info.get("bases", []))
        post_bases = set(post_info.get("bases", []))
        dropped = pre_bases - post_bases
        if not dropped:
            continue
        post_delegates = set(post_info.get("delegate_targets", set()))
        pre_delegates = set(pre_info.get("delegate_targets", set()))
        new_delegates = post_delegates - pre_delegates
        # A dropped base name now appears as a delegate field/init target.
        if dropped & new_delegates:
            detected.add("R2_delegation")

    # ----- R3. Extract Class -----
    new_classes = set(post_cls) - set(pre_cls)
    kept_classes = set(post_cls) & set(pre_cls)
    if new_classes and kept_classes:
        # Did any kept class lose methods?
        for cname in kept_classes:
            pre_m = pre_cls[cname].get("methods", set())
            post_m = post_cls[cname].get("methods", set())
            if pre_m - post_m:
                detected.add("R3_extract_class")
                break

    # ----- R4. Pull Up / Push Down Method -----
    # method name M removed from class A in pre, added to class B in post,
    # A != B, both classes are present in both states (otherwise this is
    # really R3 or rename).
    removed_methods_by_class: Dict[str, Set[str]] = {}
    added_methods_by_class: Dict[str, Set[str]] = {}
    for cname in kept_classes:
        pre_m = pre_cls[cname].get("methods", set())
        post_m = post_cls[cname].get("methods", set())
        removed_methods_by_class[cname] = pre_m - post_m
        added_methods_by_class[cname] = post_m - pre_m
    moved = False
    for src_cls, removed in removed_methods_by_class.items():
        for m in removed:
            if m in ("__init__", "<init>"):
                continue
            for dst_cls, added in added_methods_by_class.items():
                if dst_cls != src_cls and m in added:
                    moved = True
                    break
            if moved:
                break
        if moved:
            break
    if moved:
        detected.add("R4_pull_push_method")

    # ----- R5. Replace Conditional with Polymorphism -----
    pre_instanceof = sum(info.get("instanceof_chains", 0)
                         for info in pre_cls.values())
    post_instanceof = sum(info.get("instanceof_chains", 0)
                          for info in post_cls.values())
    if pre_instanceof > post_instanceof and pre_instanceof >= 1:
        # subclass added OR new override of an existing method:
        # heuristic 1: new class whose bases overlap with some kept class
        kept = set(post_cls) & set(pre_cls)
        new_subclass = False
        for nc in new_classes:
            for b in post_cls[nc].get("bases", []):
                if b in kept:
                    new_subclass = True
                    break
            if new_subclass:
                break
        # heuristic 2: an existing class had a new override method added
        new_override = False
        for cname in kept_classes:
            pre_m = pre_cls[cname].get("methods", set())
            post_m = post_cls[cname].get("methods", set())
            # method exists post but not pre AND class has bases (potential
            # override)
            if (post_m - pre_m) and pre_cls[cname].get("bases"):
                new_override = True
                break
        if new_subclass or new_override:
            detected.add("R5_polymorphism")

    return detected


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def applies(diff_text: str) -> bool:
    """True iff the diff edits at least one Python or Java source file
    containing a `class` (or `interface`/`enum`/`record`) declaration with
    both added and removed lines."""
    files = _parse_pre_post(diff_text)
    if not files:
        return False
    for path, (pre, post, lang) in files.items():
        if not pre.strip() or not post.strip():
            continue
        combined = pre + "\n" + post
        if lang == "python":
            if "class " in combined:
                return True
        elif lang == "java":
            if any(kw in combined for kw in
                   ("class ", "interface ", "enum ", "record ")):
                return True
    return False


def score(diff_text: str) -> Optional[float]:
    files = _parse_pre_post(diff_text)
    if not files:
        return None
    relevant_files = 0
    all_detected: Set[str] = set()
    for path, (pre, post, lang) in files.items():
        if not pre.strip() or not post.strip():
            continue
        combined = pre + "\n" + post
        if lang == "python" and "class " not in combined:
            continue
        if lang == "java" and not any(
                kw in combined for kw in
                ("class ", "interface ", "enum ", "record ")):
            continue
        relevant_files += 1
        try:
            d = _detect_per_file(pre, post, lang)
        except Exception:
            d = set()
        all_detected |= d
    if relevant_files == 0:
        return None
    # Cap at 4 distinct refactorings (anything beyond saturates).
    n = min(len(all_detected), 4)
    return float(n / 4.0)
