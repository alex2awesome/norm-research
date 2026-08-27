"""TRAIN-only, append-only mutation helpers for code-review verifier probes.

The helpers add an ordinary-looking source file to a unified diff.  They never
rewrite the natural item, which makes mutation isolation mechanically
checkable: the original bytes must remain an exact prefix of the mutant.  The
generated cases are diagnostics and must never be mixed into held-out data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Sequence


_SAFE_PATH = re.compile(r"^[A-Za-z0-9_./-]+$")
_EXTENSIONS = {"go", "py", "java", "js", "ts"}


class MutationError(ValueError):
    """Raised when a requested mutation would be ambiguous or unsafe."""


@dataclass(frozen=True)
class MutationManifest:
    unit_id: str
    split: str
    mutation_kind: str
    path: str
    line_start: int
    line_end: int
    original_sha256: str
    mutant_sha256: str
    appended_block_sha256: str
    original_is_exact_prefix: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MutationPair:
    """A natural TRAIN item paired with one planted violated item."""

    natural: str
    planted_violated: str
    manifest: MutationManifest


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_path(path: str) -> None:
    if not path or path.startswith(('/', '../')) or '/..' in path:
        raise MutationError("probe path must be repository-relative")
    if not _SAFE_PATH.fullmatch(path):
        raise MutationError("probe path contains unsupported characters")
    suffix = path.rsplit('.', 1)[-1] if '.' in path.rsplit('/', 1)[-1] else ''
    if suffix not in _EXTENSIONS:
        raise MutationError(f"unsupported probe extension: {suffix or '<none>'}")


def _new_file_block(path: str, lines: Sequence[str]) -> str:
    _validate_path(path)
    if not lines:
        raise MutationError("a planted file must contain at least one line")
    if any('\n' in line or '\r' in line for line in lines):
        raise MutationError("source lines must not contain newline characters")
    body = ''.join(f"+{line}\n" for line in lines)
    blob = hashlib.sha256(('\n'.join(lines) + '\n').encode('utf-8')).hexdigest()[:7]
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        f"index 0000000..{blob}\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def build_train_violation_pair(
    natural: str,
    *,
    item_key: str,
    unit_id: str,
    mutation_kind: str,
    source_lines: Sequence[str],
    extension: str,
) -> MutationPair:
    """Append one planted source file and return its load-bearing witness span.

    The path is derived from the frozen item/unit identity but does not disclose
    probe mode or the unit name.  Callers must keep the returned pair on TRAIN.
    """

    if not natural.startswith("diff --git "):
        raise MutationError("code-review item is not a unified git diff")
    if extension not in _EXTENSIONS:
        raise MutationError(f"unsupported probe extension: {extension}")
    if not item_key or not unit_id or not mutation_kind:
        raise MutationError("item_key, unit_id, and mutation_kind are required")
    token = hashlib.sha256(f"{item_key}\0{unit_id}".encode('utf-8')).hexdigest()[:12]
    path = f"internal/validation_{token}_test.{extension}"
    if f"+++ b/{path}\n" in natural:
        raise MutationError("derived probe path already exists in natural item")
    block = _new_file_block(path, tuple(source_lines))
    separator = '' if natural.endswith('\n') else '\n'
    mutant = natural + separator + block
    manifest = MutationManifest(
        unit_id=unit_id,
        split="compiler_train",
        mutation_kind=mutation_kind,
        path=path,
        line_start=1,
        line_end=len(source_lines),
        original_sha256=_sha256(natural),
        mutant_sha256=_sha256(mutant),
        appended_block_sha256=_sha256(block),
        original_is_exact_prefix=mutant.startswith(natural),
    )
    return MutationPair(natural=natural, planted_violated=mutant, manifest=manifest)


def validate_pair(pair: MutationPair) -> None:
    """Fail closed if a pair is not an isolated TRAIN-side append mutation."""

    manifest = pair.manifest
    if manifest.split != "compiler_train":
        raise MutationError("planted mutations are TRAIN-only")
    if not pair.planted_violated.startswith(pair.natural):
        raise MutationError("natural bytes were rewritten")
    if _sha256(pair.natural) != manifest.original_sha256:
        raise MutationError("natural digest drift")
    if _sha256(pair.planted_violated) != manifest.mutant_sha256:
        raise MutationError("mutant digest drift")
    separator = "" if pair.natural.endswith("\n") else "\n"
    appended = pair.planted_violated[len(pair.natural) + len(separator) :]
    if _sha256(appended) != manifest.appended_block_sha256:
        raise MutationError("appended block digest drift")
    if manifest.line_start != 1 or manifest.line_end < manifest.line_start:
        raise MutationError("manifest witness span is invalid")
    marker = f"+++ b/{manifest.path}\n"
    if pair.planted_violated.count(marker) != 1:
        raise MutationError("planted path is missing or ambiguous")
    if not manifest.original_is_exact_prefix:
        raise MutationError("manifest does not attest exact-prefix isolation")


def swallowed_error_go() -> tuple[str, ...]:
    """Parseable Go anti-pattern for structured error-handling controls."""

    return (
        "package internal",
        "",
        "func validateInput() error {",
        "\terr := runValidation()",
        "\tif err != nil {",
        "\t\t_ = err",
        "\t\treturn nil",
        "\t}",
        "\treturn nil",
        "}",
    )


def bare_debug_python() -> tuple[str, ...]:
    """Parseable Python anti-pattern for observability controls."""

    return (
        "def validate_input(payload):",
        "    print(payload)",
        "    return payload",
    )
