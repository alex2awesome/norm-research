from __future__ import annotations

import pytest

from methods.existing_metrics_runner.coded.metrics import (
    a35_information_hiding as a35,
    a72_formatting_layout as a72,
    a181_warnings_as_errors as a181,
    a309_test_source_correspondence as a309,
)


def _diff(path: str, content: str) -> str:
    lines = content.splitlines()
    body = "".join(f"+{line}\n" for line in lines)
    return (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
        f"{body}"
    )


def test_a35_final_class_attribute_is_not_a_mutable_public_field() -> None:
    final = _diff(
        "model.py", "class C:\n    value: Final[int] = 1\n    def read(self):\n        return self.value"
    )
    mutable = _diff(
        "model.py", "class C:\n    value: int = 1\n    def read(self):\n        return self.value"
    )
    assert a35.score(final) == 1.0
    assert a35.score(mutable) == 0.25


def test_a72_restores_projection_newline_and_separates_clean_from_dirty() -> None:
    clean = _diff("clean.py", "def f():\n    return 1")
    dirty = _diff("dirty.py", "def f( ):\n return 1")
    assert a72.score(clean) == 1.0
    assert a72.score(dirty) == 0.0


def test_a72_uses_actual_added_line_weights() -> None:
    clean_body = "def f():\n" + "\n".join(f"    x{i} = {i}" for i in range(20))
    mixed = _diff("large_clean.py", clean_body) + _diff(
        "tiny_dirty.py", "def g( ):\n return 1"
    )
    assert a72.score(mixed) == pytest.approx(21 / 23)


def test_a181_does_not_count_ruff_success_summary_as_a_violation() -> None:
    clean = _diff("clean.py", "def f(x: int) -> int:\n    return x + 1")
    bad = _diff("bad.py", "import os\n\ndef f():\n    x = 1\n    return 2")
    assert a181.score(clean) == 1.0
    assert a181.score(bad) is not None
    assert a181.score(bad) < 1.0


def test_a309_test_only_change_has_no_source_correspondence_denominator() -> None:
    test_only = _diff("tests/test_widget.py", "def test_widget():\n    assert True")
    assert a309.applies(test_only) is False
    assert a309.score(test_only) is None
