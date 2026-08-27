import pytest

from methods.metric_seam.verifiers.diff_lines import (
    DiffAddressError,
    added_line_index,
    address_is_visible,
    parse_new_side_lines,
    validate_verdict_addresses,
    visible_line_index,
)
from methods.metric_seam.verifiers.schema import Span, Verdict


DIFF = """diff --git a/a.py b/a.py
index 1111111..2222222 100644
--- a/a.py
+++ b/a.py
@@ -10,4 +10,5 @@ def f():
 context
-old
+new
+more
 tail
diff --git a/new.go b/new.go
new file mode 100644
--- /dev/null
+++ b/new.go
@@ -0,0 +1,2 @@
+package demo
+func f() {}
"""


def test_parse_addresses_context_and_additions():
    rows = parse_new_side_lines(DIFF)
    assert [(r.path, r.line, r.text, r.added) for r in rows] == [
        ("a.py", 10, "context", False),
        ("a.py", 11, "new", True),
        ("a.py", 12, "more", True),
        ("a.py", 13, "tail", False),
        ("new.go", 1, "package demo", True),
        ("new.go", 2, "func f() {}", True),
    ]
    assert visible_line_index(DIFF)["a.py"] == {10, 11, 12, 13}
    assert added_line_index(DIFF)["a.py"] == {11, 12}


def test_span_visibility_is_path_sensitive_and_requires_contiguity():
    index = visible_line_index(DIFF)
    assert address_is_visible(index, "a.py", 10, 13)
    assert address_is_visible(index, "new.go", 1, 2)
    assert not address_is_visible(index, "missing.py", 1, 2)
    assert not address_is_visible(index, "a.py", 11, 14)


def test_verdict_witnesses_are_bound_to_item_addresses():
    validate_verdict_addresses(
        DIFF,
        Verdict(True, True, (Span("a.py", 11, 12),)),
        require_added=True,
    )
    with pytest.raises(DiffAddressError):
        validate_verdict_addresses(
            DIFF,
            Verdict(True, True, (Span("a.py", 10, 10),)),
            require_added=True,
        )
    with pytest.raises(DiffAddressError):
        validate_verdict_addresses(
            DIFF,
            Verdict(True, True, (Span("new.go", 10, 10),)),
        )


@pytest.mark.parametrize("text", ["", "plain source", "diff --git a/x b/x\n"])
def test_rejects_non_addressable_inputs(text):
    with pytest.raises(DiffAddressError):
        parse_new_side_lines(text)
