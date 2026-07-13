from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from methods.metric_seam.hybrids.ops_code import (
    CodeOps, extract_symbols, is_test_path, parse_unified_diff,
    test_design_profile as build_test_design_profile,
)


DIFF = """diff --git a/src/calc.py b/src/calc.py
--- a/src/calc.py
+++ b/src/calc.py
@@ -1,2 +1,5 @@ def add(a, b):
+def multiply(a, b):
+    return a * b
diff --git a/tests/test_calc.py b/tests/test_calc.py
--- a/tests/test_calc.py
+++ b/tests/test_calc.py
@@ -1,2 +1,6 @@
+from src.calc import multiply
+def test_multiply():
+    assert multiply(2, 3) == 6
"""


def test_structured_diff_parser_preserves_files_hunks_and_line_numbers():
    doc = parse_unified_diff(DIFF)
    assert [f.path for f in doc.files] == ["src/calc.py", "tests/test_calc.py"]
    assert doc.files[0].hunks[0].section == "def add(a, b):"
    assert doc.files[0].hunks[0].lines[0].new_lineno == 1
    assert doc.files[0].added_lines == ["def multiply(a, b):", "    return a * b"]


def test_truncated_tail_is_not_silently_attached_to_head_file():
    doc = parse_unified_diff(DIFF.split("diff --git", 1)[0] + "\n[...]\n+orphan()")
    assert doc.truncated
    assert doc.orphan_fragments == ["+orphan()"]


def test_test_path_classifier_has_token_boundaries():
    assert is_test_path("src/FooTest.java")
    assert is_test_path("lib/foo_test.go")
    assert is_test_path("tests/foo.py")
    assert not is_test_path("src/contest.py")


def test_tree_sitter_symbols_and_test_to_source_relation():
    doc = parse_unified_diff(DIFF)
    symbols = {s.qualified_name for s in extract_symbols(doc)}
    assert "src/calc.py::multiply" in symbols
    assert "tests/test_calc.py::test_multiply" in symbols
    profile = build_test_design_profile(DIFF)
    assert profile["presence"] == 1.0
    assert profile["correspondence"] == 1.0
    assert profile["assertions"] == 1
    assert profile["test_to_source_edges"][0]["evidence"] == [
        "ast_identifier_reference", "file_stem_relation", "symbol_name_relation"
    ]


def test_unrelated_test_does_not_cross_function_wall():
    unrelated = DIFF.replace("multiply(2, 3) == 6", "unrelated() is None").replace(
        "from src.calc import multiply", "from src.other import unrelated"
    ).replace("test_multiply", "test_other")
    profile = build_test_design_profile(unrelated)
    # Matching file paths are retained as weaker evidence, while the stronger
    # symbol relationship is absent.
    assert profile["correspondence"] == 0.0
    assert profile["test_to_source_edges"] == []


def test_active_a104_h0_rewards_functional_correspondence():
    path = Path(__file__).parent / "programs_code_review/a104_h0.py"
    spec = spec_from_file_location("active_code_a104", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    good = module.score(DIFF, {}, CodeOps())
    no_tests = module.score(DIFF.split("diff --git a/tests", 1)[0], {}, CodeOps())
    assert module.PROGRAM_PROVENANCE.startswith("active_code_review_census")
    assert good > no_tests
    assert good > 0.8
