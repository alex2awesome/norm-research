from methods.existing_metrics_runner.coded.sandbox import parse_diff_added_by_file


def test_binary_block_does_not_discard_neighboring_text_hunk():
    diff = """diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -0,0 +1,2 @@
+def answer():
+    return 42
diff --git a/image.jpg b/image.jpg
new file mode 100644
index 0000000..1111111
GIT binary patch
literal 10
zabc123
"""
    assert parse_diff_added_by_file(diff) == {
        "x.py": "def answer():\n    return 42"
    }


def test_pure_binary_patch_is_relation_noncoverage_not_an_exception():
    diff = """diff --git a/image.jpg b/image.jpg
new file mode 100644
index 0000000..1111111
GIT binary patch
literal 10
zabc123
"""
    assert parse_diff_added_by_file(diff) == {}
