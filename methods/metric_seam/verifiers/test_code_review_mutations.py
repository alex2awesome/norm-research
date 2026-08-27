from __future__ import annotations

import pytest

from methods.metric_seam.verifiers.code_review_mutations import (
    MutationError,
    bare_debug_python,
    build_train_violation_pair,
    swallowed_error_go,
    validate_pair,
)


NATURAL = """diff --git a/main.go b/main.go
index 1111111..2222222 100644
--- a/main.go
+++ b/main.go
@@ -1 +1,2 @@
 package main
+func main() {}
"""


def test_build_pair_is_deterministic_and_isolated():
    kwargs = dict(
        item_key="train-1",
        unit_id="unit-error-handling",
        mutation_kind="swallowed_error",
        source_lines=swallowed_error_go(),
        extension="go",
    )
    first = build_train_violation_pair(NATURAL, **kwargs)
    second = build_train_violation_pair(NATURAL, **kwargs)
    assert first == second
    assert first.planted_violated.startswith(NATURAL)
    assert first.manifest.path.endswith("_test.go")
    assert first.manifest.line_start == 1
    assert first.manifest.line_end == len(swallowed_error_go())
    validate_pair(first)


def test_probe_source_is_ordinary_and_parseable_python():
    pair = build_train_violation_pair(
        NATURAL,
        item_key="train-2",
        unit_id="unit-observability",
        mutation_kind="bare_debug",
        source_lines=bare_debug_python(),
        extension="py",
    )
    assert "metric_seam" not in pair.planted_violated
    assert "probe" not in pair.manifest.path
    compile("\n".join(bare_debug_python()), pair.manifest.path, "exec")
    validate_pair(pair)


@pytest.mark.parametrize("extension", ["md", "exe", ""])
def test_rejects_unsupported_extensions(extension):
    with pytest.raises(MutationError):
        build_train_violation_pair(
            NATURAL,
            item_key="train-3",
            unit_id="unit",
            mutation_kind="bad",
            source_lines=("x",),
            extension=extension,
        )


def test_rejects_non_diff_and_multiline_source():
    with pytest.raises(MutationError):
        build_train_violation_pair(
            "plain source",
            item_key="train-4",
            unit_id="unit",
            mutation_kind="bad",
            source_lines=("x",),
            extension="py",
        )


def test_validate_pair_rejects_manifest_that_misstates_appended_block():
    pair = build_train_violation_pair(
        NATURAL,
        item_key="train-5",
        unit_id="unit",
        mutation_kind="bare_debug",
        source_lines=bare_debug_python(),
        extension="py",
    )
    from dataclasses import replace

    tampered = replace(
        pair,
        manifest=replace(pair.manifest, appended_block_sha256="0" * 64),
    )
    with pytest.raises(MutationError, match="appended block digest"):
        validate_pair(tampered)
    with pytest.raises(MutationError):
        build_train_violation_pair(
            NATURAL,
            item_key="train-4",
            unit_id="unit",
            mutation_kind="bad",
            source_lines=("x\ny",),
            extension="py",
        )
