#!/usr/bin/env python3
"""Explore deterministic accept/reject metrics for the mathlib deconfounded slice.

Run on sk3 with:
  ssh sk3 'HOME=/lfs/skampere3/0/alexspan /lfs/skampere3/0/alexspan/envs/ai_usage/bin/python3 -' < scripts/mathlib_accept_reject_new_metrics.py
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = Path("/lfs/skampere3/0/alexspan/norm-research/datasets/math/mathlib")
RANDOM_STATE = 0


def safe_auc(y_true, score) -> float:
    y = np.asarray(y_true)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def logit_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_predict_auc(train_df: pd.DataFrame, eval_df: pd.DataFrame, cols: list[str]) -> tuple[float, np.ndarray, Pipeline]:
    pipe = logit_pipeline()
    pipe.fit(train_df[cols].astype(float), train_df["judgement"].astype(int))
    pred = pipe.predict_proba(eval_df[cols].astype(float))[:, 1]
    return safe_auc(eval_df["judgement"].astype(int), pred), pred, pipe


def topic_resid_auc(train_df: pd.DataFrame, eval_df: pd.DataFrame, pred_train: np.ndarray, pred_eval: np.ndarray) -> float:
    """Residualize labels and predictions on topic area, fit train residual mapping, score eval.

    This matches the notes' "area one-hot partialled out" interpretation: remove the
    train-estimated area mean from both y and model score, then AUC the residual score
    against the residualized label.
    """
    try:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    z_tr = enc.fit_transform(train_df[["area"]].astype(str))
    z_ev = enc.transform(eval_df[["area"]].astype(str))
    y_tr = train_df["judgement"].astype(float).to_numpy()
    y_ev = eval_df["judgement"].astype(float).to_numpy()
    z_tr_i = np.column_stack([np.ones(len(z_tr)), z_tr])
    z_ev_i = np.column_stack([np.ones(len(z_ev)), z_ev])

    # Ridge-stabilized least squares for the redundant full one-hot basis.
    lam = 1e-6
    beta_y = np.linalg.solve(z_tr_i.T @ z_tr_i + lam * np.eye(z_tr_i.shape[1]), z_tr_i.T @ y_tr)
    beta_p = np.linalg.solve(z_tr_i.T @ z_tr_i + lam * np.eye(z_tr_i.shape[1]), z_tr_i.T @ pred_train)
    y_resid = y_ev - z_ev_i @ beta_y
    p_resid = pred_eval - z_ev_i @ beta_p
    return safe_auc((y_resid > 0).astype(int), p_resid)


def partial_topic_resid_auc_eval_only(eval_df: pd.DataFrame, pred_eval: np.ndarray) -> float:
    """Eval-only area demeaning, useful as a robustness check against the train-fit version."""
    y = eval_df["judgement"].astype(float).to_numpy()
    p = np.asarray(pred_eval, dtype=float)
    tmp = pd.DataFrame({"area": eval_df["area"].astype(str).to_numpy(), "y": y, "p": p})
    y_res = np.empty(len(tmp))
    p_res = np.empty(len(tmp))
    for _, idx in tmp.groupby("area").groups.items():
        idx = np.asarray(list(idx))
        y_res[idx] = y[idx] - y[idx].mean()
        p_res[idx] = p[idx] - p[idx].mean()
    return safe_auc((y_res > 0).astype(int), p_res)


def parse_diff_sections(diff: str) -> list[tuple[str, list[str]]]:
    """Return [(path, changed lines)] with raw unified-diff lines for each file."""
    if not isinstance(diff, str):
        return []
    sections: list[tuple[str, list[str]]] = []
    path = ""
    lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if path or lines:
                sections.append((path, lines))
            path = ""
            lines = []
            m = re.match(r"diff --git a/(.*?) b/(.*)", line)
            if m:
                path = m.group(2)
            continue
        if line.startswith("+++ b/"):
            path = line[6:]
            continue
        lines.append(line)
    if path or lines:
        sections.append((path, lines))
    return sections


def added_code_lines(diff: str, lean_only: bool = True) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for path, lines in parse_diff_sections(diff):
        if lean_only and not path.endswith(".lean"):
            continue
        for line in lines:
            if not line.startswith("+") or line.startswith("+++") or line.startswith("+--"):
                continue
            out.append((path, line[1:]))
    return out


def all_changed_lines(diff: str, lean_only: bool = True) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for path, lines in parse_diff_sections(diff):
        if lean_only and not path.endswith(".lean"):
            continue
        for line in lines:
            if line.startswith(("+++", "---")):
                continue
            if line.startswith("+"):
                out.append((path, "+", line[1:]))
            elif line.startswith("-"):
                out.append((path, "-", line[1:]))
    return out


DECL_RE = re.compile(
    r"^\s*(?:@[^\n]*\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+|partial\s+|scoped\s+|local\s+)*"
    r"(theorem|lemma|def|instance|structure|class|abbrev|axiom|opaque|inductive|coinductive|example)\b"
)
NAMED_DECL_RE = re.compile(
    r"^\s*(?:@[^\n]*\s*)*(?:private\s+|protected\s+|noncomputable\s+|unsafe\s+|partial\s+|scoped\s+|local\s+)*"
    r"(theorem|lemma|def|instance|structure|class|abbrev|axiom|opaque|inductive|coinductive)\s+([A-Za-z0-9_'.]+)?"
)
ATTR_RE = re.compile(r"@\[(.*?)\]")
TACTIC_WORDS = [
    "simp",
    "simpa",
    "aesop",
    "grind",
    "omega",
    "ring",
    "linarith",
    "nlinarith",
    "norm_num",
    "positivity",
    "exact",
    "apply",
    "intro",
    "intros",
    "rw",
    "rwa",
    "refine",
    "constructor",
    "cases",
    "rcases",
    "by_cases",
    "by_contra",
    "induction",
    "have",
    "suffices",
    "show",
    "calc",
    "convert",
    "congr",
    "ext",
    "tauto",
    "decide",
    "native_decide",
    "norm_cast",
    "push_cast",
]
TACTIC_RE = re.compile(r"\b(" + "|".join(re.escape(t) for t in TACTIC_WORDS) + r")\b")
MANUAL_REF_RE = re.compile(
    r"\b(eq|Eq|congrArg|congrarg|le_trans|lt_of_le_of_lt|le_antisymm|Subtype\.ext|funext|propext|congr)\b"
)
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_./]+)")


def count_docstrings_before_decls(lines: list[str]) -> tuple[int, int]:
    """Approximate added public declarations with immediately preceding /-- docstrings."""
    total = 0
    documented = 0
    recent_doc = False
    blank_since_doc = 0
    in_doc = False
    for line in lines:
        s = line.strip()
        if s.startswith("/--"):
            in_doc = not ("-/" in s and s.index("/--") < s.rindex("-/"))
            recent_doc = True
            blank_since_doc = 0
            continue
        if in_doc:
            if "-/" in s:
                in_doc = False
            continue
        if s == "":
            if recent_doc:
                blank_since_doc += 1
                if blank_since_doc > 1:
                    recent_doc = False
            continue
        if s.startswith("@["):
            continue
        m = NAMED_DECL_RE.match(line)
        if m:
            kind = m.group(1)
            if kind != "example":
                total += 1
                if recent_doc:
                    documented += 1
            recent_doc = False
            blank_since_doc = 0
        elif not s.startswith("--"):
            recent_doc = False
            blank_since_doc = 0
    return documented, total


def max_indent(lines: list[str]) -> int:
    vals = []
    for line in lines:
        if line.strip():
            vals.append(len(line) - len(line.lstrip(" ")))
    return max(vals) if vals else 0


def extract_new_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for number, diff in zip(df["number"], df["diff_noauth"]):
        added = added_code_lines(diff, lean_only=True)
        changed = all_changed_lines(diff, lean_only=True)
        added_lines = [x for _, x in added]
        paths = [p for p, _ in added]
        lean_paths = sorted(set(paths))
        by_path = defaultdict(list)
        for p, x in added:
            by_path[p].append(x)

        joined = "\n".join(added_lines)
        nonblank = [x for x in added_lines if x.strip()]
        code_noncomment = [x for x in nonblank if not x.strip().startswith("--")]
        n_added_code = len(code_noncomment)
        n_changed = len(changed)
        add_count = sum(1 for _, sign, _ in changed if sign == "+")
        del_count = sum(1 for _, sign, _ in changed if sign == "-")

        decl_counts = Counter()
        named_decl_count = 0
        private_decl = 0
        protected_decl = 0
        decl_name_chars = []
        declarations = []
        for line in added_lines:
            m = DECL_RE.match(line)
            if m:
                decl_counts[m.group(1)] += 1
            mn = NAMED_DECL_RE.match(line)
            if mn:
                kind, name = mn.group(1), mn.group(2)
                if kind != "example":
                    named_decl_count += 1
                    declarations.append((kind, name or ""))
                    if "private " in line[: max(0, line.find(kind))]:
                        private_decl += 1
                    if "protected " in line[: max(0, line.find(kind))]:
                        protected_decl += 1
                    if name:
                        decl_name_chars.append(len(name))

        attrs = Counter()
        for line in added_lines:
            for m in ATTR_RE.finditer(line):
                for tok in re.split(r"[,\s]+", m.group(1).strip()):
                    tok = tok.strip()
                    if tok:
                        attrs[tok] += 1

        tactic_hits = TACTIC_RE.findall(joined)
        n_tactic_hits = len(tactic_hits)
        proof_markers = len(re.findall(r"\bby\b|:=\s*by\b", joined))
        chain_have = len(re.findall(r"^\s*have\b", joined, flags=re.M))
        chain_show = len(re.findall(r"^\s*show\b", joined, flags=re.M))
        chain_calc = len(re.findall(r"^\s*calc\b", joined, flags=re.M))
        chain_suffices = len(re.findall(r"^\s*suffices\b", joined, flags=re.M))
        chain_let = len(re.findall(r"^\s*let\b", joined, flags=re.M))
        nested_bullets = len(re.findall(r"^\s*[·.-]\s", joined, flags=re.M))
        doc_n = len(re.findall(r"/--", joined))
        doc_decl_n, public_decl_n = count_docstrings_before_decls(added_lines)
        import_lines = [IMPORT_RE.match(x).group(1) for x in added_lines if IMPORT_RE.match(x)]
        imported_roots = [x.split(".")[1] if x.startswith("Mathlib.") and len(x.split(".")) > 1 else x.split(".")[0] for x in import_lines]
        file_line_counts = [len(v) for v in by_path.values()] or [0]
        new_file_markers = 0
        deleted_file_markers = 0
        for path, sec_lines in parse_diff_sections(diff):
            text = "\n".join(sec_lines[:20])
            if "new file mode" in text or "--- /dev/null" in text:
                new_file_markers += 1
            if "deleted file mode" in text or "+++ /dev/null" in text:
                deleted_file_markers += 1

        long_lines = sum(1 for x in added_lines if len(x) > 100)
        very_long_lines = sum(1 for x in added_lines if len(x) > 120)
        by_cases = len(re.findall(r"\bby_cases\b|\bClassical\b|\bDecidable\b|\bif h\s*:", joined))
        gap_tokens = len(re.findall(r"\bsorry\b|\badmit\b|\bby\s+omega\?\b|\bby\s+aesop\?\b", joined))
        manual_refs = len(MANUAL_REF_RE.findall(joined))
        simpa_using = len(re.findall(r"\bsimpa\s+using\b", joined))
        simp_only = len(re.findall(r"\bsimp\s+only\b", joined))
        simp_all = len(re.findall(r"\bsimp_all\b", joined))
        terminal_done = len(re.findall(r"\b(done|omega|aesop|grind|linarith|nlinarith|ring|norm_num)\b", joined))
        namespace = len(re.findall(r"^\s*namespace\b", joined, flags=re.M))
        section = len(re.findall(r"^\s*section\b", joined, flags=re.M))
        variable = len(re.findall(r"^\s*variable\b|^\s*variables\b", joined, flags=re.M))
        open_cmd = len(re.findall(r"^\s*open\b", joined, flags=re.M))
        example_decl = decl_counts.get("example", 0)
        theorem_like = decl_counts.get("theorem", 0) + decl_counts.get("lemma", 0)
        def_like = (
            decl_counts.get("def", 0)
            + decl_counts.get("instance", 0)
            + decl_counts.get("structure", 0)
            + decl_counts.get("class", 0)
            + decl_counts.get("abbrev", 0)
            + decl_counts.get("inductive", 0)
            + decl_counts.get("coinductive", 0)
        )
        public_api = max(0, public_decl_n - private_decl)
        row = {
            "number": number,
            "new_total_added_lean_lines": len(added_lines),
            "new_total_changed_lean_lines": n_changed,
            "new_added_code_noncomment": n_added_code,
            "new_file_count": new_file_markers,
            "deleted_file_count": deleted_file_markers,
            "new_n_lean_files_touched": len(lean_paths),
            "new_lines_per_file_mean": float(np.mean(file_line_counts)),
            "new_lines_per_file_max": float(np.max(file_line_counts)),
            "new_lines_per_file_sd": float(np.std(file_line_counts)),
            "new_import_lines_added": len(import_lines),
            "new_import_roots_added": len(set(imported_roots)),
            "new_decl_public": public_api,
            "new_decl_named_total": named_decl_count,
            "new_decl_theorem_like": theorem_like,
            "new_decl_def_like": def_like,
            "new_decl_lemma": decl_counts.get("lemma", 0),
            "new_decl_theorem": decl_counts.get("theorem", 0),
            "new_decl_def": decl_counts.get("def", 0),
            "new_decl_instance": decl_counts.get("instance", 0),
            "new_decl_structure_class": decl_counts.get("structure", 0) + decl_counts.get("class", 0),
            "new_decl_abbrev": decl_counts.get("abbrev", 0),
            "new_decl_inductive": decl_counts.get("inductive", 0) + decl_counts.get("coinductive", 0),
            "new_decl_example": example_decl,
            "new_decl_private": private_decl,
            "new_decl_protected": protected_decl,
            "new_decl_avg_name_len": float(np.mean(decl_name_chars)) if decl_name_chars else 0.0,
            "new_attr_total": sum(attrs.values()),
            "new_attr_simp": attrs.get("simp", 0),
            "new_attr_ext": attrs.get("ext", 0),
            "new_attr_to_additive": attrs.get("to_additive", 0),
            "new_attr_norm_cast": attrs.get("norm_cast", 0),
            "new_attr_deprecated": attrs.get("deprecated", 0),
            "new_docstrings": doc_n,
            "new_doc_decl_covered": doc_decl_n,
            "new_doc_public_decl_total": public_decl_n,
            "new_proof_markers": proof_markers,
            "new_tactic_hits_extra": n_tactic_hits,
            "new_have_lines": chain_have,
            "new_show_lines": chain_show,
            "new_calc_lines": chain_calc,
            "new_suffices_lines": chain_suffices,
            "new_let_lines": chain_let,
            "new_chain_lines": chain_have + chain_show + chain_calc + chain_suffices + chain_let,
            "new_nested_bullets": nested_bullets,
            "new_by_cases_decidable": by_cases,
            "new_sorry_admit": gap_tokens,
            "new_manual_lowlevel_refs": manual_refs,
            "new_simpa_using": simpa_using,
            "new_simp_only": simp_only,
            "new_simp_all": simp_all,
            "new_terminal_auto": terminal_done,
            "new_namespace_lines": namespace,
            "new_section_lines": section,
            "new_variable_lines": variable,
            "new_open_lines": open_cmd,
            "new_long_lines_100": long_lines,
            "new_very_long_lines_120": very_long_lines,
            "new_max_indent": max_indent(added_lines),
        }
        den = max(1, n_added_code)
        row.update(
            {
                "new_public_decl_per_code_line": public_api / den,
                "new_theorem_like_per_code_line": theorem_like / den,
                "new_def_like_per_code_line": def_like / den,
                "new_attr_simp_per_decl": attrs.get("simp", 0) / max(1, theorem_like + def_like),
                "new_doc_coverage_public": doc_decl_n / max(1, public_decl_n),
                "new_proof_marker_density": proof_markers / den,
                "new_tactic_density_extra": n_tactic_hits / den,
                "new_chain_density": (chain_have + chain_show + chain_calc + chain_suffices + chain_let) / den,
                "new_manual_ref_density": manual_refs / den,
                "new_import_density": len(import_lines) / den,
                "new_file_spread_per_line": len(lean_paths) / den,
                "new_long_line_rate": long_lines / den,
                "new_newfile_share": new_file_markers / max(1, len(lean_paths)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def tfidf_model(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[float, float, np.ndarray, np.ndarray]:
    vec = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_'.]*\b",
        max_features=50000,
    )
    clf = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    xtr = vec.fit_transform(train_df["diff_noauth"].fillna(""))
    xev = vec.transform(eval_df["diff_noauth"].fillna(""))
    clf.fit(xtr, train_df["judgement"].astype(int))
    ptr = clf.predict_proba(xtr)[:, 1]
    pev = clf.predict_proba(xev)[:, 1]
    return safe_auc(eval_df["judgement"].astype(int), pev), topic_resid_auc(train_df, eval_df, ptr, pev), ptr, pev


def residual_mining(train_df: pd.DataFrame, eval_df: pd.DataFrame, vprime_cols: list[str]) -> dict:
    """Fit V′+TF-IDF and summarize TF-IDF tokens separating V′ misses."""
    v_auc, v_pred_eval, _ = fit_predict_auc(train_df, eval_df, vprime_cols)
    v_pred_train = logit_pipeline().fit(train_df[vprime_cols], train_df["judgement"].astype(int)).predict_proba(train_df[vprime_cols])[:, 1]
    # Wrong = below median for accepts or above median for rejects is too threshold-sensitive;
    # use balanced 0.5 model probability because class_weight fit is roughly centered.
    y_ev = eval_df["judgement"].astype(int).to_numpy()
    wrong_mask = ((v_pred_eval >= 0.5).astype(int) != y_ev)
    wrong = eval_df.loc[wrong_mask].copy()
    correct = eval_df.loc[~wrong_mask].copy()
    vec = TfidfVectorizer(
        min_df=3,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_'.]*\b",
        max_features=40000,
    )
    xtr = vec.fit_transform(train_df["diff_noauth"].fillna(""))
    xev = vec.transform(eval_df["diff_noauth"].fillna(""))
    # Joint model for residual examples.
    xtr_joint = sparse.hstack([StandardScaler().fit_transform(SimpleImputer(strategy="constant", fill_value=0.0).fit_transform(train_df[vprime_cols])), xtr], format="csr")
    # Refit transformer explicitly for eval alignment.
    imp = SimpleImputer(strategy="constant", fill_value=0.0)
    scaler = StandardScaler()
    xtr_num = scaler.fit_transform(imp.fit_transform(train_df[vprime_cols]))
    xev_num = scaler.transform(imp.transform(eval_df[vprime_cols]))
    clf = LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    clf.fit(sparse.hstack([xtr_num, xtr], format="csr"), train_df["judgement"].astype(int))
    joint_eval = clf.predict_proba(sparse.hstack([xev_num, xev], format="csr"))[:, 1]

    # Token means among V′ wrong accepts vs wrong rejects.
    names = np.asarray(vec.get_feature_names_out())
    wrong_accept_idx = np.where(wrong_mask & (y_ev == 1))[0]
    wrong_reject_idx = np.where(wrong_mask & (y_ev == 0))[0]
    def top_tokens(a_idx, b_idx, k=25):
        if len(a_idx) == 0 or len(b_idx) == 0:
            return []
        a = np.asarray(xev[a_idx].mean(axis=0)).ravel()
        b = np.asarray(xev[b_idx].mean(axis=0)).ravel()
        d = a - b
        top = np.argsort(-np.abs(d))[:k]
        return [(names[i], float(d[i]), float(a[i]), float(b[i])) for i in top]

    return {
        "v_auc": v_auc,
        "n_wrong": int(wrong_mask.sum()),
        "n_wrong_accept": int(((wrong_mask) & (y_ev == 1)).sum()),
        "n_wrong_reject": int(((wrong_mask) & (y_ev == 0)).sum()),
        "joint_auc": safe_auc(y_ev, joint_eval),
        "top_wrong_accept_vs_wrong_reject_tokens": top_tokens(wrong_accept_idx, wrong_reject_idx, 40),
    }


def candidate_groups(all_new_cols: list[str]) -> dict[str, list[str]]:
    groups = {
        "substantive_decl_structure": [
            c
            for c in all_new_cols
            if c.startswith("new_decl_")
            or c
            in {
                "new_public_decl_per_code_line",
                "new_theorem_like_per_code_line",
                "new_def_like_per_code_line",
            }
        ],
        "attributes": [c for c in all_new_cols if c.startswith("new_attr_")],
        "docstrings": [c for c in all_new_cols if c.startswith("new_doc")],
        "proof_complexity_extra": [
            c
            for c in all_new_cols
            if c
            in {
                "new_proof_markers",
                "new_tactic_hits_extra",
                "new_have_lines",
                "new_show_lines",
                "new_calc_lines",
                "new_suffices_lines",
                "new_let_lines",
                "new_chain_lines",
                "new_nested_bullets",
                "new_by_cases_decidable",
                "new_sorry_admit",
                "new_manual_lowlevel_refs",
                "new_simpa_using",
                "new_simp_only",
                "new_simp_all",
                "new_terminal_auto",
                "new_max_indent",
                "new_proof_marker_density",
                "new_tactic_density_extra",
                "new_chain_density",
                "new_manual_ref_density",
            }
        ],
        "diff_shape_extra": [
            c
            for c in all_new_cols
            if c
            in {
                "new_total_added_lean_lines",
                "new_total_changed_lean_lines",
                "new_added_code_noncomment",
                "new_file_count",
                "deleted_file_count",
                "new_n_lean_files_touched",
                "new_lines_per_file_mean",
                "new_lines_per_file_max",
                "new_lines_per_file_sd",
                "new_import_lines_added",
                "new_import_roots_added",
                "new_import_density",
                "new_file_spread_per_line",
                "new_newfile_share",
            }
        ],
        "style_smells_extra": [
            c
            for c in all_new_cols
            if c
            in {
                "new_namespace_lines",
                "new_section_lines",
                "new_variable_lines",
                "new_open_lines",
                "new_long_lines_100",
                "new_very_long_lines_120",
                "new_long_line_rate",
            }
        ],
    }
    groups["all_new_metrics"] = sorted(set(sum(groups.values(), [])))
    return {k: [c for c in v if c in all_new_cols] for k, v in groups.items()}


def pairwise_matched_signal(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Greedy accept/reject matching by area and size bucket inside eval split."""
    ev = df[df["split"].eq("eval")].copy()
    ev["size_bucket"] = pd.qcut(ev["additions"].fillna(ev["addn"]).rank(method="first"), q=8, labels=False, duplicates="drop")
    rows = []
    rng = np.random.default_rng(RANDOM_STATE)
    for (area, bucket), g in ev.groupby(["area", "size_bucket"], dropna=False):
        acc = g[g["judgement"].eq(1)].copy()
        rej = g[g["judgement"].eq(0)].copy()
        if acc.empty or rej.empty:
            continue
        acc_idx = list(acc.index)
        rej_idx = list(rej.index)
        rng.shuffle(acc_idx)
        rng.shuffle(rej_idx)
        for ai, ri in zip(acc_idx, rej_idx):
            row = {"area": area, "size_bucket": int(bucket), "accept_number": int(ev.loc[ai, "number"]), "reject_number": int(ev.loc[ri, "number"])}
            for c in feature_cols:
                row[c] = float(ev.loc[ai, c] - ev.loc[ri, c])
            rows.append(row)
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pd.DataFrame(columns=["feature", "n_pairs", "mean_accept_minus_reject", "std", "t_like", "wins"])
    out = []
    for c in feature_cols:
        x = pairs[c].replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) == 0:
            continue
        sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
        out.append(
            {
                "feature": c,
                "n_pairs": int(len(x)),
                "mean_accept_minus_reject": float(x.mean()),
                "std": sd,
                "t_like": float(x.mean() / (sd / math.sqrt(len(x)))) if sd > 0 else 0.0,
                "wins": float((x > 0).mean()),
            }
        )
    return pd.DataFrame(out).sort_values("t_like", key=lambda s: s.abs(), ascending=False)


def make_examples(df: pd.DataFrame, feature: str, n: int = 4) -> list[dict]:
    """Pick eval rows with extreme feature values and compact changed-line snippets."""
    ev = df[df["split"].eq("eval")].copy()
    ev = ev.sort_values(feature, ascending=False)
    out = []
    for _, r in ev.head(n).iterrows():
        lines = []
        for path, sign, line in all_changed_lines(r["diff_noauth"], lean_only=True):
            if sign == "+" and line.strip():
                lines.append(f"{path}: +{line}")
            if len(lines) >= 16:
                break
        out.append(
            {
                "number": int(r["number"]),
                "judgement": int(r["judgement"]),
                "area": str(r["area"]),
                "feature": feature,
                "value": float(r[feature]),
                "additions": int(r["additions"]) if pd.notna(r["additions"]) else None,
                "snippet": lines[:12],
            }
        )
    return out


def main() -> None:
    df = pd.read_parquet(BASE / "accept_reject_clean_deconf.parquet")
    if "additions" not in df.columns and "addn" in df.columns:
        df["additions"] = df["addn"]
    v = pd.read_parquet(BASE / "mathlib_diff_v_features.parquet")
    print("SCHEMA")
    print(json.dumps({
        "df_shape": df.shape,
        "v_shape": v.shape,
        "split_counts": df["split"].value_counts(dropna=False).to_dict(),
        "base_accept": float(df["judgement"].mean()),
        "df_columns": list(df.columns),
        "v_columns": list(v.columns),
    }, indent=2, default=str))

    # Join V and keep only artifact rows.
    v_cols = [c for c in v.columns if c != "number"]
    d = df.merge(v, on="number", how="left", suffixes=("", "_v"))
    tac_cols = sorted(c for c in d.columns if c.startswith("tac_"))
    v_cols = [c for c in v_cols if c in d.columns]
    # Avoid exact duplicates if any tactic-style columns exist in V.
    vprime_cols = v_cols + [c for c in tac_cols if c not in v_cols]
    # Numeric only.
    vprime_cols = [c for c in vprime_cols if pd.api.types.is_numeric_dtype(d[c])]
    v_cols = [c for c in v_cols if pd.api.types.is_numeric_dtype(d[c])]

    new_feats = extract_new_metrics(d)
    d = d.merge(new_feats, on="number", how="left")
    new_cols = [c for c in new_feats.columns if c != "number"]
    groups = candidate_groups(new_cols)

    train = d[d["split"].eq("train")].copy()
    ev = d[d["split"].eq("eval")].copy()
    for _c in list(dict.fromkeys(v_cols + tac_cols + new_cols)):
        train[_c] = pd.to_numeric(train[_c], errors="coerce").astype(float)
        ev[_c] = pd.to_numeric(ev[_c], errors="coerce").astype(float)
    print("COUNTS")
    print(json.dumps({
        "train_n": len(train),
        "eval_n": len(ev),
        "test_n": int((d["split"] == "test").sum()),
        "train_accept": float(train["judgement"].mean()),
        "eval_accept": float(ev["judgement"].mean()),
        "v_cols": v_cols,
        "tac_cols": tac_cols,
        "vprime_n_cols": len(vprime_cols),
        "new_metric_n_cols": len(new_cols),
    }, indent=2))

    results = []

    def eval_feature_set(name: str, cols: list[str], base_cols: list[str] | None = None) -> dict:
        base_cols = base_cols or []
        auc_st, pred_st_eval, pipe_st = fit_predict_auc(train, ev, cols)
        pred_st_train = pipe_st.predict_proba(train[cols])[:, 1]
        topic_st = topic_resid_auc(train, ev, pred_st_train, pred_st_eval)
        topic_st_eval = partial_topic_resid_auc_eval_only(ev, pred_st_eval)
        if base_cols:
            auc_inc, pred_inc_eval, pipe_inc = fit_predict_auc(train, ev, base_cols + cols)
            pred_inc_train = pipe_inc.predict_proba(train[base_cols + cols])[:, 1]
            topic_inc = topic_resid_auc(train, ev, pred_inc_train, pred_inc_eval)
            topic_inc_eval = partial_topic_resid_auc_eval_only(ev, pred_inc_eval)
        else:
            auc_inc = topic_inc = topic_inc_eval = float("nan")
        return {
            "name": name,
            "n_cols": len(cols),
            "standalone_raw_auc": auc_st,
            "standalone_topic_resid_auc": topic_st,
            "standalone_topic_resid_eval_auc": topic_st_eval,
            "vprime_plus_raw_auc": auc_inc,
            "vprime_plus_topic_resid_auc": topic_inc,
            "vprime_plus_topic_resid_eval_auc": topic_inc_eval,
        }

    base_v = eval_feature_set("V_orig", v_cols)
    base_vp = eval_feature_set("Vprime_orig_plus_tactics", vprime_cols)
    results.extend([base_v, base_vp])
    vprime_raw = base_vp["standalone_raw_auc"]
    vprime_topic = base_vp["standalone_topic_resid_auc"]

    for name, cols in groups.items():
        if not cols:
            continue
        r = eval_feature_set(name, cols, vprime_cols)
        r["incremental_raw_lift_over_vprime"] = r["vprime_plus_raw_auc"] - vprime_raw
        r["incremental_topic_lift_over_vprime"] = r["vprime_plus_topic_resid_auc"] - vprime_topic
        results.append(r)

    # Each individual metric.
    indiv = []
    for c in new_cols:
        r = eval_feature_set(c, [c], vprime_cols)
        r["incremental_raw_lift_over_vprime"] = r["vprime_plus_raw_auc"] - vprime_raw
        r["incremental_topic_lift_over_vprime"] = r["vprime_plus_topic_resid_auc"] - vprime_topic
        indiv.append(r)
    indiv_df = pd.DataFrame(indiv).sort_values("incremental_topic_lift_over_vprime", ascending=False)

    # C reference and residual mining.
    c_raw, c_topic, c_train_pred, c_eval_pred = tfidf_model(train, ev)
    resid = residual_mining(train, ev, vprime_cols)

    # Pairwise matched separation among candidate metrics.
    pairwise = pairwise_matched_signal(d, new_cols)

    # Coefs for best group and individual incremental model.
    best_indiv_name = indiv_df.iloc[0]["name"]
    best_group_df = pd.DataFrame([r for r in results if r["name"] in groups])
    best_group_name = best_group_df.sort_values("incremental_topic_lift_over_vprime", ascending=False).iloc[0]["name"]

    # Examples for the strongest standalone/incremental interpretable feature.
    example_feature = best_indiv_name
    examples = make_examples(d, example_feature)

    print("RESULTS_GROUPS")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("RESULTS_INDIVIDUAL_TOP30_BY_INCREMENTAL_TOPIC")
    print(indiv_df.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("RESULTS_INDIVIDUAL_TOP30_BY_STANDALONE_TOPIC")
    print(indiv_df.sort_values("standalone_topic_resid_auc", ascending=False).head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("REFERENCE_C_TFIDF")
    print(json.dumps({"C_raw_auc": c_raw, "C_topic_resid_auc": c_topic}, indent=2))
    print("RESIDUAL_MINING")
    print(json.dumps(resid, indent=2))
    print("PAIRWISE_TOP30")
    print(pairwise.head(30).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("BEST_SELECTION")
    print(json.dumps({
        "best_individual_incremental_topic": best_indiv_name,
        "best_group_incremental_topic": best_group_name,
        "vprime_raw": vprime_raw,
        "vprime_topic": vprime_topic,
        "tfidf_raw": c_raw,
        "tfidf_topic": c_topic,
    }, indent=2))
    print("EXAMPLES")
    print(json.dumps(examples, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
