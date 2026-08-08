"""a47: Security hardening and hygiene (defense-in-depth, safe defaults).

Scope: BROADER than a267 (injection-specific). This metric measures security
*hygiene* — the things you should never do regardless of input source. a267
covers strict input validation, parameterized queries, output encoding, and
"injection prevention". a47 covers the *other* axis of defense-in-depth:
hardcoded secrets, weak crypto, insecure defaults, disabled TLS verification,
dangerous deserialization, dynamic code execution, etc.

Tool: bandit (Python static analyzer). Bandit ships a few dozen plugins each
tagged with a B-code. We hand-partition them:

  HARDENING / HYGIENE (this metric — a47)
    B102  exec_used              dynamic code execution
    B103  set_bad_file_permissions  chmod with overly permissive bits
    B104  hardcoded_bind_all_interfaces  binding 0.0.0.0
    B105  hardcoded_password_string
    B106  hardcoded_password_funcarg
    B107  hardcoded_password_default
    B108  hardcoded_tmp_directory
    B110  try_except_pass        swallowed exceptions (defense-in-depth)
    B112  try_except_continue
    B113  request_without_timeout
    B201  flask_debug_true       insecure default
    B301  pickle.loads           insecure deserialization
    B302  marshal.loads          insecure deserialization
    B303  insecure hashes (md5/sha1 via hashlib pre-3.9 calls)
    B304  insecure ciphers       weak crypto (DES, ARC2, ARC4, Blowfish)
    B305  insecure cipher modes  ECB
    B306  mktemp_q
    B307  eval                   dynamic code execution
    B311  random                 non-cryptographic RNG used for security
    B312  telnetlib              cleartext protocol
    B313-B320  XML parsers       XXE / billion-laughs defaults
    B321  ftplib                 cleartext protocol
    B322  input (py2)
    B323  unverified_context     TLS context with no cert verification
    B324  hashlib_insecure_functions  MD5/SHA1 for security
    B401-B413 import_* of risky modules
    B501  request_with_no_cert_validation  verify=False
    B502  ssl_with_bad_version
    B503  ssl_with_bad_defaults
    B504  ssl_with_no_version
    B505  weak_cryptographic_key
    B506  yaml_load              insecure yaml.load
    B507  ssh_no_host_key_verification
    B508-B509 SNMP cleartext / weak crypto
    B701  jinja2_autoescape_false
    B702  use_of_mako_templates  (XSS — borderline, kept as default-hygiene)
    B703  django_mark_safe
    B704  markupsafe_markup_xss

  INJECTION (a267's scope — EXCLUDED here)
    B608  hardcoded_sql_expressions
    B609  linux_commands_wildcard_injection
    B610  django_extra_used      ORM injection
    B611  django_rawsql_used
    B602  subprocess_popen_with_shell_equals_true
    B603  subprocess_without_shell_equals_true
    B604  any_other_function_with_shell_equals_true
    B605  start_process_with_a_shell
    B606  start_process_with_no_shell
    B607  start_process_with_partial_path

Score. Bandit emits issues at LOW/MEDIUM/HIGH severity. We weight findings by
severity (LOW=1, MEDIUM=2, HIGH=4) and divide by added-Python-lines to get
weighted_density. Score = exp(-weighted_density * 8.0), so:
  0 issues       → 1.000
  1 HIGH / 50 LOC → exp(-4/50*8) = exp(-0.64) = 0.527
  1 MED  / 100 LOC → exp(-2/100*8) = exp(-0.16) = 0.852
  3 LOW  / 200 LOC → exp(-3/200*8) = exp(-0.12) = 0.887

Restricted to Python (where bandit operates). JS/TS would need semgrep with
custom rulesets and is left for a future per-language sibling.

Conformance interpretation: 1.0 = no hygiene issues in added code; lower means
more / more-severe insecure-default findings. Whether this correlates with PR
merge is the empirical question.
"""
from __future__ import annotations

import json
import math
import shutil
from typing import Optional, Set

from ..sandbox import added_files_by_ext, have_tool, run, write_temp_files

ASPECT_ID = "a47"
ASPECT_NAME = "Security hardening and hygiene"
TIER = 3
TOOLS = ["bandit"]
APPLIES_TO_LANGS = ["Python"]
CLASSIFICATION = "THIN"

PY_EXTS = [".py", ".pyi"]

# Hardening / hygiene IDs. Injection-class IDs (B6xx mostly) are EXCLUDED
# because a267 is the dedicated injection metric — we want a distinct signal.
HARDENING_IDS: Set[str] = {
    "B102", "B103", "B104", "B105", "B106", "B107", "B108",
    "B110", "B112", "B113",
    "B201",
    "B301", "B302", "B303", "B304", "B305", "B306", "B307",
    "B311", "B312",
    "B313", "B314", "B315", "B316", "B317", "B318", "B319", "B320",
    "B321", "B322", "B323", "B324",
    "B401", "B402", "B403", "B404", "B405", "B406", "B407",
    "B408", "B409", "B410", "B411", "B412", "B413",
    "B501", "B502", "B503", "B504", "B505", "B506", "B507",
    "B508", "B509",
    "B701", "B702", "B703", "B704",
}

# Severity weights for blending.
SEV_W = {"LOW": 1.0, "MEDIUM": 2.0, "HIGH": 4.0}


def applies(diff_text: str) -> bool:
    """Applies if the diff adds any Python source lines."""
    return bool(added_files_by_ext(diff_text, PY_EXTS))


def score(diff_text: str) -> Optional[float]:
    if not have_tool("bandit"):
        return None
    by_path = added_files_by_ext(diff_text, PY_EXTS)
    if not by_path:
        return None
    total_lines = sum(1 + t.count("\n") for t in by_path.values())
    if total_lines == 0:
        return None
    td = write_temp_files(by_path)
    try:
        # bandit -r <dir> -f json: machine-readable findings list.
        # --quiet suppresses banner; --severity-level all keeps LOW too.
        rc, out, _err = run(
            ["bandit", "-r", "-f", "json", "-q",
             "--severity-level", "all", str(td)],
            timeout=20.0,
        )
        # bandit exits 0 if no issues, 1 if issues found, other codes on
        # error. We accept either via JSON parse.
        if rc < 0:
            return None
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return None
        results = data.get("results", []) or []
        weighted = 0.0
        for r in results:
            tid = r.get("test_id", "")
            if tid not in HARDENING_IDS:
                # Filter out injection-class findings (a267's territory).
                continue
            sev = (r.get("issue_severity") or "").upper()
            weighted += SEV_W.get(sev, 1.0)
        density = weighted / max(total_lines, 1)
        return float(math.exp(-density * 8.0))
    finally:
        shutil.rmtree(td, ignore_errors=True)
