"""a280: Session management security and controls.

The norm: "Architect and enforce secure session handling: bind sessions
appropriately (user/device), prefer secure cookie-based session ID exchange,
prevent fixation/ID exposure, apply safe caching/headers to avoid stale
sensitive content, and use WAF mitigations where code changes are infeasible."

DISTINCT FROM a47 (security hardening) AND a267 (injection prevention).

a47 = generic Python-source hygiene (hardcoded secrets, weak crypto, insecure
default deserialization, TLS verify=False, dangerous yaml.load, etc) via
bandit. a47 has NO bandit rules for session cookie flags or JWT verification
patterns. a267 = injection (SQL/shell/eval/XSS) via bandit's B6xx + injection
B7xx. Neither covers SESSION-specific anti-patterns:

  - Cookies set without `HttpOnly`, `Secure`, `SameSite` flags
  - Session IDs appearing in URLs / query strings
  - JWT decode/verify with `verify=False` or `algorithms=['none']`
  - JWT signing without `exp` (expiration) claim
  - Flask `session.permanent = True` without lifetime control
  - Long absolute timeouts (PERMANENT_SESSION_LIFETIME in days)
  - Missing CSRF token wiring on session-state-changing endpoints
  - Cookie name leaks session in URL parameter (?sessionid=...)

These are tied to HTTP session management, which has no off-the-shelf
Python tool with high precision (bandit lacks; semgrep p/owasp-top-ten covers
some but is not in the sandbox). We use **tree-sitter AST walks across
Python, JS/TS, Java, and Go** to identify the relevant call sites and check
for the security flags / verification kwargs.

LANGUAGE-SPECIFIC SIGNALS WE COUNT
==================================

Python
------
  Bad cookie-setter (no security flags):
    resp.set_cookie("name", value)            # no httponly/secure/samesite
    response.set_cookie(..., httponly=False)
    response.set_cookie(..., secure=False)
  Good cookie-setter:
    resp.set_cookie(..., httponly=True, secure=True, samesite="Lax")
  Bad JWT:
    jwt.decode(tok, verify=False)
    jwt.decode(tok, options={"verify_signature": False})
    jwt.decode(tok, algorithms=["none"])
  Good JWT:
    jwt.decode(tok, key, algorithms=[...])    # no verify=False
    jwt.encode({..., "exp": ...}, key, ...)   # has exp claim

JavaScript / TypeScript
-----------------------
  Bad (express):
    res.cookie("sid", v)                      # no options object
    res.cookie("sid", v, {})
    res.cookie("sid", v, {httpOnly: false})
    res.cookie("sid", v, {secure: false})
  Good (express):
    res.cookie("sid", v, {httpOnly: true, secure: true, sameSite: "strict"})
    app.use(session({cookie: {httpOnly: true, secure: true, sameSite: "lax"}}))
  Bad JWT:
    jwt.verify(tok, secret, {algorithms: ["none"]})
    jsonwebtoken.decode(tok)                  # decode without verify

Java
----
  Bad: cookie.setHttpOnly(false), cookie.setSecure(false), no setHttpOnly call
  Good: cookie.setHttpOnly(true), cookie.setSecure(true)

Go
--
  Bad: &http.Cookie{Name: ..., Value: ...} with HttpOnly:false or omitted
  Good: &http.Cookie{... HttpOnly: true, Secure: true, SameSite: ...}

APPLIES GATE
============
Narrow. We only apply if added code contains at least one session-relevant
identifier: cookie setter call, JWT call, session-config call, `Set-Cookie`
header literal, or session-object configuration. Most code_review diffs will
NOT touch session code → applies()=False → abstain (correct behavior).

SCORE
=====
Score = (good + 0.5) / (good + bad + 1.0). Laplace-smoothed ratio so a
single-call file with all-good (1, 0) -> 0.75, with all-bad (0, 1) -> 0.25,
balanced (1, 1) -> 0.5. Abstain if neither good nor bad sites detected.

CLASSIFICATION = PARTIALLY_THIN. Cookie flags and JWT verify kwargs are
deterministically checkable. But "session bound to user/device", "WAF
mitigations", and the broader architectural ask ("architect and enforce
secure session handling") cannot be verified from a diff alone — they need
runtime context. So this metric measures a structural lower bound, not the
whole norm.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from ..sandbox import parse_diff_added_by_file

ASPECT_ID = "a280"
ASPECT_NAME = "Session management security and controls"
TIER = 2
TOOLS = ["tree-sitter-python", "tree-sitter-javascript",
         "tree-sitter-typescript", "tree-sitter-java", "tree-sitter-go"]
APPLIES_TO_LANGS = ["Python", "JavaScript", "TypeScript", "Java", "Go"]
CLASSIFICATION = "PARTIALLY_THIN"

EXT_TO_LANG = {
    ".py": "py", ".pyi": "py",
    ".js": "js", ".jsx": "js", ".mjs": "js", ".cjs": "js",
    ".ts": "ts", ".tsx": "ts",
    ".java": "java",
    ".go": "go",
}

# Identifier names that surface in calls/configs we want to inspect.
# Used both as an applicability prefilter (cheap substring) and as
# AST node-name targets inside _collect.
SESSION_TOKENS = (
    # cookies
    "set_cookie", "setcookie", "set-cookie",  # method/string
    "cookie", "Cookie", "HttpOnly", "httpOnly", "Secure", "SameSite",
    # jwt
    "jwt", "JWT", "jsonwebtoken", "PyJWT", "verify_signature",
    "algorithms", "encode", "decode",
    # session config
    "session", "Session", "express-session", "PERMANENT_SESSION_LIFETIME",
    "SESSION_COOKIE", "SecureRandom", "csrf", "CSRF",
)

_PARSERS: Dict[str, object] = {}


def _get_parser(lang: str):
    if lang in _PARSERS:
        return _PARSERS[lang]
    try:
        from tree_sitter import Language, Parser
        if lang == "py":
            import tree_sitter_python as m; L = m.language()
        elif lang == "js":
            import tree_sitter_javascript as m; L = m.language()
        elif lang == "ts":
            import tree_sitter_typescript as m; L = m.language_typescript()
        elif lang == "java":
            import tree_sitter_java as m; L = m.language()
        elif lang == "go":
            import tree_sitter_go as m; L = m.language()
        else:
            return None
        _PARSERS[lang] = Parser(Language(L))
        return _PARSERS[lang]
    except ImportError:
        return None


def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf8", errors="replace")


# ----- Python ---------------------------------------------------------------

def _py_keywords(call_node, src: bytes) -> Dict[str, str]:
    """Return {kw_name: kw_value_text} from a python call's argument list."""
    kw: Dict[str, str] = {}
    args = call_node.child_by_field_name("arguments")
    if args is None:
        return kw
    for c in args.children:
        if c.type == "keyword_argument":
            name_node = None
            val_node = None
            for cc in c.children:
                if cc.type == "identifier" and name_node is None:
                    name_node = cc
                elif cc.type not in ("=", ",", "(", ")"):
                    val_node = cc
            if name_node is not None and val_node is not None:
                kw[_text(name_node, src)] = _text(val_node, src)
    return kw


def _py_collect(root, src: bytes) -> Tuple[int, int]:
    good = bad = 0

    def walk(node):
        nonlocal good, bad
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                fn_txt = _text(fn, src)
                fn_last = fn_txt.rsplit(".", 1)[-1]
                # Cookie setters: foo.set_cookie(...) / response.set_cookie
                if fn_last == "set_cookie":
                    kw = _py_keywords(node, src)
                    httponly = kw.get("httponly", "").lower()
                    secure = kw.get("secure", "").lower()
                    samesite = kw.get("samesite", "")
                    if httponly == "false" or secure == "false":
                        bad += 1
                    else:
                        flags = 0
                        if httponly == "true":
                            flags += 1
                        if secure == "true":
                            flags += 1
                        if samesite and samesite.strip("\"'").lower() in (
                                "lax", "strict", "none"):
                            flags += 1
                        if flags >= 2:
                            good += 1
                        else:
                            bad += 1
                # JWT decode/verify
                elif fn_last == "decode" and "jwt" in fn_txt.lower():
                    kw = _py_keywords(node, src)
                    verify = kw.get("verify", "").lower()
                    opts = kw.get("options", "")
                    algos = kw.get("algorithms", "")
                    bad_flag = False
                    if verify == "false":
                        bad_flag = True
                    if "verify_signature" in opts and "false" in opts.lower():
                        bad_flag = True
                    if "'none'" in algos.lower() or '"none"' in algos.lower():
                        bad_flag = True
                    if bad_flag:
                        bad += 1
                    elif algos:  # explicit algorithms passed
                        good += 1
                    else:
                        # decode w/o verify but no explicit kill switch — neutral
                        pass
                elif fn_last == "encode" and "jwt" in fn_txt.lower():
                    # Look for 'exp' literal anywhere in the call text
                    txt = _text(node, src)
                    if '"exp"' in txt or "'exp'" in txt:
                        good += 1
                    else:
                        bad += 1
                # Flask session.permanent = True patterns; assignment, not call
        # Assignments e.g. session.permanent = True
        if node.type == "assignment":
            txt = _text(node, src)
            if "session.permanent" in txt and "true" in txt.lower():
                bad += 1
            elif "SESSION_COOKIE_HTTPONLY" in txt and "true" in txt.lower():
                good += 1
            elif "SESSION_COOKIE_SECURE" in txt and "true" in txt.lower():
                good += 1
            elif "SESSION_COOKIE_SAMESITE" in txt:
                good += 1
        for c in node.children:
            walk(c)

    walk(root)
    return good, bad


# ----- JS / TS --------------------------------------------------------------

def _js_object_to_dict(obj_node, src: bytes) -> Dict[str, str]:
    """Crudely flatten an object_expression's top-level pairs."""
    out: Dict[str, str] = {}
    if obj_node is None:
        return out
    for c in obj_node.children:
        if c.type in ("pair", "property_assignment"):
            key_node = None
            val_node = None
            for cc in c.children:
                if cc.type == "property_identifier" or cc.type == "identifier":
                    key_node = cc; break
                if cc.type == "string":
                    key_node = cc; break
            # value is the last meaningful child after ':'
            saw_colon = False
            for cc in c.children:
                if cc.type == ":":
                    saw_colon = True
                    continue
                if saw_colon and cc.type not in (",",):
                    val_node = cc
            if key_node is not None and val_node is not None:
                k = _text(key_node, src).strip("\"'")
                out[k] = _text(val_node, src)
    return out


def _js_collect(root, src: bytes) -> Tuple[int, int]:
    good = bad = 0

    def walk(node):
        nonlocal good, bad
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            args = node.child_by_field_name("arguments")
            if fn is not None and fn.type == "member_expression":
                prop = fn.child_by_field_name("property")
                prop_txt = _text(prop, src) if prop is not None else ""
                # res.cookie("name", val [, options])
                if prop_txt == "cookie" and args is not None:
                    opts = None
                    arg_children = [c for c in args.children
                                    if c.type not in (",", "(", ")")]
                    if len(arg_children) >= 3 and \
                            arg_children[2].type in ("object", "object_expression"):
                        opts = arg_children[2]
                    if opts is None:
                        bad += 1
                    else:
                        d = _js_object_to_dict(opts, src)
                        ho = d.get("httpOnly", "").lower()
                        sc = d.get("secure", "").lower()
                        ss = d.get("sameSite", "").lower()
                        if ho == "false" or sc == "false":
                            bad += 1
                        else:
                            flags = sum(1 for v in (ho, sc, ss) if v and v != "false")
                            if flags >= 2:
                                good += 1
                            else:
                                bad += 1
                # jwt.verify(tok, secret, {algorithms: ["none"]})
                elif prop_txt == "verify":
                    obj = fn.child_by_field_name("object")
                    obj_txt = _text(obj, src) if obj is not None else ""
                    if "jwt" in obj_txt.lower() or "jsonwebtoken" in obj_txt.lower():
                        txt = _text(node, src)
                        if '"none"' in txt or "'none'" in txt:
                            bad += 1
                        else:
                            good += 1
                # jwt.decode(...) without verify → typically unsafe in node
                elif prop_txt == "decode":
                    obj = fn.child_by_field_name("object")
                    obj_txt = _text(obj, src) if obj is not None else ""
                    if "jwt" in obj_txt.lower() or "jsonwebtoken" in obj_txt.lower():
                        bad += 1
                # express-session: session({cookie: {...}})
                elif prop_txt in ("use", "session"):
                    txt = _text(node, src)
                    if "session(" in txt and "cookie" in txt:
                        # Look for httpOnly/secure flags in the call text
                        has_ho = "httpOnly" in txt
                        has_sc = "secure:" in txt or "secure :" in txt
                        if (has_ho or has_sc) and "false" not in txt.lower():
                            good += 1
        for c in node.children:
            walk(c)

    walk(root)
    return good, bad


# ----- Java -----------------------------------------------------------------

def _java_collect(root, src: bytes) -> Tuple[int, int]:
    good = bad = 0

    def walk(node):
        nonlocal good, bad
        if node.type == "method_invocation":
            nm = node.child_by_field_name("name")
            method = _text(nm, src) if nm is not None else ""
            args = node.child_by_field_name("arguments")
            if method in ("setHttpOnly", "setSecure"):
                if args is not None:
                    a_txt = _text(args, src).strip("()").strip().lower()
                    if a_txt == "true":
                        good += 1
                    elif a_txt == "false":
                        bad += 1
            elif method == "setMaxAge":
                if args is not None:
                    a_txt = _text(args, src).strip("()").strip()
                    # Excessively long max-age in seconds (> 30 days = 2592000) bad
                    try:
                        v = int(a_txt)
                        if v > 60 * 60 * 24 * 30:
                            bad += 1
                        else:
                            good += 1
                    except ValueError:
                        pass
        # new Cookie(...) is benign; we count subsequent setter calls.
        for c in node.children:
            walk(c)

    walk(root)
    return good, bad


# ----- Go -------------------------------------------------------------------

def _go_collect(root, src: bytes) -> Tuple[int, int]:
    good = bad = 0

    def walk(node):
        nonlocal good, bad
        if node.type == "composite_literal":
            # Detect &http.Cookie{...} or http.Cookie{...}
            txt = _text(node, src)
            if "Cookie" in txt and ("HttpOnly" in txt or "Secure" in txt or
                                    "SameSite" in txt or "Name" in txt):
                has_ho = "HttpOnly: true" in txt or "HttpOnly:true" in txt
                has_sc = "Secure: true" in txt or "Secure:true" in txt
                has_ss = "SameSite:" in txt
                bad_ho = "HttpOnly: false" in txt or "HttpOnly:false" in txt
                bad_sc = "Secure: false" in txt or "Secure:false" in txt
                if bad_ho or bad_sc:
                    bad += 1
                else:
                    flags = sum(1 for x in (has_ho, has_sc, has_ss) if x)
                    if flags >= 2:
                        good += 1
                    elif "Name:" in txt or "Value:" in txt:
                        # Cookie literal w/o security flags
                        bad += 1
        for c in node.children:
            walk(c)

    walk(root)
    return good, bad


# ----- Dispatch -------------------------------------------------------------

def _path_lang(path: str) -> Optional[str]:
    p = path.lower()
    for ext, lang in EXT_TO_LANG.items():
        if p.endswith(ext):
            return lang
    return None


def _collect(code: bytes, lang: str) -> Optional[Tuple[int, int]]:
    parser = _get_parser(lang)
    if parser is None:
        return None
    try:
        root = parser.parse(code).root_node
    except Exception:
        return None
    if lang == "py":
        return _py_collect(root, code)
    if lang in ("js", "ts"):
        return _js_collect(root, code)
    if lang == "java":
        return _java_collect(root, code)
    if lang == "go":
        return _go_collect(root, code)
    return None


def _has_session_token(text: str) -> bool:
    """Cheap substring prefilter to gate applies()."""
    lowered = text.lower()
    # Tight tokens (less ambiguous than generic "session")
    tight = ("set_cookie", "set-cookie", ".cookie(",
             "httponly", "samesite", "jwt.decode", "jwt.encode",
             "jwt.verify", "jsonwebtoken", "session_cookie",
             "permanent_session_lifetime", "session.permanent",
             "sethttponly", "setsecure", "http.cookie")
    return any(t in lowered for t in tight)


def applies(diff_text: str) -> bool:
    """Tight gate: only apply if a session/cookie/jwt token appears in added code.

    Most code_review diffs will NOT touch session code; abstaining is correct.
    """
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return False
    for p, c in by_path.items():
        if _path_lang(p) is None:
            continue
        if _has_session_token(c):
            return True
    return False


def score(diff_text: str) -> Optional[float]:
    by_path = parse_diff_added_by_file(diff_text)
    if not by_path:
        return None

    tot_good = tot_bad = 0
    saw_any = False

    for path, content in by_path.items():
        lang = _path_lang(path)
        if lang is None:
            continue
        if not _has_session_token(content):
            continue
        res = _collect(content.encode("utf8", errors="replace"), lang)
        if res is None:
            continue
        g, b = res
        tot_good += g
        tot_bad += b
        if g or b:
            saw_any = True

    if not saw_any:
        return None

    # Laplace smoothing.
    return float((tot_good + 0.5) / (tot_good + tot_bad + 1.0))
