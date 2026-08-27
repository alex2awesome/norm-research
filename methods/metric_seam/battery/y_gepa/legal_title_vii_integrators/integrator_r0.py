def combine(units: dict, text: str) -> float:
    """
    Deterministic integrator for Title VII plaintiff-favorability forecast.
    Higher output => more likely the court rules FOR the plaintiff.
    """
    try:
        def _c(uid, lo=0.0, hi=1.0):
            v = units.get(uid)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            try:
                v = float(v)
            except (TypeError, ValueError):
                return 0.0
            return max(lo, min(hi, v))

        def _p(uid, lo=0.0, hi=10.0):
            v = _c(uid, lo, hi)
            return v / 10.0

        stage      = _c("u1")
        severity   = _c("u2")
        corrective = _c("u3")
        retal_evi  = _c("u4")
        stats      = _c("u5")
        aff_def    = _c("u6")
        gdisf      = _p("u7")

        # --- Heuristics from text (procedural context) ---
        tl = (text or "").lower()
        is_mt  = any(k in tl for k in [
            "motion to dismiss", "rule 12", "12(b)(6)", "dismiss for failure",
            "motion for judgment as a matter", "judgment on the pleadings",
        ])
        is_sj  = any(k in tl for k in [
            "summary judgment", "rule 56", "genuine dispute", "material fact",
        ])
        is_bench_verdict = "judgment as a matter of law" in tl or ("renewed" in tl and "judgment" in tl)
        is_appeal = "affirmed" in tl or "reversed" in tl or "remanded" in tl or "court of appeals" in tl

        # --- Merit gate (plaintiff must show *something*) ---
        merit = max(severity, retal_evi, stats, 0.0)
        if merit < 0.08:
            return 0.18

        # --- Procedural stage scaling ---
        if is_mt:
            procedural_boost = 0.06
            evidence_weight  = 0.55
            sj_weight        = 0.08
        elif is_sj:
            procedural_boost = 0.0
            evidence_weight  = 0.38
            sj_weight        = 0.34
        else:
            procedural_boost = 0.0
            evidence_weight  = 0.32
            sj_weight        = 0.10

        # --- Stage-level agreement ---
        stage_discrim = max(severity, retal_evi, stats * 0.85)
        stage_agree = gdisf * 0.7 + stage_discrim * 0.3

        # --- Retaliation branch ---
        retaliation_path = retal_evi * 0.50 + stage_agree * 0.50

        # --- Discrimination / harassment branch ---
        severe_perv = max(severity, (severity ** 0.8) * (1.0 - corrective * 0.55))
        stat_boost  = max(stats, stats * 0.75 * severe_perv)

        discrim_path = (
            severe_perv * 0.38
            + stat_boost * 0.28
            + stage_agree * 0.28
            + procedural_boost
        )

        # --- Blend ---
        if retal_evi > severity + 0.18:
            base = retaliation_path
        elif severity > retal_evi + 0.15:
            base = discrim_path
        else:
            base = 0.40 * retaliation_path + 0.60 * discrim_path

        # --- Procedural overrides ---
        if is_sj and gdisf > 0.70 and max(severity, retal_evi, stats) > 0.55:
            base = max(base, 0.78)

        if is_mt and max(severity, retal_evi, stats) > 0.50 and gdisf > 0.45:
            base = max(base, 0.70)

        # --- Genuine-dispute floor (procedural fairness) ---
        gdisf_floor = gdisf * 0.34
        if gdisf > 0.55:
            gdisf_floor += 0.10
        base = max(base, gdisf_floor)

        # --- Defense penalties ---
        if aff_def > 0.55:
            base -= 0.30
        if aff_def > 0.80:
            base -= 0.12

        # --- Lopsided evidence modifier ---
        if corrective > 0.70 and severity < 0.40:
            base -= 0.15

        # --- Cap by procedural ceiling ---
        if is_mt:
            cap = 0.92
        elif is_sj:
            cap = 0.88
        else:
            cap = 0.95

        final = max(0.05, min(cap, base))

        if 0.20 < aff_def < 0.55:
            final *= (1.0 - 0.08 * aff_def)

        return float(round(final, 4))

    except Exception:
        return 0.5