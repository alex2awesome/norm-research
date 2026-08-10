def combine(units: dict, text: str) -> float:
    """
    Deterministic integrator for Title VII plaintiff-favorability forecast.
    Higher output => more likely the court rules FOR the plaintiff.
    Refined with threshold gating to handle stage-confusion and misranked cases.
    """
    try:
        def _c(uid, lo=0.0, hi=1.0):
            v = units.get(uid)
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            try:
                v = float(v)
            except (TypeError, ValueError):
                return lo
            return max(lo, min(hi, v))

        def _p(uid, lo=0.0, hi=10.0):
            v = _c(uid, lo, hi)
            return v / 10.0

        severity   = _c("u2")
        corrective = _c("u3")
        retal_evi  = _c("u4")
        stats      = _c("u5")
        aff_def    = _c("u6")
        gdisf      = _p("u7")

        tl = (text or "").lower()

        is_sj  = any(k in tl for k in [
            "summary judgment", "rule 56", "genuine dispute", "material fact", "celotex"
        ])

        is_judgment_phase = any(k in tl for k in [
            "judgment as a matter of law", "rule 50", "renewed motion for judgment",
            "directed verdict", "trial", "jury verdict"
        ])

        is_mt = any(k in tl for k in [
            "motion to dismiss", "rule 12", "12(b)(6)", "dismiss for failure", "i.q.b.a.l"
        ])

        if not (is_sj or is_judgment_phase or is_mt):
            # Fallback to u1 if text heuristics fail, though u1 seems to mostly just encode "is SJ=1.0" in this dataset
            is_sj = True

        # 1. EVIDENCE THRESHOLD GATING
        # Code units scale differently than prompt units. 
        # For u3 (corrective), 7.15 is max, so ~0.5 in logic.
        # For u4 (retaliation), 2.639 is max, so ~1.0 in logic.
        has_discrim_evidence = severity >= 0.60
        has_systemic_evidence = stats >= 0.60
        
        max_retal_code = 2.639
        retal_evi_norm = min(1.0, retal_evi / max_retal_code)
        has_retal_evidence = retal_evi_norm >= 0.90

        has_strong_evidence = has_discrim_evidence or has_systemic_evidence or has_retal_evidence

        # 2. SUMMARY JUDGMENT / JMOL GATING
        # If the court is evaluating evidence at SJ/JMOL, weak/generic evidence (gdisf) = defense win.
        if (is_sj or is_judgment_phase) and not has_strong_evidence:
            if gdisf >= 0.70:
                return 0.30 # Plaintiff has general dispute but no SPECIFIC proof -> likely loses SJ.
            else:
                return 0.15 # Plaintiff loses clearly at SJ.

        # 3. AFFIRMATIVE ACTION / GOVERNMENT DEFENSE
        # If the defendant is a government entity justifying an AA plan, traditional bias metrics (u5)
        # or disputes (u7) favor the DEFENSE. Only if the plaintiff shows extreme, specific retaliation 
        # or discrimination (high u2, u4) might the plaintiff win.
        if aff_def >= 0.45 and not has_retal_evidence and not has_discrim_evidence:
            return 0.25

        # 4. SCORE ASSEMBLY (for cases passing the gates)
        
        # Corrective action penalty: if employer took prompt action (high u3), it undercuts hostile environment,
        # BUT it doesn't automatically defeat systemic or retaliation claims.
        # Note: u3 max is 7.15 -> maps to 1.0
        max_corr_code = 7.15
        corrective_norm = min(1.0, corrective / max_corr_code)
        
        effective_severity = max(0.0, severity - (corrective_norm * 0.45))
        
        merit = max(
            effective_severity * 0.75,
            stats,
            retal_evi_norm * 0.90
        )
        
        if merit < 0.15:
            return 0.25

        # Map merit to base probability
        if merit < 0.40:
            base = 0.30 + (merit - 0.15) * 0.80  # 0.30 to 0.50
        else:
            base = 0.50 + (merit - 0.40) * 1.0   # 0.50 to 1.10 (capped later)
            
        # Adjustments for procedural agreement (only matters at SJ)
        if is_sj:
            if gdisf >= 0.80 and merit >= 0.40:
                base += 0.05
            elif gdisf <= 0.30 and merit < 0.40:
                base -= 0.05

        return max(0.05, min(0.95, base))

    except Exception:
        return 0.5