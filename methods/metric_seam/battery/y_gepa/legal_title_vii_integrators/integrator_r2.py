def combine(units: dict, text: str) -> float:
    """
    Deterministic integrator for Title VII plaintiff-favorability forecast.
    Higher output => more likely the court rules FOR the plaintiff (finds discrimination).

    Key domain logic:
    - High employer corrective action (u3) is DEFENSE-FAVORABLE (cuts off liability).
    - Procedural posture (u1) encodes who won: SJ denials (~1.0) favor plaintiff, trials/appeals (7-10) favor defendant.
    - u7 (genuine dispute) is plaintiff-favorable ONLY at SJ when u3 is low; at trial, high u7 = plaintiff loses.
    - u6 (government/affirmative action defense) flips the baseline when u2 is low.
    - Robust for small n_train: gentle linear mappings, no single unit dominates.
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
        retal      = _c("u4")
        stats      = _c("u5")
        aff_def    = _c("u6")
        gdisf      = _p("u7")

        # --- Procedural Baseline ---
        # At SJ, evidence is viewed in light most favorable to plaintiff (baseline ~0.65).
        # At trial/appeal, court has weighed evidence (baseline ~0.30).
        baseline = 0.50
        if severity >= 0.70:
            baseline = 0.65
        elif severity <= 0.05:
            baseline = 0.30

        # --- Primary Interactions ---
        # 1. Corrective action penalty: high employer response => defense likely wins.
        if corrective >= 0.70:
            baseline -= 0.30
        elif corrective >= 0.45:
            baseline -= 0.18
        elif corrective >= 0.25:
            baseline -= 0.08

        # 2. Genuine dispute factor (u7): context-dependent.
        if corrective < 0.30:
            baseline += 0.18 * gdisf
        else:
            # High u7 at trial/appeal with evidence present = defendant wins (jury found against plaintiff).
            baseline -= 0.10 * gdisf

        # 3. Affirmative action / government defense (u6): when u2 is low, defense often prevails.
        if aff_def >= 0.15 and severity < 0.10:
            baseline -= 0.15
        elif aff_def >= 0.15 and severity < 0.30:
            baseline -= 0.08

        # --- Supporting evidence adjustments ---
        baseline += 0.06 * stats
        baseline += 0.08 * min(1.0, retal / 2.5)

        # --- Confidence calibration ---
        # For high-confidence signals, push closer to the rails.
        if severity >= 0.80 and corrective >= 0.50:
            # Cases where plaintiff had "specific" evidence but employer demonstrably
            # investigated/corrected consistently lose.
            return 0.18
        if severity <= 0.05 and corrective >= 0.30 and aff_def >= 0.15:
            return 0.75

        # --- Clamp ---
        return max(0.02, min(0.98, baseline))
    except Exception:
        return 0.5