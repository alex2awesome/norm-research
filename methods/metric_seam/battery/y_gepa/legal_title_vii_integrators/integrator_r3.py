def combine(units: dict, text: str) -> float:
    """
    Deterministic integrator for Title VII plaintiff-favorability forecast.
    Higher output => more likely the court rules FOR the plaintiff (finds discrimination).

    Key insight from train data: u7 (genuine dispute) is the dominant signal at SJ stage.
    u1 <= 1 marks the strongest procedural tilt (Motion to Dismiss / SJ with plaintiff evidence).
    At u1 <= 1, u7 >= 0.50 has a very high win rate regardless of u2/u3/u4/u5/u6.

    Domain logic:
    - Strong procedural baseline (u1<=1, u7 high) => plaintiff wins (~0.78).
    - High employer corrective action (u3) is defense-favorable but muted at SJ.
    - Severe harassment (u2) strongly favors plaintiff (concrete evidence in record).
    - u4/u5 add supporting weight via gated accumulation.
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

        proc      = _c("u1", 0.0, 10.0)
        severity  = _c("u2")
        corrective = _c("u3", 0.0, 10.0) / 10.0
        retal     = _c("u4")
        stats     = _c("u5")
        aff_def   = _c("u6")
        gdisf     = _p("u7")

        # --- Procedural Baseline ---
        if proc <= 1.5:
            baseline = 0.50 + 0.24 * gdisf
        elif proc <= 2.5:
            baseline = 0.42 + 0.15 * gdisf
        elif proc <= 5.5:
            baseline = 0.30 + 0.05 * gdisf
        else:
            baseline = 0.22 + 0.02 * gdisf

        # --- Harassment Severity (concrete evidence of discrimination) ---
        if severity >= 0.65:
            baseline += 0.18
        elif severity >= 0.40:
            baseline += 0.10
        elif severity >= 0.20:
            baseline += 0.04
        elif severity <= 0.02 and gdisf < 0.45 and stats < 0.12 and retal < 0.30:
            baseline -= 0.08

        # --- Employer Corrective Action (defense-favorable) ---
        if corrective >= 0.70:
            baseline -= 0.10
        elif corrective >= 0.45:
            baseline -= 0.05

        # --- Supporting Evidence (accumulation) ---
        if retal >= 0.60:
            baseline += 0.07
        elif retal >= 0.35:
            baseline += 0.03
        if stats >= 0.25:
            baseline += 0.05
        elif stats >= 0.10:
            baseline += 0.02

        # --- Affirmative Action / Government Defense ---
        if aff_def >= 0.15 and severity < 0.20:
            baseline -= 0.07

        # --- Unusual Feature Bonus ---
        features = 0
        if severity >= 0.40: features += 1
        if retal >= 0.35: features += 1
        if stats >= 0.10: features += 1
        if features >= 3:
            baseline += 0.06

        return max(0.02, min(0.98, baseline))
    except Exception:
        return 0.5