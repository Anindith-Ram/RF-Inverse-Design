"""
Evaluation metrics for RF inverse design models.

All functions operate on numpy arrays. The PRIMARY metric for model selection
is round-trip MSE — component MSE is secondary/diagnostic.
"""

import numpy as np
from scipy import signal as sci_signal

FREQ_HZ = np.linspace(40e9, 90e9, 101)
FREQ_HZ_OTFL301 = np.linspace(18e9, 46e9, 101)   # OTFL301 band: 24–40 GHz + 6 GHz transition inset


def component_mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean squared error on predicted g-values vs ground truth.

    Ignores NaN positions (used for N=3 samples where g5, g6 are undefined).

    Args:
        y_pred: predicted g-values, shape (n_samples, n_gvals) or (n_gvals,)
        y_true: ground-truth g-values, same shape; NaN where undefined

    Returns:
        mse: scalar float (dimensionless, g-values are dimensionless)
    """
    mask = ~np.isnan(y_true)
    return float(np.mean((y_pred[mask] - y_true[mask]) ** 2))


def component_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Mean absolute error on predicted g-values vs ground truth.

    Ignores NaN positions.

    Args:
        y_pred: predicted g-values, shape (n_samples, n_gvals) or (n_gvals,)
        y_true: ground-truth g-values, same shape; NaN where undefined

    Returns:
        mae: scalar float (dimensionless)
    """
    mask = ~np.isnan(y_true)
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask])))


def r2_per_component(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    R² (coefficient of determination) per g-value position.

    Args:
        y_pred: predicted g-values, shape (n_samples, n_gvals)
        y_true: ground-truth g-values, shape (n_samples, n_gvals); NaN where undefined

    Returns:
        r2: array of shape (n_gvals,); NaN where fewer than 2 valid samples
    """
    n_cols = y_true.shape[1]
    r2 = np.full(n_cols, np.nan)
    for k in range(n_cols):
        mask = ~np.isnan(y_true[:, k])
        if mask.sum() < 2:
            continue
        ss_res = np.sum((y_true[mask, k] - y_pred[mask, k]) ** 2)
        ss_tot = np.sum((y_true[mask, k] - y_true[mask, k].mean()) ** 2)
        if ss_tot < 1e-10:
            # Zero-variance target (e.g., gN+1 = 1.0 always for odd N) — R² undefined
            continue
        r2[k] = 1.0 - ss_res / ss_tot
    return r2


def _infer_ripple_from_g1(g1: float, N: int) -> float:
    """
    Infer Chebyshev ripple_dB from the first g-value g1 and filter order N.

    Inverts: g1 = 2*sin(π/(2N)) / sinh(β/(2N))
    where β = ln(coth(ripple_dB * ln(10) / 40)).

    Args:
        g1: first prototype g-value (dimensionless)
        N: filter order (integer >= 1)

    Returns:
        ripple_dB: inferred passband ripple in dB, clamped to [0.01, 5.0]
    """
    a1 = np.sin(np.pi / (2.0 * N))
    sinh_val = max(2.0 * a1 / max(float(g1), 1e-6), 1e-9)
    beta = 2.0 * N * np.arcsinh(sinh_val)
    arg = np.clip(np.exp(-beta), 1e-15, 1.0 - 1e-15)
    ripple_pred = 40.0 * np.arctanh(arg) / np.log(10.0)
    return float(np.clip(ripple_pred, 0.01, 5.0))


def synthesize_from_gvalues(
    g_pred: np.ndarray,
    N: int,
    fc_GHz: float,
    fbw: float,
    freq_hz: np.ndarray = None,
) -> tuple:
    """
    Synthesize S-parameter curves from predicted g-values.

    Infers ripple_dB from g_pred[0] via the Chebyshev inverse formula,
    then calls scipy.signal.cheby1 with (N, ripple_pred, fc_GHz, fbw).

    Args:
        g_pred: predicted g-values, array (N+1,)
        N: filter order (integer)
        fc_GHz: center frequency in GHz
        fbw: fractional bandwidth (dimensionless)
        freq_hz: frequency grid in Hz, shape (101,); defaults to 40–90 GHz

    Returns:
        s21_db: synthesized S21 in dB, shape (101,); or None on failure
        s11_db: synthesized S11 in dB, shape (101,); or None on failure
    """
    if freq_hz is None:
        freq_hz = FREQ_HZ

    ripple_pred = _infer_ripple_from_g1(float(g_pred[0]), N)
    fc_hz = fc_GHz * 1e9
    f_low = fc_hz * (1.0 - fbw / 2.0)
    f_high = fc_hz * (1.0 + fbw / 2.0)
    Wn = [2.0 * np.pi * f_low, 2.0 * np.pi * f_high]

    try:
        b, a = sci_signal.cheby1(N, ripple_pred, Wn, btype='bandpass', analog=True)
        _, h = sci_signal.freqs(b, a, worN=2.0 * np.pi * freq_hz)
        h_abs = np.abs(h)
        if np.max(h_abs) < 1e-30:
            return None, None
        h_norm = h / np.max(h_abs)
        s21_db = 20.0 * np.log10(np.abs(h_norm) + 1e-12)
        s11_db = 20.0 * np.log10(
            np.sqrt(np.maximum(1.0 - np.abs(h_norm) ** 2, 1e-12)) + 1e-12
        )
        return s21_db, s11_db
    except Exception:
        return None, None


# ── LC-based synthesis (realistic dataset) ───────────────────────────────────

_Z0 = 50.0   # reference impedance (Ohm)


def _abcd_series(Z: complex) -> np.ndarray:
    return np.array([[1.0, Z], [0.0, 1.0]], dtype=complex)


def _abcd_shunt(Y: complex) -> np.ndarray:
    return np.array([[1.0, 0.0], [Y, 1.0]], dtype=complex)


def _abcd_to_s21_s11(M: np.ndarray) -> tuple:
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    denom = A + B / _Z0 + _Z0 * C + D
    if np.abs(denom) < 1e-30:
        return 0.0 + 0j, 1.0 + 0j
    return 2.0 / denom, (A + B / _Z0 - _Z0 * C - D) / denom


def _compute_L_eff(L_vals: np.ndarray, coupling_M: np.ndarray, N: int) -> np.ndarray:
    """Apply mutual inductance to get effective per-element inductance."""
    L_eff = L_vals.copy().astype(float)
    for k in range(N - 1):
        L_eff[k]     += float(coupling_M[k])
        L_eff[k + 1] += float(coupling_M[k])
    return L_eff


def synthesize_from_lc(
    L_vals: np.ndarray,
    C_vals: np.ndarray,
    N: int,
    fc_GHz: float,
    Q_L: np.ndarray = None,
    Q_C: np.ndarray = None,
    freq_hz: np.ndarray = None,
    *,
    coupling_M: np.ndarray = None,
    C_sub: np.ndarray = None,
    alpha_C: np.ndarray = None,
) -> tuple:
    """
    Synthesize S-parameter curves from physical LC values via ABCD cascade.

    Loss model (used when Q_L and Q_C are provided):
        Inductor: R_s(f) = (omega0·L_eff/Q_L) · sqrt(f/f0)  (skin-effect)
        Capacitor: R_esr = (1/(omega0·C·Q_C)) · (f/f0)^alpha_C  (freq-dep loss)
    If Q_L/Q_C are None, uses ideal (lossless) elements.

    Optional physics kwargs (all three must be provided together for parasitic path):
        coupling_M: (N-1,) mutual inductances — modifies L_eff per element
        C_sub:      (N,)   substrate parasitic capacitances — shunt path per element
        alpha_C:    (N,)   freq-dep Q_C exponent per element

    When all three are None (default), falls back to basic ABCD synthesis (backward
    compatible). Pass zero arrays to explicitly request basic behaviour.

    Normalizes S21 peak to 0 dB.

    Args:
        L_vals:    (N,) inductances in Henries
        C_vals:    (N,) capacitances in Farads
        N:         filter order
        fc_GHz:    center frequency in GHz
        Q_L:       (N,) inductor Q-factors at fc; None → lossless
        Q_C:       (N,) capacitor Q-factors at fc; None → lossless
        freq_hz:   frequency grid in Hz; defaults to 40–90 GHz (101 pts)
        coupling_M: optional (N-1,) mutual inductances (H)
        C_sub:      optional (N,) substrate capacitances (F)
        alpha_C:    optional (N,) freq-dep Q_C exponents (dimensionless)

    Returns:
        s21_db: (101,) S21 in dB, or None on failure
        s11_db: (101,) S11 in dB, or None on failure
    """
    if freq_hz is None:
        freq_hz = FREQ_HZ

    fc_hz  = fc_GHz * 1e9
    omega0 = 2.0 * np.pi * fc_hz
    f0     = fc_hz
    lossless = (Q_L is None) or (Q_C is None)

    use_parasitics = (coupling_M is not None) and (C_sub is not None) and (alpha_C is not None)

    # Pre-compute effective inductances if mutual coupling provided
    L_eff = _compute_L_eff(L_vals, coupling_M, N) if use_parasitics else np.array(L_vals, dtype=float)

    try:
        s21_db = np.zeros(len(freq_hz))
        s11_db = np.zeros(len(freq_hz))

        for fi, f in enumerate(freq_hz):
            omega = 2.0 * np.pi * f
            M = np.eye(2, dtype=complex)

            for k in range(N):
                Lk_eff = float(L_eff[k])
                Ck     = float(C_vals[k])

                if lossless:
                    Z_L = 1j * omega * Lk_eff
                    Z_C = 1.0 / (1j * omega * Ck)
                elif use_parasitics:
                    R_s   = (omega0 * abs(Lk_eff) / float(Q_L[k])) * np.sqrt(f / f0)
                    R_esr = (1.0 / (omega0 * Ck * float(Q_C[k]))) * (f / f0) ** float(alpha_C[k])
                    Z_L = R_s + 1j * omega * Lk_eff
                    Z_C = R_esr + 1.0 / (1j * omega * Ck)
                else:
                    R_s   = (omega0 * Lk_eff / float(Q_L[k])) * np.sqrt(f / f0)
                    R_esr = 1.0 / (omega0 * Ck * float(Q_C[k]))
                    Z_L = R_s + 1j * omega * Lk_eff
                    Z_C = R_esr + 1.0 / (1j * omega * Ck)

                Z_res = Z_L + Z_C

                if use_parasitics:
                    if (k + 1) % 2 == 1:
                        # Series resonator: shunt C_sub before element
                        M = M @ _abcd_shunt(1j * omega * float(C_sub[k]))
                        M = M @ _abcd_series(Z_res)
                    else:
                        # Shunt resonator: C_sub in parallel with shunt admittance
                        M = M @ _abcd_shunt(1.0 / Z_res + 1j * omega * float(C_sub[k]))
                else:
                    if (k + 1) % 2 == 1:
                        M = M @ _abcd_series(Z_res)
                    else:
                        M = M @ _abcd_shunt(1.0 / Z_res)

            s21_c, s11_c = _abcd_to_s21_s11(M)
            s21_db[fi] = 20.0 * np.log10(np.abs(s21_c) + 1e-12)
            s11_db[fi] = 20.0 * np.log10(np.abs(s11_c) + 1e-12)

        peak = np.max(s21_db)
        s21_db -= peak
        s21_lin_norm = 10.0 ** (s21_db / 20.0)
        s11_db = 20.0 * np.log10(
            np.sqrt(np.maximum(1.0 - s21_lin_norm ** 2, 1e-12)) + 1e-12
        )
        return s21_db, s11_db

    except Exception:
        return None, None


def roundtrip_mse_lc(
    L_pred: np.ndarray,
    C_pred: np.ndarray,
    N: int,
    fc_GHz: float,
    Q_L: np.ndarray,
    Q_C: np.ndarray,
    target_s21_db: np.ndarray,
    target_s11_db: np.ndarray,
    freq_hz: np.ndarray = None,
    *,
    coupling_M: np.ndarray = None,
    C_sub: np.ndarray = None,
    alpha_C: np.ndarray = None,
) -> float:
    """
    Round-trip MSE for LC predictions.

    Synthesizes S-params from predicted L/C using ground-truth Q values (and
    optionally ground-truth parasitic metadata), then computes MSE against the
    stored target S-params.

    Using ground-truth Q (and parasitics) isolates LC prediction error. Without
    parasitics, the eval synthesis is basic ABCD — valid for clean-physics datasets
    but introduces a structural floor on realistic datasets.

    Args:
        L_pred:        (N,) predicted inductances in Henries
        C_pred:        (N,) predicted capacitances in Farads
        N:             filter order
        fc_GHz:        center frequency in GHz
        Q_L:           (N,) ground-truth inductor Q-factors
        Q_C:           (N,) ground-truth capacitor Q-factors
        target_s21_db: (101,) target S21 in dB
        target_s11_db: (101,) target S11 in dB
        freq_hz:       frequency grid; defaults to 40–90 GHz (101 pts)
        coupling_M:    optional (N-1,) ground-truth mutual inductances (H)
        C_sub:         optional (N,)   ground-truth substrate capacitances (F)
        alpha_C:       optional (N,)   ground-truth freq-dep Q_C exponents

    Returns:
        mse: scalar float (dB²); inf if synthesis failed
    """
    if freq_hz is None:
        freq_hz = FREQ_HZ

    s21_pred, s11_pred = synthesize_from_lc(
        L_pred, C_pred, N, fc_GHz, Q_L, Q_C, freq_hz,
        coupling_M=coupling_M, C_sub=C_sub, alpha_C=alpha_C,
    )
    if s21_pred is None:
        return float('inf')

    target = np.concatenate([target_s21_db, target_s11_db])
    pred   = np.concatenate([s21_pred, s11_pred])
    return float(np.mean((pred - target) ** 2))


def roundtrip_mse(
    g_pred: np.ndarray,
    N: int,
    fc_GHz: float,
    fbw: float,
    ripple_dB: float,
    freq_hz: np.ndarray,
    target_s21_db: np.ndarray = None,
    target_s11_db: np.ndarray = None,
) -> float:
    """
    Round-trip MSE: MSE between target S-params and S-params synthesized from g_pred.

    Ripple is inferred from g_pred[0] (not taken from ripple_dB) so the metric
    is sensitive to prediction error. The ripple_dB argument is used only to
    build the target when target_s21_db / target_s11_db are not provided.

    Args:
        g_pred: predicted g-values, array (N+1,)
        N: filter order (integer)
        fc_GHz: center frequency in GHz
        fbw: fractional bandwidth (dimensionless)
        ripple_dB: ground-truth ripple in dB (builds target if not supplied)
        freq_hz: frequency grid in Hz, shape (101,)
        target_s21_db: target S21 in dB, shape (101,); optional
        target_s11_db: target S11 in dB, shape (101,); optional

    Returns:
        mse: scalar float (dB²); inf if synthesis failed
    """
    if target_s21_db is None or target_s11_db is None:
        import warnings
        fc_hz = fc_GHz * 1e9
        f_low = fc_hz * (1.0 - fbw / 2.0)
        f_high = fc_hz * (1.0 + fbw / 2.0)
        Wn = [2.0 * np.pi * f_low, 2.0 * np.pi * f_high]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b, a = sci_signal.cheby1(N, ripple_dB, Wn, btype='bandpass', analog=True)
                _, h = sci_signal.freqs(b, a, worN=2.0 * np.pi * freq_hz)
            h_norm = h / np.max(np.abs(h))
            target_s21_db = 20.0 * np.log10(np.abs(h_norm) + 1e-12)
            target_s11_db = 20.0 * np.log10(
                np.sqrt(np.maximum(1.0 - np.abs(h_norm) ** 2, 1e-12)) + 1e-12
            )
        except Exception:
            return float('inf')

    s21_pred, s11_pred = synthesize_from_gvalues(g_pred, N, fc_GHz, fbw, freq_hz)
    if s21_pred is None:
        return float('inf')

    target = np.concatenate([target_s21_db, target_s11_db])
    pred = np.concatenate([s21_pred, s11_pred])
    return float(np.mean((pred - target) ** 2))
