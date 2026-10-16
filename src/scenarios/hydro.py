"""
hydro.py  —  Hydrological building blocks for the synthetic scenario generator.

All time quantities are explicit about units. STEP_MIN = 15.

Kirpich (1940), metric form
---------------------------
    Tc[min] = 0.0195 * L[m]^0.77 * S[m/m]^-0.385

The coefficient 0.0195 with L in metres returns MINUTES (the imperial
form 0.0078 * L[ft]^0.77 * S^-0.385 also returns minutes). The manuscript's
Eq. 9 text ("Tc in hours") is wrong; the code returns hours explicitly as
kirpich_tc_hours() = minutes / 60 and converts to 15-min steps only in
hours_to_steps(). tests/test_hydro.py checks a hand-computed value.

SCS triangular unit hydrograph (NEH-4 / NEH-630 Ch.16)
-----------------------------------------------------
    lag  = 0.6 * Tc
    Tp   = D/2 + lag          (D = rainfall time step)
    Tb   = 2.67 * Tp          (standard; the original code used 2.67*Tc)

The manuscript disclosed Tb = 2.67*Tc as a study-specific
simplification. It is replaced by the standard relation here; set
`base_factor_on='tc'` to reproduce the old shape for comparison.
"""
from __future__ import annotations

import numpy as np

STEP_MIN = 15
STEP_HR = STEP_MIN / 60.0


def kirpich_tc_minutes(L_m: float, S: float) -> float:
    if L_m <= 0 or S <= 0:
        raise ValueError(f"Kirpich needs L>0, S>0 (got L={L_m}, S={S})")
    return 0.0195 * L_m ** 0.77 * S ** -0.385


def kirpich_tc_hours(L_m: float, S: float) -> float:
    return kirpich_tc_minutes(L_m, S) / 60.0


def hours_to_steps(hours: float) -> int:
    return int(round(hours / STEP_HR))


def scs_triangle_times(tc_hr: float, d_hr: float = STEP_HR,
                       base_factor_on: str = "tp") -> tuple[float, float]:
    """Return (Tp_hr, Tb_hr)."""
    tp = d_hr / 2.0 + 0.6 * tc_hr
    if base_factor_on == "tp":
        tb = 2.67 * tp
    elif base_factor_on == "tc":          # legacy behaviour, for comparison only
        tb = 2.67 * tc_hr
    else:
        raise ValueError(base_factor_on)
    return tp, tb


def unit_triangle_kernel(tc_hr: float, n_steps: int,
                         base_factor_on: str = "tp") -> np.ndarray:
    """Unit-PEAK triangular kernel sampled at 15-min steps.

    k[0] corresponds to the step in which rain falls. Peak (=1) occurs at
    Tp; the kernel returns to zero at Tb. Sampling uses the continuous
    triangle evaluated at step centres so that very short Tc still gives
    a non-degenerate kernel.
    """
    tp, tb = scs_triangle_times(tc_hr, base_factor_on=base_factor_on)
    t = (np.arange(n_steps) + 0.5) * STEP_HR
    k = np.where(t <= tp, t / tp, np.clip((tb - t) / (tb - tp), 0.0, None))
    k = np.clip(k, 0.0, 1.0)
    return k / k.max() if k.max() > 0 else k


def stage_response(excess_rain_mm: np.ndarray, peak_rate_m_per_mm: float,
                   tc_hr: float, base_factor_on: str = "tp") -> np.ndarray:
    """Linear (convolution) stage response to an excess-rain series.

    Calibration meaning of peak_rate: median(peak stage rise / event rain
    depth) over historical events. A convolution with a unit-peak kernel
    reproduces exactly that ratio for a short pulse (duration << Tp) and
    stays linear for longer storms. The response begins when the rain
    begins; it is NOT additionally delayed by Tp (the original S1/S5 code
    started the kernel at pulse_onset + Tp, delaying the peak to ~2*Tp).
    """
    n = len(excess_rain_mm)
    k = unit_triangle_kernel(tc_hr, n, base_factor_on)
    return peak_rate_m_per_mm * np.convolve(excess_rain_mm, k)[:n]


def route_delta(delta_src: np.ndarray, lag_steps: int,
                attenuation: float) -> np.ndarray:
    """Fixed-lag, constant-attenuation routing of a stage DELTA (Eq. 10)."""
    out = np.zeros_like(delta_src)
    lag = max(int(lag_steps), 0)
    if lag == 0:
        out[:] = attenuation * delta_src
    elif lag < len(delta_src):
        out[lag:] = attenuation * delta_src[:-lag]
    return out


def samuels_backwater_length_m(depth_m: float, bed_slope: float) -> float:
    """Samuels (1989) backwater length scale L ~ 0.7 * D / S0 (m)."""
    s = max(float(bed_slope), 1e-5)
    return 0.7 * max(float(depth_m), 0.05) / s
