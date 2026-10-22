"""
rheogp.preprocessing
=====================
Generic, numpy-only signal-cleaning utilities for chirp (or any
transient) rheometry records, factored out of the ad-hoc multichirp
fitting script so they are importable, documented and unit-tested
like the rest of the package.

Typical use (see `examples/multichirp_master_curve.py` for the full
pipeline, including the experiment-specific CSV loading and
temperature parsing that stay in the example rather than the
package):

    from rheogp.preprocessing import clean_chirp, parse_temperature

    T = parse_temperature("chirp_02_T90C.csv")
    t_clean, strain_clean, stress_clean = clean_chirp(t, strain, stress)
"""

import re
import os

import numpy as np

__all__ = ["rolling_rms", "clean_chirp", "parse_temperature"]


def rolling_rms(x, nwin):
    """Rolling RMS of `x` over a window of `nwin` samples (same length
    as `x`; the first `nwin - 1` samples are padded with the first
    valid value rather than a partial/biased window)."""
    x = np.asarray(x, dtype=float)
    if nwin <= 1:
        return np.abs(x)
    nwin = min(nwin, len(x))
    x2 = np.r_[0.0, np.cumsum(x ** 2)]
    rms = np.sqrt((x2[nwin:] - x2[:-nwin]) / nwin)
    return np.r_[np.full(nwin - 1, rms[0] if len(rms) else 0.0), rms]


def clean_chirp(t, strain, stress, discard_s=0.5, pad_s=0.5,
                detrend=True):
    """
    Clean a single chirp record matching the BOTTS paper methodology:
    1. Drop finite/non-increasing time points.
    2. Discard the initial `discard_s` (default 0.5s) to eliminate instrument startup artifacts.
    3. Fit a least-squares linear trend using only the first `pad_s` and 
       last `pad_s` of the remaining signal (the static padding segments).
    4. Subtract that linear trend from the entire strain and stress arrays.

    Parameters
    ----------
    t, strain, stress : array-like, same length
    discard_s : float -- duration [s] of startup artifact to discard at t=0
    pad_s     : float -- duration [s] of pre-chirp and post-chirp padding used to fit linear drift
    detrend   : bool -- if False, skip steps 3-4 entirely: the record is
                only cleaned of non-finite/non-monotonic samples, truncated
                by `discard_s`, and re-referenced to t=0. Nothing is
                subtracted from strain or stress.

                Use this when the waveform has no usable TRAILING pad.
                Steps 3-4 assume both ends of the record are static, and
                fit one line through both. A chirp that tapers to zero
                only in its last few samples has no trailing pad of usable
                length, so a nonzero `pad_s` feeds live signal into the
                fit and the "drift" removed is partly real material
                response. Note that `pad_s=0` does NOT avoid this: the pad
                mask then selects 2 points, falls through to the 4-point
                fallback below, and still fits and subtracts a line --
                through the first two and LAST two samples.

    Returns
    -------
    t_clean, strain_clean, stress_clean : np.ndarray
        `t_clean` starts at 0.
    """
    t = np.asarray(t, dtype=float)
    strain = np.asarray(strain, dtype=float)
    stress = np.asarray(stress, dtype=float)
    
    if not (t.shape == strain.shape == stress.shape):
        raise ValueError("t, strain, stress must have the same shape.")

    # Drop NaNs / Inf and enforce chronological time order
    mask = np.isfinite(t) & np.isfinite(strain) & np.isfinite(stress)
    t, strain, stress = t[mask], strain[mask], stress[mask]
    order = np.argsort(t)
    t, strain, stress = t[order], strain[order], stress[order]
    
    dt_ok = np.r_[True, np.diff(t) > 1e-12]
    t, strain, stress = t[dt_ok], strain[dt_ok], stress[dt_ok]

    # 1. Discard the initial startup artifact (first 0.5s)
    t_start = t[0] + discard_s
    valid_idx = t >= t_start
    t, strain, stress = t[valid_idx], strain[valid_idx], stress[valid_idx]

    if len(t) < 8:
        raise ValueError("clean_chirp: insufficient samples remaining after startup truncation.")

    # Reset time array to start at t=0
    t2 = t - t[0]
    
    if not detrend:
        return t2, strain, stress

    # 2. Identify the padding indices (first pad_s and last pad_s of signal)
    t_max = t2[-1]
    pad_mask = (t2 <= pad_s) | (t2 >= (t_max - pad_s))

    if np.sum(pad_mask) < 4:
        # Fallback if sampling density is sparse
        pad_mask = np.zeros_like(t2, dtype=bool)
        pad_mask[:2] = True
        pad_mask[-2:] = True

    # 3. Fit linear trend (y = m*t + c) using ONLY the padding segments
    X_pad = np.c_[t2[pad_mask], np.ones(np.sum(pad_mask))]
    coef_g, *_ = np.linalg.lstsq(X_pad, strain[pad_mask], rcond=None)
    coef_s, *_ = np.linalg.lstsq(X_pad, stress[pad_mask], rcond=None)

    # 4. Subtract linear detrending function from the full dataset
    s2 = strain - (coef_g[0] * t2 + coef_g[1])
    q2 = stress - (coef_s[0] * t2 + coef_s[1])

    return t2, s2, q2

def parse_temperature(filename, pattern=r"[Tt]\s*([+-]?\d+(?:\.\d+)?)\s*[Cc]"):
    """
    Extract a temperature (float, °C) from a filename such as
    'chirp_02_T90C.csv' (case-insensitive).  Returns None if the
    pattern is not found; pass a custom `pattern` for other naming
    conventions.
    """
    m = re.search(pattern, os.path.basename(str(filename)))
    return float(m.group(1)) if m else None
