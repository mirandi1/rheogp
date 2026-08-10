"""
rheogp.results
==============
`FitResult` — a plain-numpy container holding *every* quantity a
fitted model produces, so figures can be made entirely outside the
package.

Usage
-----
>>> res = model.results()
>>> res.t, res.stress, res.prediction          # fit panel
>>> res.features[:, 0]                         # physical x_1(t)
>>> res.sensitivities["V"]["mean"]             # dσ̄/dx_1(t)
>>> res.omega, res.Gp, res.Gdp                 # model G*, G**
>>> res.fft_omega, res.fft_Gp, res.fft_Gdp     # FFT estimates
>>> res.history["loss"], res.history["params"] # training traces
>>> res.save("outputs/run1")                   # .npz + .json
>>> res2 = FitResult.load("outputs/run1")
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

__all__ = ["FitResult"]


@dataclass
class FitResult:
    # metadata -----------------------------------------------------
    model:       str = ""
    gp_kernel:   str = ""
    control:     str = "strain"           # "strain" | "stress"
    convolution: str = "causal"           # "causal" | "steady"
    quadrature:  str = "exact"

    # time-domain signals -----------------------------------------
    t:              np.ndarray = None
    strain:         np.ndarray = None
    stress:         np.ndarray = None
    input:          np.ndarray = None     # signal driving the features
    target:         np.ndarray = None     # signal the GP fits
    prediction:     np.ndarray = None     # posterior mean of target
    prediction_std: np.ndarray = None     # posterior std of target

    # physics-informed features (physical units, raw input) --------
    features:      np.ndarray = None      # (N, n_features)
    feature_names: list = field(default_factory=list)

    # sensitivities / prefactors -----------------------------------
    # {name: {mean, lower, upper, samples, scalar_mean,
    #         scalar_lower, scalar_upper}}
    sensitivities: dict = field(default_factory=dict)

    # parameters ---------------------------------------------------
    phys_params:    dict = field(default_factory=dict)
    derived_params: dict = field(default_factory=dict)
    metrics:        dict = field(default_factory=dict)

    # frequency domain --------------------------------------------
    omega: np.ndarray = None
    Gp:    np.ndarray = None
    Gdp:   np.ndarray = None
    fft_omega: np.ndarray = None
    fft_Gp:    np.ndarray = None
    fft_Gdp:   np.ndarray = None

    # training -----------------------------------------------------
    history: dict = field(default_factory=dict)

    # -------------------------------------------------------------
    def to_dict(self):
        return asdict(self)

    @property
    def tan_delta(self):
        """Loss tangent  G''/G'  on the model frequency grid."""
        return self.Gdp / self.Gp

    def scalar_prefactors(self):
        return {k: v["scalar_mean"] for k, v in self.sensitivities.items()}

    # -------------------------------------------------------------
    def save(self, path):
        """Save to <path>/result.npz + <path>/result.json."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)

        arrays = {}
        for key in ("t", "strain", "stress", "input", "target",
                    "prediction", "prediction_std", "features",
                    "omega", "Gp", "Gdp",
                    "fft_omega", "fft_Gp", "fft_Gdp"):
            v = getattr(self, key)
            if v is not None:
                arrays[key] = np.asarray(v)
        for name, d in self.sensitivities.items():
            for stat in ("mean", "lower", "upper", "samples"):
                if stat in d and d[stat] is not None:
                    arrays[f"sens__{name}__{stat}"] = np.asarray(d[stat])
        for k, v in self.history.items():
            if k == "params":
                for pk, pv in v.items():
                    arrays[f"hist__param__{pk}"] = np.asarray(pv)
            else:
                arrays[f"hist__{k}"] = np.asarray(v)
        np.savez_compressed(root / "result.npz", **arrays)

        meta = dict(
            model=self.model, gp_kernel=self.gp_kernel,
            control=self.control, convolution=self.convolution,
            quadrature=self.quadrature,
            feature_names=self.feature_names,
            phys_params=self.phys_params,
            derived_params=self.derived_params,
            metrics=self.metrics,
            sensitivity_scalars={
                name: {k: float(d[k]) for k in
                       ("scalar_mean", "scalar_lower", "scalar_upper")}
                for name, d in self.sensitivities.items()
            },
        )
        with open(root / "result.json", "w") as f:
            json.dump(meta, f, indent=2, default=float)
        return root

    # -------------------------------------------------------------
    @classmethod
    def load(cls, path):
        root = Path(path)
        with open(root / "result.json") as f:
            meta = json.load(f)
        data = np.load(root / "result.npz")

        res = cls(
            model=meta["model"], gp_kernel=meta["gp_kernel"],
            control=meta["control"], convolution=meta["convolution"],
            quadrature=meta.get("quadrature", "exact"),
            feature_names=meta["feature_names"],
            phys_params=meta["phys_params"],
            derived_params=meta["derived_params"],
            metrics=meta["metrics"],
        )
        for key in ("t", "strain", "stress", "input", "target",
                    "prediction", "prediction_std", "features",
                    "omega", "Gp", "Gdp",
                    "fft_omega", "fft_Gp", "fft_Gdp"):
            if key in data:
                setattr(res, key, data[key])

        sens = {}
        for k in data.files:
            if k.startswith("sens__"):
                _, name, stat = k.split("__")
                sens.setdefault(name, {})[stat] = data[k]
        for name, scal in meta.get("sensitivity_scalars", {}).items():
            sens.setdefault(name, {}).update(scal)
        res.sensitivities = sens

        hist = {"params": {}}
        for k in data.files:
            if k.startswith("hist__param__"):
                hist["params"][k.split("__", 2)[2]] = data[k]
            elif k.startswith("hist__"):
                hist[k.split("__", 1)[1]] = data[k]
        res.history = hist
        return res
