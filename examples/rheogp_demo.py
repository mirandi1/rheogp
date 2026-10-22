"""
rheogp_demo.py  —  all 8 models
"""
from rheogp import SPGP
from rheogp.utils import (
    compare_models,
    make_synthetic_maxwell, make_synthetic_springpot,
    make_synthetic_fmg, make_synthetic_fml, make_synthetic_fmm,
    make_synthetic_fkvs, make_synthetic_fkvd, make_synthetic_fkv,
)

COMMON = dict(n=1500, dt=1e-3, noise=0.02)
FIT    = dict(inducing_points=200, n_epochs=5_000, patience=300, verbose=True)

datasets = {
    "Maxwell":                make_synthetic_maxwell(Gc=100, tau_c=0.5,            **COMMON),
    "Springpot":              make_synthetic_springpot(V=100, alpha=0.5,           **COMMON),
    "FractionalMaxwellGel":   make_synthetic_fmg(V=80, G=100, alpha=0.5,           **COMMON),
    "FractionalMaxwellLiquid":make_synthetic_fml(G_bb=80, eta=100, beta=0.5,       **COMMON),
    "FractionalMaxwell":      make_synthetic_fmm(V=80, G_bb=80, alpha=0.7,
                                                  beta=0.3,                        **COMMON),
    "FractionalKelvinVoigtS": make_synthetic_fkvs(V=80, G=60, alpha=0.5,           **COMMON),
    "FractionalKelvinVoigtD": make_synthetic_fkvd(G_bb=80, eta=60, beta=0.4,       **COMMON),
    "FractionalKelvinVoigt":  make_synthetic_fkv(V=80, G_bb=60, alpha=0.7,
                                                  beta=0.3,                        **COMMON),
}

for model_name, (t, strain, stress) in datasets.items():
    print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
    m = SPGP(model=model_name, omega_min=0.01, omega_max=200, **FIT)
    m.fit(t, strain, stress)
    print(m.summary())
    m.plot_fit()
    m.plot_prefactors()
    m.plot_Gstar()

# model comparison on FKV data
t, strain, stress = make_synthetic_fkv(V=80, G_bb=60, alpha=0.7, beta=0.3, **COMMON)
fitted = {}
for name in ["Springpot", "FractionalMaxwell",
             "FractionalKelvinVoigt", "FractionalKelvinVoigtS"]:
    fitted[name] = SPGP(model=name, omega_min=0.01, omega_max=200, **FIT
                        ).fit(t, strain, stress)
best = compare_models(fitted)
print(best.summary())