from src.analysis_utils import (
    load_dataset,
    build_electrons,
    z_mass_numpy,
    process_mc,
    apply_nn
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import joblib

model_file = "results/electron_classifier_2.h5"
# model_file = "results/electron_classifier_adversarial.h5"

model = load_model(model_file)
scaler = joblib.load("results/scaler_2.pkl")
# scaler = joblib.load("results/scaler_adversarial.pkl")
# Using processed data files
real_data = "data/processed/real/real.root"

mc_dy_high_data = "data/processed/signal/mcDYhigh.root"
mc_dy_low_data = "data/processed/signal/mcDYlow.root"

total_events = 85388673 
entry = 843234

wsumHigh = 3.2887e+10
wsumLow = 8.26117e+10

wsum_ttbar = 6.69266e+09
wsum_tw = 5.10478e+07
wsum_aw = 2.98974e+07
wsum_st = 5.97701e+08
wsum_sa = 1.45713e+08
wsum_zz = 71000
wsum_wz = 2.127e+06
wsum_ww = 3.61803e+06

L_int = 8746231868.215154648 / 1e6; # pb^-1

sigmaDYhigh = 6422.0 # pb
sigmaDYlow = 20480.0 # pb
sigma_ttbar = 756.1
sigma_tw = 32.45    
sigma_aw = 32.51    
sigma_st = 119.7    
sigma_sa = 71.74
sigma_zz = 12.08    
sigma_wz = 27.56    
sigma_ww = 75.88

bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
                 115,120,126,133,141,150,160,171,185,200,220,243,273,
                 320,380,440,510,600,700,830,1000,1500,2000,3000])

# DY total
dy_low = process_mc(mc_dy_low_data, sigmaDYlow, wsumLow, "DY low")
dy_high = process_mc(mc_dy_high_data, sigmaDYhigh, wsumHigh, "DY high")

dy_total = {
    "mass": np.concatenate([dy_low["mass"], dy_high["mass"]]),
    "weights": np.concatenate([dy_low["weights"], dy_high["weights"]]),
    "label": r"DY $\rightarrow e^+ e^-$"
}

# --- DY tau total ---
dy_low_tau = process_mc("data/processed/background/mcDYlow_tau.root", sigmaDYlow, wsumLow, "DY low tau")
dy_high_tau = process_mc("data/processed/background/mcDYhigh_tau.root", sigmaDYhigh, wsumHigh, "DY high tau")

dy_tau_total = {
    "mass": np.concatenate([dy_low_tau["mass"], dy_high_tau["mass"]]),
    "weights": np.concatenate([dy_low_tau["weights"], dy_high_tau["weights"]]),
    "label": r"DY $\rightarrow \tau \tau$"
}

# --- Single top total ---
tw = process_mc("data/processed/background/tW.root", sigma_tw, wsum_tw, "tW")
aw = process_mc("data/processed/background/antitopW.root", sigma_aw, wsum_aw, "tWbar")
st = process_mc("data/processed/background/singletop.root", sigma_st, wsum_st, "st")
sa = process_mc("data/processed/background/sa.root", sigma_sa, wsum_sa, "sa")

single_top = {
    "mass": np.concatenate([tw["mass"], aw["mass"], st["mass"], sa["mass"]]),
    "weights": np.concatenate([tw["weights"], aw["weights"], st["weights"], sa["weights"]]),
    "label": "Single Top"
}

# --- TTbar ---
ttbar = process_mc("data/processed/background/ttbar.root", sigma_ttbar, wsum_ttbar, r"$t\bar{t}$")

# --- Diboson ---
zz = process_mc("data/processed/background/zz.root", sigma_zz, wsum_zz, "ZZ")
wz = process_mc("data/processed/background/wz.root", sigma_wz, wsum_wz, "WZ")
ww = process_mc("data/processed/background/ww.root", sigma_ww, wsum_ww, "WW")

mc_stack = [
    ww,
    wz,
    zz,
    single_top,
    ttbar,
    dy_tau_total,
    dy_total
]

events_real = load_dataset(real_data)
electrons_real, _ = build_electrons(events_real)
data_values = z_mass_numpy(electrons_real)

real_entries, edges = np.histogram(data_values, bins=bins)

masses = [s["mass"] for s in mc_stack]
weights = [s["weights"] for s in mc_stack]
labels = [s["label"] for s in mc_stack]

mc_entries = np.array([
    np.histogram(m["mass"], bins=bins, weights=m["weights"])[0]
    for m in mc_stack
])

mc_total_entries = np.abs(np.sum(mc_entries, axis=0))

bin_centers = 0.5 * (edges[:-1] + bins[1:])

fig, (ax, rax) = plt.subplots(
    2, 1,
    figsize=(8, 8),
    gridspec_kw={"height_ratios": [7, 2]},
    sharex=True
)

ax.hist(
    masses,
    bins=bins,
    weights=weights,
    stacked=True,
    label=labels,
    zorder = 10
)
ax.errorbar(
    bin_centers,
    real_entries,
    yerr=np.sqrt(real_entries),
    fmt='o',
    color='black',
    markersize=3,
    linewidth=1,
    label = "Data",
    zorder = 10
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Counts")
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True, bottom = True, left = True)
ax.grid(which='minor', axis='x', linestyle=':')
ax.grid(which = "major", linestyle=':')
ax.legend()
rax.set_xlim(bins[0]-10, bins[-1]+1000)

ratio = real_entries / mc_total_entries
ratio_err = np.sqrt(real_entries) / mc_total_entries

rax.errorbar(
    bin_centers,
    ratio,
    yerr=ratio_err,
    fmt='o',
    color='black',
    markersize=3,
    linewidth=1,
    capsize=2
)

rax.axhline(1.0, lw = 0.5, color='k')
rax.minorticks_on()
rax.tick_params(which='both', direction='in', top=True, right=True, bottom = True, left = True)
rax.grid(which='minor', axis='x', linestyle=':')
rax.grid(which = "major", linestyle=':')
rax.set_ylabel("Data / Pred.")
rax.set_xlabel("Mass [GeV]")
rax.set_xscale("log")
rax.set_ylim(0.5, 1.5)
rax.set_xlim(bins[0]-10, bins[-1]+1000)
plt.tight_layout()
plt.show()

# Real vs MC after NN selection
threshold = 0.7

events_real_nn = apply_nn(events_real, model=model, scaler=scaler, threshold=threshold)
electrons_real_nn, _ = build_electrons(events_real_nn)
mass_real_nn = z_mass_numpy(electrons_real_nn)

# --- DY ---
dy_low_nn = process_mc(mc_dy_low_data, sigmaDYlow, wsumLow,
                      r"DY $\rightarrow e^+ e^-$",
                      apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_high_nn = process_mc(mc_dy_high_data, sigmaDYhigh, wsumHigh,
                       r"DY $\rightarrow e^+ e^-$",
                       apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_total_nn = {
    "mass": np.concatenate([dy_low_nn["mass"], dy_high_nn["mass"]]),
    "weights": np.concatenate([dy_low_nn["weights"], dy_high_nn["weights"]]),
    "label": r"DY $\rightarrow e^+ e^-$"
}

# --- DY tau tau ---
dy_low_tau_nn = process_mc("data/processed/background/mcDYlow_tau.root", sigmaDYlow, wsumLow,
                           r"DY $\rightarrow \tau \tau$",
                           apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_high_tau_nn = process_mc("data/processed/background/mcDYhigh_tau.root", sigmaDYhigh, wsumHigh,
                            r"DY $\rightarrow \tau \tau$",
                            apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_tau_total_nn = {
    "mass": np.concatenate([dy_low_tau_nn["mass"], dy_high_tau_nn["mass"]]),
    "weights": np.concatenate([dy_low_tau_nn["weights"], dy_high_tau_nn["weights"]]),
    "label": r"DY $\rightarrow \tau \tau$"
}

tw_nn = process_mc("data/processed/background/tW.root", sigma_tw, wsum_tw,
                   "tW", True, model, scaler, threshold)
aw_nn = process_mc("data/processed/background/antitopW.root", sigma_aw, wsum_aw,
                   "tWbar", True, model, scaler, threshold)
st_nn = process_mc("data/processed/background/singletop.root", sigma_st, wsum_st,
                   "st", True, model, scaler, threshold)
sa_nn = process_mc("data/processed/background/sa.root", sigma_sa, wsum_sa,
                   "sa", True, model, scaler, threshold)

single_top_nn = {
    "mass": np.concatenate([tw_nn["mass"], aw_nn["mass"], st_nn["mass"], sa_nn["mass"]]),
    "weights": np.concatenate([tw_nn["weights"], aw_nn["weights"], st_nn["weights"], sa_nn["weights"]]),
    "label": "Single Top"
}

ttbar_nn = process_mc("data/processed/background/ttbar.root",
                      sigma_ttbar, wsum_ttbar,
                      r"$t\bar{t}$", True, model, scaler, threshold)

zz_nn = process_mc("data/processed/background/zz.root",
                   sigma_zz, wsum_zz, "ZZ", True, model, scaler, threshold)

wz_nn = process_mc("data/processed/background/wz.root",
                   sigma_wz, wsum_wz, "WZ", True, model, scaler, threshold)

ww_nn = process_mc("data/processed/background/ww.root",
                   sigma_ww, wsum_ww, "WW", True, model, scaler, threshold)

mc_stack_nn = [
    ww_nn,
    wz_nn,
    zz_nn,
    single_top_nn,
    ttbar_nn,
    dy_tau_total_nn,
    dy_total_nn
]

real_entries_nn, edges = np.histogram(mass_real_nn, bins=bins)

masses_nn = [s["mass"] for s in mc_stack_nn]
weights_nn = [s["weights"] for s in mc_stack_nn]
labels_nn = [s["label"] for s in mc_stack_nn]

mc_entries = np.array([
    np.histogram(m["mass"], bins=bins, weights=m["weights"])[0]
    for m in mc_stack_nn
])

mc_total_entries_nn = np.abs(np.sum(mc_entries, axis=0))

bin_centers = 0.5 * (edges[:-1] + bins[1:])

fig, (ax, rax) = plt.subplots(
    2, 1,
    figsize=(8, 8),
    gridspec_kw={"height_ratios": [7, 2]},
    sharex=True
)

ax.hist(
    masses_nn,
    bins=bins,
    weights=weights_nn,
    stacked=True,
    label=labels_nn,
    zorder = 10
)
ax.errorbar(
    bin_centers,
    real_entries_nn,
    yerr=np.sqrt(real_entries_nn),
    fmt='o',
    color='black',
    markersize=3,
    linewidth=1,
    label = "Data",
    zorder = 10
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel("Counts")
ax.minorticks_on()
ax.tick_params(which='both', direction='in', top=True, right=True, bottom = True, left = True)
ax.grid(which='minor', axis='x', linestyle=':')
ax.grid(which = "major", linestyle=':')
ax.legend()
rax.set_xlim(bins[0]-10, bins[-1]+1000)

ratio = real_entries_nn / mc_total_entries_nn
ratio_err = np.sqrt(real_entries_nn) / mc_total_entries_nn

rax.errorbar(
    bin_centers,
    ratio,
    yerr=ratio_err,
    fmt='o',
    color='black',
    markersize=3,
    linewidth=1,
    capsize=2
)

rax.axhline(1.0, lw = 0.5, color='k')
rax.minorticks_on()
rax.tick_params(which='both', direction='in', top=True, right=True, bottom = True, left = True)
rax.grid(which='minor', axis='x', linestyle=':')
rax.grid(which = "major", linestyle=':')
rax.set_ylabel("Data / Pred.")
rax.set_xlabel("Mass [GeV]")
rax.set_xscale("log")
rax.set_ylim(0.5, 1.5)
rax.set_xlim(bins[0]-10, bins[-1]+1000)
plt.tight_layout()
plt.show()