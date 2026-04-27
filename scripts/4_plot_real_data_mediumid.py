from src.analysis_utils import (
    load_dataset,
    build_electrons,
    z_mass_numpy,
    process_mc,
    apply_nn,
    plot_data_vs_mc
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import joblib

model_file = "results/electron_classifier_2.h5"
model = load_model(model_file)
scaler = joblib.load("results/scaler_2.pkl")
# Using processed data files
real_data = "data/processed/real/real_id.root"

mc_dy_high_data = "data/processed/signal/mcDYhigh_id.root"
mc_dy_low_data = "data/processed/signal/mcDYlow_id.root"

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
dy_low_tau = process_mc("data/processed/background/mcDYlow_id_tau_id.root", sigmaDYlow, wsumLow, "DY low tau")
dy_high_tau = process_mc("data/processed/background/mcDYhigh_id_tau_id.root", sigmaDYhigh, wsumHigh, "DY high tau")

dy_tau_total = {
    "mass": np.concatenate([dy_low_tau["mass"], dy_high_tau["mass"]]),
    "weights": np.concatenate([dy_low_tau["weights"], dy_high_tau["weights"]]),
    "label": r"DY $\rightarrow \tau \tau$"
}

# --- Single top total ---
tw = process_mc("data/processed/background/tW_id.root", sigma_tw, wsum_tw, "tW")
aw = process_mc("data/processed/background/antitopW_id.root", sigma_aw, wsum_aw, "tWbar")
st = process_mc("data/processed/background/singletop_id.root", sigma_st, wsum_st, "st")
sa = process_mc("data/processed/background/sa_id.root", sigma_sa, wsum_sa, "sa")

single_top = {
    "mass": np.concatenate([tw["mass"], aw["mass"], st["mass"], sa["mass"]]),
    "weights": np.concatenate([tw["weights"], aw["weights"], st["weights"], sa["weights"]]),
    "label": "Single Top"
}

# --- TTbar ---
ttbar = process_mc("data/processed/background/ttbar_id.root", sigma_ttbar, wsum_ttbar, r"$t\bar{t}$")

# --- Diboson ---
zz = process_mc("data/processed/background/zz_id.root", sigma_zz, wsum_zz, "ZZ")
wz = process_mc("data/processed/background/wz_id.root", sigma_wz, wsum_wz, "WZ")
ww = process_mc("data/processed/background/ww_id.root", sigma_ww, wsum_ww, "WW")

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

plot_data_vs_mc(
    data_values=data_values,
    mc_stack=mc_stack,
    bins=bins,
    title = "Before NN selection with mediumID"
)
plt.savefig("results/stack_before_nn_id.png")
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
dy_low_tau_nn = process_mc("data/processed/background/mcDYlow_id_tau_id.root", sigmaDYlow, wsumLow,
                           r"DY $\rightarrow \tau \tau$",
                           apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_high_tau_nn = process_mc("data/processed/background/mcDYhigh_id_tau_id.root", sigmaDYhigh, wsumHigh,
                            r"DY $\rightarrow \tau \tau$",
                            apply_nn_flag=True, model=model, scaler=scaler, threshold=threshold)

dy_tau_total_nn = {
    "mass": np.concatenate([dy_low_tau_nn["mass"], dy_high_tau_nn["mass"]]),
    "weights": np.concatenate([dy_low_tau_nn["weights"], dy_high_tau_nn["weights"]]),
    "label": r"DY $\rightarrow \tau \tau$"
}

tw_nn = process_mc("data/processed/background/tW_id.root", sigma_tw, wsum_tw,
                   "tW", True, model, scaler, threshold)
aw_nn = process_mc("data/processed/background/antitopW_id.root", sigma_aw, wsum_aw,
                   "tWbar", True, model, scaler, threshold)
st_nn = process_mc("data/processed/background/singletop_id.root", sigma_st, wsum_st,
                   "st", True, model, scaler, threshold)
sa_nn = process_mc("data/processed/background/sa_id.root", sigma_sa, wsum_sa,
                   "sa", True, model, scaler, threshold)

single_top_nn = {
    "mass": np.concatenate([tw_nn["mass"], aw_nn["mass"], st_nn["mass"], sa_nn["mass"]]),
    "weights": np.concatenate([tw_nn["weights"], aw_nn["weights"], st_nn["weights"], sa_nn["weights"]]),
    "label": "Single Top"
}

ttbar_nn = process_mc("data/processed/background/ttbar_id.root",
                      sigma_ttbar, wsum_ttbar,
                      r"$t\bar{t}$", True, model, scaler, threshold)

zz_nn = process_mc("data/processed/background/zz_id.root",
                   sigma_zz, wsum_zz, "ZZ", True, model, scaler, threshold)

wz_nn = process_mc("data/processed/background/wz_id.root",
                   sigma_wz, wsum_wz, "WZ", True, model, scaler, threshold)

ww_nn = process_mc("data/processed/background/ww_id.root",
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

plot_data_vs_mc(
    data_values=mass_real_nn,
    mc_stack=mc_stack_nn,
    bins=bins,
    title = "After NN selection with mediumID [Default]"
)
plt.savefig("results/stack_after_nn_id_default.png")
plt.show()