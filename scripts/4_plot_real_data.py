from src.analysis_utils import (
    load_dataset,
    build_electrons,
    z_mass_numpy,
    plot_hist,
    prepare_input
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib

model_file = "results/electron_classifier.h5"
model = load_model(model_file)
scaler = joblib.load("results/scaler.pkl")
# Using processed data files
real_data = "data/processed/real/real.root"

mc_dy_high_data = "data/processed/signal/mcDYhigh.root"
mc_dy_low_data = "data/processed/signal/mcDYlow.root"

total_events = 85388673 

wsumHigh = 3.2887e+10
wsumLow = 8.26117e+10

L_int = 8746231868.215154648 / 1e6; # pb^-1

sigmaDYhigh = 6422.0 # pb
sigmaDYlow = 20480.0 # pb

branches = [
    "Electron_pt",
    "Electron_eta",
    "Electron_phi",
    "Electron_mass",
    "Electron_charge",
    "genWeight",
    "Electron_miniPFRelIso_all",
    "Electron_miniPFRelIso_chg",
    "Electron_dz",
    "Electron_dxy",
    "Electron_ip3d",
    "Jet_pt",
    "Jet_eta",
    "Jet_phi",
    "Jet_btagDeepFlavB"
]

events_real = load_dataset(real_data)
electrons, _ = build_electrons(events_real)

plt.hist(z_mass_numpy(electrons), bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
                 115,120,126,133,141,150,160,171,185,200,220,243,273,
                 320,380,440,510,600,700,830,1000,1500,2000,3000]))
plt.xscale("log")
plt.yscale("log")
plt.savefig("results/real.png")
plt.show()
prepared = prepare_input(events_real)

X_real = prepared.values
X_scaled = scaler.transform(X_real)  # <-- scale using the saved scaler

y_pred = model.predict(X_scaled, batch_size=1024)  # choose batch_size to fit memory
threshold = 0.8  # example cutoff
selected_mask = y_pred.flatten() > threshold
selected_events_real = events_real[selected_mask]
electrons, _ = build_electrons(selected_events_real)
Z_mass = z_mass_numpy(electrons)

bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
                 115,120,126,133,141,150,160,171,185,200,220,243,273,
                 320,380,440,510,600,700,830,1000,1500,2000,3000])

mc_dy_low = load_dataset(root_file = mc_dy_low_data)

electrons_low, weights_low = build_electrons(mc_dy_low)

Z_ee_mass_low = z_mass_numpy(electrons_low)
entry_events = len(electrons_low)
scale_low = (sigmaDYlow * L_int) * (entry_events/total_events) / wsumLow


mc_dy_high = load_dataset(root_file = mc_dy_high_data)

electrons_high, weights_high = build_electrons(mc_dy_high)

Z_ee_mass_high = z_mass_numpy(electrons_high)
entry_events = len(electrons_high)
scale_high = (sigmaDYhigh * L_int) * (entry_events/total_events) / wsumHigh

mass_mc = np.concatenate([Z_ee_mass_low, Z_ee_mass_high])
weights_mc = np.concatenate([weights_low*scale_low, weights_high*scale_high])

plt.figure(figsize=(7, 5))
plot_hist(
    mass_mc,
    bins=bins,
    weights=weights_mc,
    label = "Drell-Yan MC",
    xlabel="Mass [GeV]",
    title="Z → ee Invariant Mass",
    logx=True,
    logy=True
)

plot_hist(
    Z_mass, 
    bins=bins, 
    xlabel="Invariant Mass [GeV]",
    label="Real data (NN-selected)",
    title="Z → ee Invariant Mass",
    histtype="step",
    color = 'k',
    logx=True,
    logy=True
)
plt.savefig("results/mc_vs_real.png")
plt.show()