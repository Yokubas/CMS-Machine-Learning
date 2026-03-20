from src.analysis_utils import (
    load_dataset,
    build_electrons,
    z_mass_numpy,
    plot_hist,
    prepare_input,
    process_mc
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # type: ignore
import joblib

model_file = "results/electron_classifier.h5"
model = load_model(model_file)
scaler = joblib.load("results/scaler.pkl")
# Using processed data files
real_data = "data/processed/real/real.root"

mc_dy_high_data = "data/processed/signal/mcDYhigh.root"
mc_dy_low_data = "data/processed/signal/mcDYlow.root"

total_events = 85388673 
entry = 843234

wsumHigh = 3.2887e+10
wsumLow = 8.26117e+10

wsum_ttbar = 
wsum_tw =
wsum_aw =
wsum_st =
wsum_sa =
wsum_zz =
wsum_wz =
wsum_ww =

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

mc_samples = []

mc_samples.append(process_mc(mc_dy_low_data, sigmaDYlow, wsumLow, "DY low"))
mc_samples.append(process_mc(mc_dy_high_data, sigmaDYhigh, wsumHigh, "DY high"))

mc_samples.append(process_mc("data/processed/background/ttbar.root", sigma_ttbar, wsum_ttbar, "ttbar"))
mc_samples.append(process_mc("data/processed/background/tW.root", sigma_tw, wsum_tw, "tW"))
mc_samples.append(process_mc("data/processed/background/antitopW.root", sigma_aw, wsum_aw, "tWbar"))
mc_samples.append(process_mc("data/processed/background/singletop.root", sigma_st, wsum_st, "single top"))
mc_samples.append(process_mc("data/processed/background/sa.root", sigma_sa, wsum_sa, "single antitop"))
mc_samples.append(process_mc("data/processed/background/zz.root", sigma_zz, wsum_zz, "ZZ"))
mc_samples.append(process_mc("data/processed/background/wz.root", sigma_wz, wsum_wz, "WZ"))
mc_samples.append(process_mc("data/processed/background/ww.root", sigma_ww, wsum_ww, "WW"))

events_real = load_dataset(real_data)
electrons_real, _ = build_electrons(events_real)

plt.figure(figsize=(7,5))

masses = [s["mass"] for s in mc_samples]
weights = [s["weights"] for s in mc_samples]
labels = [s["label"] for s in mc_samples]

plt.hist(
    masses,
    bins=bins,
    weights=weights,
    stacked=True,
    label=labels
)

# real data
plt.hist(
    z_mass_numpy(electrons_real),
    bins=bins,
    histtype="step",
    color="black",
    label="Real Data"
)

plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.title("Before NN selection")

plt.savefig("results/stack_before_nn.png")
plt.show()

# events_real = load_dataset(real_data)
# electrons_real, _ = build_electrons(events_real)

# prepared = prepare_input(events_real)

# X_real = prepared.values
# X_scaled = scaler.transform(X_real)  # <-- scale using the saved scaler

# y_pred = model.predict(X_scaled, batch_size=1024)  # choose batch_size to fit memory
# threshold = 0.8  # example cutoff
# selected_mask = y_pred.flatten() > threshold
# print(selected_mask)
# selected_events_real = events_real[selected_mask]
# electrons, _ = build_electrons(selected_events_real)
# Z_mass = z_mass_numpy(electrons)

# bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
#                  115,120,126,133,141,150,160,171,185,200,220,243,273,
#                  320,380,440,510,600,700,830,1000,1500,2000,3000])
# plt.hist(z_mass_numpy(electrons_real), bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
#                  115,120,126,133,141,150,160,171,185,200,220,243,273,
#                  320,380,440,510,600,700,830,1000,1500,2000,3000]), label = "Real Data")
# plt.hist(Z_mass, bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
#                  115,120,126,133,141,150,160,171,185,200,220,243,273,
#                  320,380,440,510,600,700,830,1000,1500,2000,3000]), label = "Real Data (NN-selected)")
# plt.xscale("log")
# plt.yscale("log")
# plt.legend()
# plt.savefig("results/real.png")
# plt.show()

# mc_dy_low = load_dataset(root_file = mc_dy_low_data)

# electrons_low, weights_low = build_electrons(mc_dy_low)

# Z_ee_mass_low = z_mass_numpy(electrons_low)
# scale_low = (sigmaDYlow * L_int) * (entry/total_events) / wsumLow

# mc_dy_high = load_dataset(root_file = mc_dy_high_data)

# electrons_high, weights_high = build_electrons(mc_dy_high)

# Z_ee_mass_high = z_mass_numpy(electrons_high)
# scale_high = (sigmaDYhigh * L_int) * (entry/total_events) / wsumHigh

# mass_mc = np.concatenate([Z_ee_mass_low, Z_ee_mass_high])
# weights_mc = np.concatenate([weights_low*scale_low, weights_high*scale_high])

# plt.figure(figsize=(7, 5))
# plot_hist(
#     mass_mc,
#     bins=bins,
#     weights=weights_mc,
#     label = "Drell-Yan MC",
#     xlabel="Mass [GeV]",
#     title="Z → ee Invariant Mass",
#     logx=True,
#     logy=True
# )

# plot_hist(
#     Z_mass, 
#     bins=bins, 
#     xlabel="Invariant Mass [GeV]",
#     label="Real Data (NN-selected)",
#     title="Z → ee Invariant Mass",
#     histtype="step",
#     color = 'k',
#     logx=True,
#     logy=True
# )
# plt.savefig("results/mc_vs_real.png")
# plt.show()