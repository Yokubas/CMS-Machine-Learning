from src.analysis_utils import (
    load_dataset,
    build_electrons,
    select_two_opposite_sign_same_flavour,
    z_mass_numpy,
    plot_hist
)
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
# uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.XRootDSource

txt_file = "data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt"

mc_dy_high = "data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt"
mc_dy_low = "data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt"

root_folder = "data/real/"

mc_dy_signal_folder = "data/raw/signal/"

total_events = 85388673 

L_int = 8746231868.215154648 / 1e6; # pb^-1

sigmaDYhigh = 6422.0 # pb
sigmaDYlow = 20480.0 # pb
# model = load_model("results/electron_classifier.h5")

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

mc_dy_low_data = load_dataset(mc_dy_low, branches, folder = mc_dy_signal_folder, max_files=1)

mc_dy_high_data = load_dataset(mc_dy_high, branches, folder = mc_dy_signal_folder, max_files=1)

mc_dy_combined = ak.concatenate([mc_dy_low_data, mc_dy_high_data])

# print("Total combined DY events:", len(mc_dy_combined))


electrons_high = build_electrons(mc_dy_high_data)
electrons_low = build_electrons(mc_dy_low_data)

two_el_high = select_two_opposite_sign_same_flavour(electrons_high)
two_el_low = select_two_opposite_sign_same_flavour(electrons_low)

Z_ee_mass1 = z_mass_numpy(two_el_high)
Z_ee_mass2 = z_mass_numpy(two_el_low)

entry_events = len(two_el_high) + len(two_el_low)

bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
                 115,120,126,133,141,150,160,171,185,200,220,243,273,
                 320,380,440,510,600,700,830,1000,1500,2000,3000])

weights_padded = ak.pad_none(electrons_high.weight, 1, axis=1)
weights_filled = ak.fill_none(weights_padded, 0)

event_weights1 = ak.firsts(weights_filled)

wsum = ak.sum(event_weights1)

scale1 = (sigmaDYhigh * L_int) * (entry_events/total_events) / wsum
print(scale1)

weights_events = ak.firsts(two_el_high.weight)
plt.figure(figsize=(7, 5))
plot_hist(
    Z_ee_mass1,
    bins=bins,
    weights=weights_events*scale1,
    label = "Drell-Yan MC",
    xlabel="Mass [GeV]",
    title="Z → ee Invariant Mass",
    logx=True,
    logy=True
)

weights_padded = ak.pad_none(electrons_low.weight, 1, axis=1)
weights_filled = ak.fill_none(weights_padded, 0)

# Now take the first element safely
event_weights = ak.firsts(weights_filled)

wsum = ak.sum(event_weights)

scale2 = (sigmaDYlow * L_int) * (entry_events/total_events) / wsum
weights_events = ak.firsts(two_el_low.weight)

plot_hist(
    Z_ee_mass2,
    bins=bins,
    weights=weights_events*scale2,
    # label = "Drell-Yan MC",
    xlabel="Mass [GeV]",
    title="Z → ee Invariant Mass",
    logx=True,
    logy=True
)
plt.show()