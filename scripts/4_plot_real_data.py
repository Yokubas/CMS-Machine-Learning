import numpy as np
# from tensorflow.keras.models import load_model
import uproot
import awkward as ak
import os 
import vector
import matplotlib.pyplot as plt

txt_file = "data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt"

mc_dy_high = "data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_30000_file_index.txt"
mc_dy_low = "data/raw/signal/CMS_mc_RunIISummer20UL16NanoAODv9_DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8_NANOAODSIM_106X_mcRun2_asymptotic_v17-v1_2520000_file_index.txt"

root_folder = "data/real/"

mc_dy_signal_folder = "data/raw/signal/"

total_events = 85388673 

L_int = 8746231868.215154648 / 1e6; # pb^-1

sigmaDY = 6422.0 # pb

weight = (sigmaDY * L_int) / total_events

# model = load_model("results/electron_classifier.h5")

branches = [
    "Electron_pt",
    "Electron_eta",
    "Electron_phi",
    "Electron_mass",
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

def create_chain_from_file_list(txt_file, branches, max_files=-1, folder, max_events=None):
    files = np.loadtxt(txt_file, dtype=str)

    files = [os.path.join(folder, os.path.basename(f)) for f in files]
    
    if max_files != -1:
        files = files[:max_files]

    chain = uproot.concatenate(
        [f"{f}:Events" for f in files],
        expressions=branches,      
        entry_stop=max_events,     
        library="ak",               
    )
    
    return chain

tree = create_chain_from_file_list(
    txt_file,
    branches,
    root_folder,
    max_files=1
)


mc_dy_high_tree = create_chain_from_file_list(
    mc_dy_high,
    branches,
    mc_dy_signal_folder,
    max_files=1
)

mc_dy_low_data = create_chain_from_file_list(
    mc_dy_low,
    branches,
    mc_dy_signal_folder,
    max_files=1
)

mc_dy_combined = ak.concatenate([mc_dy_low_data, mc_dy_high_tree])
print("Total combined DY events:", len(mc_dy_combined))

if mc_dy_combined[ak.num(mc_dy_combined["Electron_pt"] >= 2)]:
    v = vector.awk(
        pt=tree["Electron_pt"],
        eta=tree["Electron_eta"],
        phi=tree["Electron_phi"],
        mass=tree["Electron_mass"]  # or zeros
    )

    # invariant mass of first two electrons
    m_inv = (v[:,0] + v[:,1]).mass


# print(len(tree))  # number of events
plt.figure(figsize=(8,6))
plt.hist(m_inv, bins=50, range=(0,200), weights=weight, histtype='step', label="DY MC")
plt.xlabel("Invariant mass [GeV]")
plt.ylabel("Events (scaled to L_int)")
plt.title("DY dilepton invariant mass")
plt.legend()
plt.grid(True)
plt.show()