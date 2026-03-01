import numpy as np
from tensorflow.keras.models import load_model
import uproot
import awkward as ak
import os 

txt_file = "data/real/CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt"
root_folder = "data/real/"

model = load_model("results/electron_classifier.h5")

branches = [
    "Electron_pt",
    "Electron_eta",
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

def create_chain_from_file_list(txt_file, branches, max_files=-1, max_events=None):
    files = np.loadtxt(txt_file, dtype=str)

    files = [os.path.join(root_folder, os.path.basename(f)) for f in files]
    
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
    max_files=1
)

print(len(tree))  # number of events
