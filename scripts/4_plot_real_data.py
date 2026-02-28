import numpy as np
from tensorflow.keras.models import load_model
import uproot

txt_file = "CMS_Run2016H_DoubleEG_NANOAOD_UL2016_MiniAODv2_NanoAODv9-v1_100000_file_index.txt"

model = load_model("results/electron_classifier.h5")

def create_chain_from_file_list(txt_file, max_files = -1):
    files = np.loadtxt(txt_file, dtype=str)
    
    if max_files != -1:
        files = files[:max_files]

    # Create a "chained" view of all files
    chain = uproot.concatenate(
        [f"{f}:Events" for f in files],
        library="np"
    )

    return chain