import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import os

def load_dataset(txt_file, branches, folder, max_files=-1, max_events=None):
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

def build_electrons(events):
  
    electron = ak.zip(
        {
            "pt": events["Electron_pt"],
            "eta": events["Electron_eta"],
            "phi": events["Electron_phi"],
            "mass": events["Electron_mass"] if "Electron_mass" in events.fields else 0 * events["Electron_pt"],
            "energy": np.sqrt(events["Electron_pt"]**2 * np.cosh(events["Electron_eta"])**2),
            "charge": events["Electron_charge"],
            "weight": events["genWeight"]
        },
        with_name="Momentum4D"
    )

    return electron

def select_two_opposite_sign_same_flavour(leps):
    mask_two = (ak.num(leps) == 2)
    leps2 = leps[mask_two]
    mask_os = leps2.charge[:,0] * leps2.charge[:,1] < 0
    
    return leps2[mask_os]

def z_mass_numpy(leps):
    # Convert to numpy
    l0 = ak.to_numpy(leps[:,0])
    l1 = ak.to_numpy(leps[:,1])

    # compute px, py, pz, E in numpy
    pxZ = l0["pt"]*np.cos(l0["phi"]) + l1["pt"]*np.cos(l1["phi"])
    pyZ = l0["pt"]*np.sin(l0["phi"]) + l1["pt"]*np.sin(l1["phi"])
    pzZ = l0["pt"]*np.sinh(l0["eta"]) + l1["pt"]*np.sinh(l1["eta"])
    EZ  = l0["energy"] + l1["energy"]

    return np.sqrt(EZ**2 - pxZ**2 - pyZ**2 - pzZ**2)

def plot_hist(values, bins, xlabel, ylabel="Events", label = None, title=None, logx = False, logy = False, weights = None):
    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, label = label, weights = weights, zorder = 10, color = 'r')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.grid(True)
    # plt.xlim(0, 400)
    plt.tight_layout()
    plt.legend()
    # plt.show()