import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_dataset(root_file, max_events=None):
    
    file = uproot.open(root_file)
    tree = file["Events"]
    data = tree.arrays(library = "ak", entry_stop = max_events)

    return data

def build_electrons(events):
  
    electron = ak.zip(
        {
            "pt": events["Electron_pt"],
            "eta": events["Electron_eta"],
            "phi": events["Electron_phi"],
            "mass": events["Electron_mass"] if "Electron_mass" in events.fields else 0 * events["Electron_pt"],
            "energy": np.sqrt(events["Electron_pt"]**2 * np.cosh(events["Electron_eta"])**2), 
        },
        with_name="Momentum4D"
    )
    weights = events["genWeight"] if "genWeight" in events.fields else None
    return electron, weights

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

def prepare_input(arr):
    df = pd.DataFrame()

    df["nElectron"] = ak.to_numpy(ak.count_nonzero(arr["Electron_pt"] > 0, axis=1))

    padded_pt  = ak.pad_none(arr["Electron_pt"],  2)
    padded_eta = ak.pad_none(arr["Electron_eta"], 2)

    for i in range(2):
        df[f"Electron{i+1}_pt"]  = ak.to_numpy(ak.fill_none(padded_pt[:,  i],  0))
        df[f"Electron{i+1}_eta"] = ak.to_numpy(ak.fill_none(padded_eta[:, i], 0))

    df["nJet"] = ak.to_numpy(ak.count_nonzero(arr["Jet_pt"] > 0, axis=1))
    
    padded_pt  = ak.pad_none(arr["Jet_pt"], 4)
    padded_eta = ak.pad_none(arr["Jet_eta"], 4)
    padded_phi = ak.pad_none(arr["Jet_phi"], 4)
    padded_btag = ak.pad_none(arr["Jet_btagDeepFlavB"], 4)
    
    for i in range(4):
        df[f"Jet{i+1}_pt"] = ak.to_numpy(ak.fill_none(padded_pt[:, i], 0))
        df[f"Jet{i+1}_eta"] = ak.to_numpy(ak.fill_none(padded_eta[:, i], 0))
        df[f"Jet{i+1}_phi"] = ak.to_numpy(ak.fill_none(padded_phi[:, i], 0))
        df[f"Jet{i+1}_btag"] = ak.to_numpy(ak.fill_none(padded_btag[:, i], 0))

    return df.reset_index(drop=True)

def process_mc(file, sigma, wsum, label, L_int = 8746231868.215154648 / 1e6, entry = 843234, total_events = 85388673):
    events = load_dataset(file)
    electrons, weights = build_electrons(events)

    mass = z_mass_numpy(electrons)

    scale = (sigma * L_int) * (entry/total_events) / wsum

    return {
        "label": label,
        "mass": mass,
        "weights": weights * scale,
        "events": events
    }

def apply_nn(events, model = model, threshold = threshold, scaler = scaler):
    prepared = prepare_input(events)
    X = prepared.values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled, batch_size=1024)
    mask = y_pred.flatten() > threshold
    return events[mask]

def plot_hist(values, bins, xlabel, ylabel="Events", color = 'r', histtype= 'bar', label = None, title=None, logx = False, logy = False, weights = None):
    # plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, label = label, weights = weights, histtype = histtype, zorder = 10, color = color)
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