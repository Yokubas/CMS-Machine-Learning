import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from src.plot_training import plot_training_history, plot_auc
import joblib
from src.analysis_utils import (
    load_dataset,
    prepare_training,
    build_electrons,
    z_mass_numpy
)
# signal
mcDYhigh = load_dataset("data/processed/signal/mcDYhigh.root")
mcDYlow = load_dataset("data/processed/signal/mcDYlow.root")

# backgound
# --- DY -> TauTau --- 
dy_low_tau = load_dataset("data/processed/background/mcDYlow_tau.root")
dy_high_tau = load_dataset("data/processed/background/mcDYhigh_tau.root")

# --- Single top total ---
tw = load_dataset("data/processed/background/tW.root")
aw = load_dataset("data/processed/background/antitopW.root")
st = load_dataset("data/processed/background/singletop.root")
sa = load_dataset("data/processed/background/sa.root")

# --- TTbar ---
ttbar = load_dataset("data/processed/background/ttbar.root")

# --- Diboson ---
zz = load_dataset("data/processed/background/zz.root")
wz = load_dataset("data/processed/background/wz.root")
ww = load_dataset("data/processed/background/ww.root")

wjets = load_dataset("data/processed/background/wjets.root")

# --- QCD ---
qcd1 = load_dataset("data/processed/background/qcd1.root")
qcd2 = load_dataset("data/processed/background/qcd2.root")
qcd3 = load_dataset("data/processed/background/qcd3.root")
qcd4 = load_dataset("data/processed/background/qcd4.root")
qcd5 = load_dataset("data/processed/background/qcd5.root")
qcd6 = load_dataset("data/processed/background/qcd6.root")
qcd7 = load_dataset("data/processed/background/qcd7.root")
qcd8 = load_dataset("data/processed/background/qcd8.root")

df_bkg = pd.concat([
    prepare_training(dy_low_tau, label = 0, process = "dy_low_tau"),
    prepare_training(dy_high_tau, label = 0, process = "dy_high_tau"),
    prepare_training(tw, label = 0, process = "tw"),
    prepare_training(aw, label = 0, process = "aw"),
    prepare_training(st, label = 0, process="st"),
    prepare_training(sa, label = 0, process="sa"),
    prepare_training(ttbar, label = 0, process="ttbar"),
    prepare_training(zz, label = 0, process="zz"),
    prepare_training(wz, label = 0, process="wz"),
    prepare_training(ww, label = 0, process="ww"),
    prepare_training(wjets, label = 0, process="wjets"),
    prepare_training(qcd1, label = 0, process="qcd1"),
    prepare_training(qcd2, label = 0, process="qcd2"),
    prepare_training(qcd3, label = 0, process="qcd3"),
    prepare_training(qcd4, label = 0, process="qcd4"),
    prepare_training(qcd5, label = 0, process="qcd5"),
    prepare_training(qcd6, label = 0, process="qcd6"),
    prepare_training(qcd7, label = 0, process="qcd7"),
    prepare_training(qcd8, label = 0, process="qcd8")
], ignore_index=True)

df_signal = pd.concat([
    prepare_training(mcDYhigh, label = 1, process = "mcDYhigh"),
    prepare_training(mcDYlow, label = 1, process = "mcDYlow")
], ignore_index=True)

electrons_signal_high, _ = build_electrons(mcDYhigh)
electrons_signal_low, _ = build_electrons(mcDYlow)

masses_signal = np.concatenate([
    z_mass_numpy(electrons_signal_high),
    z_mass_numpy(electrons_signal_low)
])

bins = np.array([40,45,50,55,60,64,68,72,76,81,86,91,96,101,106,110,
                 115,120,126,133,141,150,160,171,185,200,220,243,273,
                 320,380,440,510,600,700,830,1000,1500,2000,3000])
# background_samples = [
#     dy_low_tau,
#     dy_high_tau,
#     tw,
#     aw,
#     st,
#     sa,
#     ttbar,
#     zz,
#     wz,
#     ww,
#     wjets,
#     qcd1,
#     qcd2,
#     qcd3,
#     qcd4,
#     qcd5,
#     qcd6,
#     qcd7,
#     qcd8
# ]

# masses_background = []

# for sample in background_samples:
#     electrons_bkg, _ = build_electrons(sample)
#     masses_bkg = z_mass_numpy(electrons_bkg)
#     masses_background.append(masses_bkg)

# masses_background = np.concatenate(masses_background)

# --------------------------------------------------
# Signal weights (flatten signal)
# --------------------------------------------------

indices_signal = np.digitize(masses_signal, bins)
indices_signal = np.clip(indices_signal, 1, len(bins) - 1)

counts_signal = np.bincount(
    indices_signal,
    minlength=len(bins) + 1
)

counts_signal[counts_signal == 0] = 1

weights_signal = 1.0 / np.sqrt(counts_signal)
weights_signal = weights_signal[indices_signal]
weights_signal = weights_signal / np.mean(weights_signal)

# --------------------------------------------------
# Background weights (flatten background too)
# --------------------------------------------------

# indices_background = np.digitize(masses_background, bins)
# indices_background = np.clip(indices_background, 1, len(bins) - 1)

# counts_background = np.bincount(
#     indices_background,
#     minlength=len(bins) + 1
# )

# counts_background[counts_background == 0] = 1

# weights_background = 1.0 / np.sqrt(counts_background)
# weights_background = weights_background[indices_background]
# weights_background = weights_background / np.mean(weights_background)

# --------------------------------------------------
# Final training dataframe
# --------------------------------------------------

df_train = pd.concat([df_signal, df_bkg], ignore_index=True)
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna()

# --------------------------------------------------
# Sample weights for BOTH signal + background
# --------------------------------------------------

sample_weights = np.ones(len(df_train))

signal_mask = df_train["label"] == 1
qcd1_mask = df_train["process"] == "qcd1"
qcd2_mask = df_train["process"] == "qcd2"
qcd3_mask = df_train["process"] == "qcd3"
qcd4_mask = df_train["process"] == "qcd4"
qcd5_mask = df_train["process"] == "qcd5"
qcd6_mask = df_train["process"] == "qcd6"
qcd7_mask = df_train["process"] == "qcd7"
qcd8_mask = df_train["process"] == "qcd8"


sample_weights[signal_mask] = weights_signal
weight = 5
# upweight QCD strongly
sample_weights[qcd1_mask] *= weight
sample_weights[qcd2_mask] *= weight
sample_weights[qcd3_mask] *= weight
sample_weights[qcd4_mask] *= weight
sample_weights[qcd5_mask] *= weight
sample_weights[qcd6_mask] *= weight
sample_weights[qcd7_mask] *= weight
sample_weights[qcd8_mask] *= weight

sample_weights = sample_weights / np.mean(sample_weights)
# background_mask = df_train["label"] == 0

assert len(weights_signal) == signal_mask.sum()
# assert len(weights_background) == background_mask.sum()

# sample_weights[signal_mask] = weights_signal
# sample_weights[background_mask] = weights_background

# Optional: normalize global weights
# sample_weights = sample_weights / np.mean(sample_weights)

# # Separate features and labels
X = df_train.drop(columns = ["label", "process"]).to_numpy()
y = df_train["label"].to_numpy()
# Scale features to mean 0 and std 1 for stable and efficient training (does not get biased towards larger numerical values)

scaler = StandardScaler()

# Fit on training data and transform
X = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, "results/scaler_mass.pkl")

# Split into training and test sets
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42, shuffle = True
)

# # Build feed-forward neural network
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X.shape[1],)), # Number of features per event
#     tf.keras.layers.Dense(64),                  # First hidden layer with 64 neurons (fully connected)
#     tf.keras.layers.LeakyReLU(),                # Activation function: introduces non-linearity; allows small gradient for negative inputs to keep learning
#     tf.keras.layers.Dropout(0.1),               # Pridetas

#     tf.keras.layers.Dense(32),                  # Second hidden layer with 32 neurons
#     tf.keras.layers.LeakyReLU(),                # Same as above, helps network learn complex patterns

#     tf.keras.layers.Dense(16),                  # Third hidden layer with 16 neurons
#     tf.keras.layers.LeakyReLU(),                # Same purpose: non-linearity + stable learning for negative inputs
#     tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer: 1 neuron, sigmoid activation
# ])                                                  # Outputs a probability between 0 and 1
#                                                     # 1 → signal, 0 → background

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),

    tf.keras.layers.Dense(48, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(24, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(12),
    tf.keras.layers.LeakyReLU(),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X.shape[1],)),

#     tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.LeakyReLU(),
#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.LeakyReLU(),
#     tf.keras.layers.Dropout(0.2),

#     tf.keras.layers.Dense(12),
#     tf.keras.layers.LeakyReLU(),

#     tf.keras.layers.Dense(1, activation="sigmoid")
# ])

model.summary() # Prints a summary of the network: layers, output shapes, and number of parameters

model.compile(
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),    # Adam optimizer adjusts weights efficiently; learning_rate controls step size
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),       # buvo 0.001
    loss="binary_crossentropy", # Loss function for binary classification: measures how far predictions are from true labels
    metrics=["accuracy",    # Fraction of correctly classified events (threshold 0.5)
             tf.keras.metrics.AUC(name="auc")] # ROC-AUC: evaluates signal vs background separation across all thresholds 
)

##############################################################
# Handling class imbalance (optional for balanced datasets) #
##############################################################
#
# If the number of signal events and background events is very different,
# the model might become biased toward the majority class. 
#
# To address this, you can compute class weights which give more 
# importance to the minority class during training:
#
# from sklearn.utils import class_weight
# class_weights = class_weight.compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(y_train),
#     y=y_train
# )
# class_weights_dict = dict(enumerate(class_weights))
#
# Then, pass these weights to model.fit:
#
# model.fit(
#     X_train, 
#     y_train, 
#     epochs=30, 
#     batch_size=128, 
#     validation_split=0.2, 
#     class_weight=class_weights_dict
# )

history = model.fit(
    X_train,                # Training features
    y_train,
    sample_weight=w_train,  # Training labels (signal=1, background=0)
    epochs=30,              # Number of times the model sees the full training data
    batch_size=128,         # Number of events processed before updating weights
    validation_split=0.2,   # 20% of training data used to monitor performance (not trained on)
    verbose=2               # Controls logging: 2 = progress per epoch
)

model.save("results/electron_classifier_mass.h5")

plot_training_history(history, save_path="results/training_plot.png")

plot_auc(history, save_path="results/auc_plot.png")