import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from src.plot_training import plot_training_history, plot_auc
import joblib
from src.analysis_utils import (
    load_dataset,
    prepare_training
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
    prepare_training(dy_low_tau, label = 0),
    prepare_training(dy_high_tau, label = 0),
    prepare_training(tw, label = 0),
    prepare_training(aw, label = 0),
    prepare_training(st, label = 0),
    prepare_training(sa, label = 0),
    prepare_training(ttbar, label = 0),
    prepare_training(zz, label = 0),
    prepare_training(wz, label = 0),
    prepare_training(ww, label = 0),
    prepare_training(qcd1, label = 0),
    prepare_training(qcd2, label = 0),
    prepare_training(qcd3, label = 0),
    prepare_training(qcd4, label = 0),
    prepare_training(qcd5, label = 0),
    prepare_training(qcd6, label = 0),
    prepare_training(qcd7, label = 0),
    prepare_training(qcd8, label = 0)
], ignore_index=True)

df_signal = pd.concat([
    prepare_training(mcDYhigh, label = 1),
    prepare_training(mcDYlow, label = 1)
], ignore_index=True)

df_train = pd.concat([df_signal, df_bkg], ignore_index=True)
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna()

# # Separate features and labels
X = df_train.drop(columns = ["label"]).to_numpy()
y = df_train["label"].to_numpy()
# Scale features to mean 0 and std 1 for stable and efficient training (does not get biased towards larger numerical values)

scaler = StandardScaler()

# Fit on training data and transform
X = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, "results/scaler_2.pkl")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle = True
)

# Build feed-forward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)), # Number of features per event
    tf.keras.layers.Dense(64),                  # First hidden layer with 64 neurons (fully connected)
    tf.keras.layers.LeakyReLU(),                # Activation function: introduces non-linearity; allows small gradient for negative inputs to keep learning

    tf.keras.layers.Dense(32),                  # Second hidden layer with 32 neurons
    tf.keras.layers.LeakyReLU(),                # Same as above, helps network learn complex patterns

    tf.keras.layers.Dense(16),                  # Third hidden layer with 16 neurons
    tf.keras.layers.LeakyReLU(),                # Same purpose: non-linearity + stable learning for negative inputs
    tf.keras.layers.Dense(1, activation="sigmoid")  # Output layer: 1 neuron, sigmoid activation
])                                                  # Outputs a probability between 0 and 1
                                                    # 1 → signal, 0 → background

model.summary() # Prints a summary of the network: layers, output shapes, and number of parameters

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),    # Adam optimizer adjusts weights efficiently; learning_rate controls step size
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
    y_train,                # Training labels (signal=1, background=0)
    epochs=30,              # Number of times the model sees the full training data
    batch_size=128,         # Number of events processed before updating weights
    validation_split=0.2,   # 20% of training data used to monitor performance (not trained on)
    verbose=2               # Controls logging: 2 = progress per epoch
)

model.save("results/electron_classifier_2.h5")

plot_training_history(history, save_path="results/training_plot.png")

plot_auc(history, save_path="results/auc_plot.png")