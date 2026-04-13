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
    prepare_training(wjets, label = 0)
], ignore_index=True)

df_signal = pd.concat([
    prepare_training(mcDYhigh, label = 1),
    prepare_training(mcDYlow, label = 1)
], ignore_index=True)

electrons_signal_high, _ = build_electrons(mcDYhigh)
electrons_signal_low, _ = build_electrons(mcDYlow)

masses_signal = np.concatenate([
    z_mass_numpy(electrons_signal_high),
    z_mass_numpy(electrons_signal_low)
])

df_train = pd.concat([df_signal, df_bkg], ignore_index=True)
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna()

masses = np.zeros(len(df_train))
signal_mask = df_train["label"] == 1
masses[signal_mask] = masses_signal
assert len(masses_signal) == signal_mask.sum()
masses[~signal_mask] = 0.0

mass_mean = np.mean(masses)
mass_std = np.std(masses)

masses = (masses - mass_mean) / mass_std

# # Separate features and labels
X = df_train.drop(columns = ["label"]).to_numpy()
y = df_train["label"].to_numpy()
# Scale features to mean 0 and std 1 for stable and efficient training (does not get biased towards larger numerical values)

scaler = StandardScaler()

# Fit on training data and transform
X = scaler.fit_transform(X)

# Save the scaler for later use
joblib.dump(scaler, "results/scaler_adversarial.pkl")

# Split into training and test sets
X_train, X_test, y_train, y_test, mass_train, mass_test = train_test_split(
    X, y, masses, test_size=0.2, random_state=42, shuffle=True
)

# Build feed-forward neural network
classifier = tf.keras.Sequential([
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
adversary = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),   # takes classifier output
    tf.keras.layers.Dense(32),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(16),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1)             # predicts mass (regression)
])

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train, mass_train))
dataset = dataset.shuffle(10000).batch(128)

bce = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(0.001)
adv_optimizer = tf.keras.optimizers.Adam(0.001)

@tf.function
def train_step(x, y, mass):
    
    with tf.GradientTape() as tape:
        
        y_pred = classifier(x, training=True)
        
        mass_pred = adversary(x, training=True)

        class_loss = bce(y, y_pred)
        adv_loss = mse(mass, mass_pred)

        total_loss = class_loss - 1 * adv_loss

    # gradients
    grads = tape.gradient(total_loss, classifier.trainable_variables)
    optimizer.apply_gradients(zip(grads, classifier.trainable_variables))

    # adversary update (separate tape)
    with tf.GradientTape() as tape2:
        y_pred = classifier(x, training=True)
        mass_pred = adversary(x, training=True)
        adv_loss = mse(mass, mass_pred)

    adv_grads = tape2.gradient(adv_loss, adversary.trainable_variables)
    adv_optimizer.apply_gradients(zip(adv_grads, adversary.trainable_variables))

    return class_loss, adv_loss

for epoch in range(30):
    for x_batch, y_batch, mass_batch in dataset:
        cl, al = train_step(x_batch, y_batch, mass_batch)

    print(f"Epoch {epoch} | class loss {cl.numpy()} | adv loss {al.numpy()}")

classifier.save("results/electron_classifier_adversarial.h5")