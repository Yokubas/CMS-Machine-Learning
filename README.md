# CMS Electron Classification Project

This project trains a neural network to classify CMS experiment events as **signal** or **background** using electron and jet features. The **signal** is defined as electron–positron pairs originating from the Drell–Yan process in proton–proton collisions. Any other events that contain a real electron pair—or events where detector imperfections cause other particles to be misidentified as electrons—are treated as **background noise**.

---

## Folder Structure
```
project_root/
│
├─ data/
│ ├─ raw/ # Raw ROOT files listed in txt
│ ├─ processed/ # Processed data ready for analysis
│ └─ real/ # Real CMS experiment data
│ 
├─ cpp/
│ ├─ Makefile
│ └─ read_root.cpp
│
├─ src/ # Python modules
│ ├─ __init__.py
│ ├─ preprocessing.py
│ ├─ plot_training.py
│ └─ analysis_utils.py
│
├─ scripts/ # Scripts for dataset prep, training, evaluation
│ ├─ 1_prepare_dataset.py
│ ├─ 2_train.py
│ ├─ 3_evaluate.py
│ └─ 4_plot_real_data.py
│
├─ results/ # Trained models and plots
│ ├─ electron_classifier.h5
│ ├─ training_plot.png
│ ├─ auc_plot.png
│ └─ roc_plot.png
│
├─ environment.yml # Conda environment
├─ report.pdf
└─ README.md 
```

## Requirements / Setup

**1. Clone the repository:**
``` bash
git clone https://github.com/Yokubas/CMS-Machine-Learning
```
**2. Create the Conda environment for Python analysis:**
``` bash
conda env create -f environment.yml
conda activate cern_tf
```
**3. For C++ workflow:**

**1. Install Docker**

Go to the [official Docker website](https://docs.docker.com/get-started/get-docker/) and follow the installation instructions for your operating system.

**2. Run the ROOT Docker container**

Open a terminal in the project folder (where you cloned the repository) and run:

``` bash
docker run -it --name my_root -P -p 5901:5901 -p 6080:6080 -v $PWD:/code gitlab-registry.cern.ch/cms-cloud/root-vnc:latest
```
## Usage

Run scripts from the project root using the ```-m``` flag to handle imports correctly:

**1. Prepare data set:**
``` bash
python -m scripts.1_prepare_dataset
```
- Loads signal and background ROOT files
- Flattens electrons and jets
- Saves proccesed dataset as ```data/processed/electron_dataset.csv```

**2. Train model**
``` bash
python -m scripts.2_train
```
- Trains a neural network on the preprocessed dataset
- Saves trained model as ```results/electron_classifier.h5```
- Generates plots for training history and AUC (```results/```)

**3. Evaluate model**
``` bash
python -m scripts.3_evaluate
```
- Loads saved model
- Computes predictions and ROC curve
- Saves evaluation plots (```results/roc_curve.png```)

## Notes
- Electron and jet features are flattened to a fixed number of objects per event.
- All features are standardized to mean 0 and standard deviation 1.
- Optional: class weights can be used in training to handle imbalanced datasets.

## References / Data
- Data used for training from the CMS experiment (NanoAODSIM format for 2016 collision data).
- Relevant Python libraries: ```pandas```, ```numpy```, ```awkward```, ```uproot```, ```tensorflow```, ```scikit-learn```, ```matplotlib```