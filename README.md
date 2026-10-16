# Classifying Higgs Boson Signal Events in Simulated ATLAS Collision Data

DATA 400 Individual Mini-Project, Fall 2026
Author: Emerson Carmine

## Project overview

Collider physics has a needle-in-a-haystack problem: background processes can leave detector signatures that look a lot like a Higgs boson decaying into two tau particles. This project asks:

> Given the kinematic and derived features of a simulated ATLAS collision event, how accurately can a classification model separate a Higgs decay into two tau particles from background processes with a similar detector signature?

To answer it, I train a LightGBM gradient-boosting classifier on the ATLAS Higgs Boson Machine Learning Challenge data and evaluate it with two metrics:

- **ROC-AUC**, which measures how well the model ranks signal events above background events.
- **Approximate median significance (AMS)**, which measures how significant the selected signal looks at a chosen probability threshold. AMS uses the event weights, so it is much closer to what a physicist actually cares about.

The notebook also retrains the model on only the derived (DER) features and only the primitive (PRI) features, to see how much the physics-derived variables contribute. Everything here uses simulated events, so none of it is a claim about real Higgs discoveries.

## Data source

The data is the **ATLAS Higgs Boson Machine Learning Challenge 2014** dataset, hosted on the CERN Open Data Portal.

- Record: https://opendata.cern.ch/record/328
- File used: `atlas-higgs-challenge-2014-v2.csv` (downloaded from the record as a `.csv.gz`)
- Contents: about 818,000 simulated collision events with roughly 30 kinematic and derived features, a signal or background label (`Label`), and event weights (`Weight`, `KaggleWeight`).
- Feature families: `PRI_*` columns are primitive measured or reconstructed quantities; `DER_*` columns are derived from them using physics knowledge.
- Missing values: unavailable quantities are coded as `-999.0`, which the code converts to `NaN`.
- Training subset: the `KaggleSet` column marks the original challenge splits. The analysis uses the 250,000 rows where `KaggleSet == "t"` (training), with `KaggleWeight` as the event weight.

The CSV is **not stored in this repository** (it is git-ignored because of its size). The Output notebook automatically downloads it for you the first time you run it.

## Repository contents

```
.
├── Output_notebook.ipynb      Main analysis: EDA, LightGBM training, ROC, feature
│                              importance, AMS curves, DER vs PRI comparison
├── src/
│   ├── __init__.py
│   ├── Import_data.py         Dataloader class: downloads and unzips the dataset
│   └── higgs_lgbm.py          Data prep, weighting, AMS functions, LightGBM
│                              training, and plotting helpers
├── presentation/
│   └── Higgs_Boson_ML_Challenge_Final_Presentation.pptx Presentation given in class
|
├── writeup/
│   └── writeup/Emerson Miniproject Writeup.docx  This is the writeup
|
├── requirements.txt           Python package list
├── .gitignore
└── data/                      Created on first run (git-ignored); holds the CSV
```

## Requirements

- Python 3.10 or newer (developed on a Python 3.14 virtual environment)
- An internet connection for the first run, to download the dataset
- Python packages (pinned loosely in `requirements.txt`):

| Package | Minimum version |
|---|---|
| pandas | 2.0 |
| numpy | 1.24 |
| scikit-learn | 1.3 |
| matplotlib | 3.7 |
| seaborn | 0.12 |
| lightgbm | 4.0 |
| jupyter | 1.0 |
| requests | 2.31 |

The notebook's import cell also imports `xgboost`, which is **not** listed in `requirements.txt`, so install it too (see step 3).

## How to run the analysis

All commands are run from the repository root, because the data path (`data/`) and the `src` imports are relative to it.

**1. Clone the repository**


**2. Create and activate a virtual environment** (recommended, sometimes works for me in base python but venv is much easier) 


**3. Run Output_notebook.ipynb** (Should automatically install everything necessary)

After choosing **Run All** the notebook will:

1. Load the dataset (downloading it first if needed).
2. Run exploratory analysis: class balance, `-999` sentinel counts, a transverse-mass histogram by class, and a feature correlation heatmap.
3. Load the 250,000-event training set and print the class counts and total event weight per class.
4. Train LightGBM on all features with an 80/20 stratified train and validation split (seed 42), early stopping on validation AUC, and class-balanced weights for fitting.
5. Sweep the probability threshold to find the cut that maximizes AMS, with the validation weights rescaled to the full-sample totals.
6. Plot the event-weight distribution, ROC curves, top-15 feature importances, and AMS curves.
7. Retrain on DER-only and PRI-only features and plot their AMS curves together.

The training cells print the best boosting iteration, the AMS-maximizing threshold, the best AMS, and the percent of events selected. Exact values can shift slightly with package versions, so check the printed output instead of relying on numbers quoted elsewhere.

