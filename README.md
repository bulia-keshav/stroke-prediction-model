Stroke Prediction Model
=======================

Overview
--------
This repository contains the code and notebooks used to develop and evaluate a stroke lesion segmentation and outcome-prediction system. The core idea is simple: use an automated segmentation model (U‑Net) to estimate stroke lesion volumes from MRI/CT slices, combine those volumes with routine clinical features, and train an XGBoost classifier to predict functional outcome (mRS) at 90 days.

This workspace collects preprocessing scripts, the U‑Net training pipeline, prediction tools to convert segmentation masks into volumes, and an existing clinical classification pipeline that uses XGBoost.

Highlights
----------
- A regularized U‑Net was trained to segment stroke lesions. The heavily-regularized version improves generalization (validation F1 and IoU) and produces more reliable lesion volumes.
- Integrating the improved U‑Net's volumes into the existing XGBoost pipeline increased classification accuracy on the test split from ~80.8% (baseline) to ~86.5% (improved). The improved pipeline also notably increased recall for the poor-outcome class (38% → 77%).
- Volume correlation (predicted vs truth) is strong (≈0.73), and U‑Net segmentation F1/IoU and volume statistics are included for reference.

What’s in this repo
-------------------
(High level — folder / file descriptions)

- `Scripts/UNET/` — Training notebooks and code for the U‑Net models.
  - `training.ipynb` — Full training pipeline, data generators, model definitions, metrics, and training runs.
  - Models created during training are (when present) saved into `Scripts/Prediction/` for downstream use.

- `Scripts/Prediction/` — Prediction utilities and scripts to run the saved U‑Net on full patient volumes and convert masks to volumes.
  - `Prediction.ipynb` — Inference notebook: loads a saved U‑Net, applies it slice-by-slice, and converts segmentation masks to volumes.
  - `Voxel2Volume.py` — Utility to compute physical volume from NIfTI header voxel dimensions and a binary mask.
  - `UNET_improved_regularized.h5`, `UNET_improved_regularized.keras` — The improved U‑Net models exported for inference (when present).

- `Scripts/MRS Classification/` — The clinical feature engineering and classification pipeline (XGBoost) used to predict mRS outcomes.
  - `Clot Burden/` — Folder includes cleaned clinical sheets, notebooks that run feature selection, cross-validation and XGBoost experiments which originally produced ~81% accuracy.
  - Notebooks here combine clinical variables (NIHSS, collaterals, lab values, etc.) with stroke volume features.

- `images/` — Example image folders used for visualization and training (MRI slices, masks, overlays). 

Quick workflow (how things fit together)
---------------------------------------
1. Preprocessing: normalize and resample source MR/CT volumes (see `Scripts/Preprocessing/`).
2. Train U‑Net: use `Scripts/UNET/training.ipynb` to train and export a U‑Net model. The training notebook contains metrics, regularization experiments, and final saved weights.
3. Predict volumes: load the exported U‑Net in `Scripts/Prediction/Prediction.ipynb`, run `predictVolume(...)` to get 3D mask volumes, and convert to real-world volumes using `Voxel2Volume.py`.
4. Clinical model: combine the predicted stroke volumes with clinical features and run the XGBoost pipeline in `Scripts/MRS Classification/Clot Burden/` to predict mRS (good vs poor outcome). 

Reproducibility & usage notes
-----------------------------
- The notebooks contain the full code for preprocessing, training, inference and the XGBoost experiments. Run cells in order; the training notebook expects the dataset laid out under `Slices/Train` and `Slices/Test`.
- The inference notebook (`Scripts/Prediction/Prediction.ipynb`) expects NIfTI/.nrrd patient data under `Scripts/Prediction/Data/` for single-case runs.
- Large model files and environment-specific artifacts are intentionally not included in version control by default. See `.gitignore`.

Key metrics (example run)
-------------------------
- Baseline XGBoost accuracy (original volumes): ~80.8% on test split
- Improved pipeline with new U‑Net volumes: ~86.5% accuracy (≈ +5.7 percentage points)
- U‑Net validation F1 (improved, regularized model): ≈0.067 (for slice-level segmentation metrics in this run) and IoU ≈0.11. While slice-level F1 is modest due to class imbalance, volume-level correlation to ground truth was strong (~0.73) and produced practical downstream benefits.

Notes on class imbalance and best practices
------------------------------------------
- Lesions are very sparse in many MRI slices (<<1% positive pixels). For this reason the notebooks use combined loss (BCE + Dice), heavy regularization, augmentation, and careful validation splits to avoid patient-level leakage.
- When retraining, ensure train/val/test splits are by patient to avoid optimistic results.

How to run the comparison experiment quickly
------------------------------------------
1. In `Scripts/UNET/training.ipynb` ensure the final model is saved to `Scripts/Prediction/UNET_improved_regularized.h5`.
2. Open `Scripts/Prediction/Compare_Models.ipynb` and run the cells. That notebook re-runs the clinical XGBoost pipeline with the new volumes and reports the direct change from the baseline.

License & data
---------------
- No new external data is included. Use with your institutional data following local regulations.
- Add a license file if you want a specific license (MIT, Apache-2.0, etc.).

Contact / Notes
----------------
For questions about reproducing training runs, evaluating segmentations or integrating the model into a larger pipeline (for example a web app or batch inference), check the notebooks or reach out to the repository owner.

