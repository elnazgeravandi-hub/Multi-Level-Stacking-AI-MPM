# Multi-Level Stacking AI-MPM

This repository contains the Python implementation associated with the manuscript:

**A Multi-Level Stacking Ensemble Architecture: Advantages of Random Forests, Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM) for Porphyry-related AI-based Mineral Prospectivity Mapping**

The study applies machine learning and multi-level stacking ensemble modeling for AI-based mineral prospectivity mapping of porphyry Cu±Au systems in the Ahar–Arasbaran metallogenic belt, Iran.

## Repository purpose

This repository is provided to support reproducibility of the modeling workflow, including:

* preprocessing of input predictor layers;
* spatial block-based train/validation partitioning;
* training of base-level machine learning models;
* generation of out-of-fold predictions using spatial GroupKFold validation;
* implementation of the multi-level stacking ensemble architecture;
* performance evaluation using AUC and confusion-matrix-based metrics;
* generation of GIS-ready prospectivity outputs.

## Model architecture

The implemented workflow follows a multi-level stacking architecture:

1. **Level-0 base learners**

   * Random Forest
   * XGBoost
   * LightGBM
   * Support Vector Regression
   * Multilayer Perceptron

2. **Out-of-fold prediction generation**

   * Spatial block identifiers are used as grouping variables.
   * A 5-fold GroupKFold procedure is used to generate spatially separated out-of-fold predictions.

3. **Level-1 meta-learner**

   * Ridge Regression

4. **Level-2 meta-learner**

   * LightGBM Regressor

5. **Final output**

   * Continuous prospectivity scores in the range [0, 1]
   * Binary classification using a threshold of 0.5
   * Five-class prospectivity maps for GIS interpretation

## Spatial validation strategy

To reduce spatial leakage and account for spatial autocorrelation, the study area is partitioned into spatial blocks.

* Block size: 5 km × 5 km
* Training blocks: 70%
* Validation blocks: 30%
* Random seed: 42
* All samples within the same spatial block are assigned exclusively to either the training or validation subset.

## Repository structure

```text
Multi-Level-Stacking-AI-MPM/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── train.py
│   ├── stacking_pipeline.py
│   │
│   ├── level0_models/
│   │   ├── rf_model.py
│   │   ├── xgb_model.py
│   │   ├── lgbm_model.py
│   │   ├── svr_model.py
│   │   └── mlp_model.py
│   │
│   ├── level1_models/
│   │   └── ridge_model.py
│   │
│   └── level2_models/
│       └── lgbm_level2.py
│
├── data/
│   └── README.md
│
└── results/
    └── README.md
```

## Installation

Create a clean Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Data availability and anonymization

The original geographic coordinates of mineral occurrences and background/non-occurrence samples are sensitive and are therefore not released in their raw form.

To support reproducibility while protecting sensitive spatial information, the shared datasets should use anonymized coordinates transformed into a relative local coordinate system. The anonymized datasets include:

* predictor variables;
* class labels;
* anonymized spatial coordinates;
* variables required to reproduce the spatial block partitioning and modeling workflow.

The real geographic coordinates are not required for reproducing the machine learning workflow.

## Expected input format

The input dataset should include:

* `pointid`: sample identifier
* `x`: anonymized or relative x-coordinate
* `y`: anonymized or relative y-coordinate
* `Label`: binary class label
* predictor variables used as evidential layers

For G2, the selected predictor variables are:

```text
F1_clr
F4_clr
Fe_oxide
Phylic
argillic
Microdiorite_Micromonzonite
Granodiorite
NS
Fault_Densi
NWSE
```

## Running the workflow

Example command for G1:

```bash
python src/train.py --group G1 --train data/G1/Train_data.xlsx --test data/G1/testnew.xlsx --out results/G1
```

Example command for G2:

```bash
python src/train.py --group G2 --train data/G2/Train_data.xlsx --test data/G2/testnew.xlsx --out results/G2
```

## Output files

The workflow generates:

* validation performance metrics;
* ROC curves;
* continuous prospectivity scores;
* binary prospectivity classes;
* five-class prospectivity maps;
* GIS-ready CSV outputs.

## Reproducibility note

All models are implemented using fixed random seeds where applicable. Spatial GroupKFold validation is used within the stacking framework to reduce spatial leakage during out-of-fold prediction generation.

## Citation

If this repository is used, please cite the associated manuscript once published.

## Contact

For questions regarding the manuscript or reproducibility materials, please contact the corresponding author.
