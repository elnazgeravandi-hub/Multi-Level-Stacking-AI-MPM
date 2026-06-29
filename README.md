# Multi-Level Stacking AI-MPM

This repository provides the Python implementation of a multi-level stacking ensemble workflow for AI-based mineral prospectivity mapping.

The repository was prepared as a reproducibility and workflow-verification supplement for the manuscript:

**A Multi-Level Stacking Ensemble Architecture: Advantages of Random Forests, Extreme Gradient Boosting (XGBoost), and Light Gradient Boosting Machine (LightGBM) for Porphyry-related AI-based Mineral Prospectivity Mapping**

## Repository purpose

This repository allows reviewers and readers to run the Python workflow using anonymized demonstration datasets.

The purpose of the repository is to support reproducibility of the computational workflow, not to disclose the original confidential geospatial datasets.

The repository supports:

* execution of the Python modeling pipeline;
* verification of the G1 and G2 modeling workflows;
* spatial block-based train/validation partitioning;
* training of Level-0 base machine-learning models;
* generation of spatially constrained out-of-fold predictions;
* implementation of the multi-level stacking ensemble model;
* calculation of validation metrics;
* generation of example GIS-ready output tables.

The public datasets are demonstration datasets. They preserve the structure required to run the workflow, but they are not the original datasets used to produce the final manuscript results.

## Data availability and anonymization

The original geospatial datasets used in the manuscript contain sensitive information related to mineral occurrence and non-occurrence locations. Therefore, the exact original datasets are not publicly released.

To support reproducibility while protecting sensitive mineral-location information, this repository provides anonymized demonstration datasets.

The released datasets do not disclose:

* original mineral occurrence coordinates;
* original non-occurrence coordinates;
* the exact study-area coordinate network;
* original UTM coordinates;
* original CRS information;
* original UTM zone;
* original coordinate origin;
* raw GIS layers;
* confidential final prospectivity outputs.

The public `x` and `y` fields are expressed in a relative local coordinate system. They should not be interpreted as real-world coordinates or real UTM coordinates.

Spatial validation is preserved using the provided `spatial_block_id` field. This field is an anonymized spatial grouping identifier and is used only for spatial block splitting and spatially constrained validation. It must not be used as a predictor variable.

## Demonstration datasets

The demonstration datasets are designed to allow the scripts to run and produce outputs with the same structure as those reported in the manuscript.

However, outputs generated from these public datasets are demonstration outputs only. They should not be interpreted as the final confidential prospectivity results of the study.

The goal is to verify that the computational workflow is executable and reproducible under an anonymized data-release setting.

## Model architecture

The implemented workflow follows a multi-level stacking ensemble structure.

### Level-0 base learners

The Level-0 base learners include:

* Random Forest
* LightGBM Regressor
* XGBoost Regressor
* Support Vector Regression
* Multilayer Perceptron Regressor

### Spatial out-of-fold prediction

Spatial block identifiers are used as grouping variables.

A spatial GroupKFold procedure is used to generate spatially constrained out-of-fold predictions and reduce spatial leakage.

### Level-1 meta-learner

The Level-1 meta-learner is Ridge Regression.

It receives the Level-0 out-of-fold prediction matrix and produces an intermediate probability-like score.

### Level-2 meta-learner

The Level-2 meta-learner is LightGBM.

It receives the Level-1 Ridge output and produces the final prospectivity score.

Therefore, the implemented architecture is:

`Level-0 base learners → Level-1 Ridge Regression → Level-2 LightGBM meta-learner`

## Outputs

When the workflow is executed, it generates:

* validation probabilities;
* validation performance metrics;
* ROC curve;
* AUC value;
* accuracy, precision, recall, and F1-score;
* continuous prospectivity scores;
* binary prospectivity classes;
* five-class prospectivity classes;
* GIS-ready CSV output tables.

These outputs are generated locally from the anonymized demonstration datasets.

## Repository structure

```text
Multi-Level-Stacking-AI-MPM/
│
├── README.md
├── requirements.txt
├── DATA_ANONYMIZATION_NOTE.txt
│
├── src/
│   ├── train.py
│   ├── stacking_pipeline.py
│   ├── level0_models/
│   ├── level1_models/
│   └── level2_models/
│
├── data/
│   ├── README.md
│   ├── G1/
│   │   ├── Train_data.xlsx
│   │   └── testnew.xlsx
│   └── G2/
│       ├── Train_data.xlsx
│       └── testnew.xlsx
│
└── results/
    └── README.md
```

## Installation

Create a clean Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

## Running the workflow

Run the G1 workflow:

```bash
python src/train.py --group G1
```

Run the G2 workflow:

```bash
python src/train.py --group G2
```

Optional parameters:

```bash
python src/train.py --group G1 --threshold 0.5 --block_size 5000 --n_splits 5 --train_ratio 0.7 --seed 42
```

The same command structure is used for both G1 and G2. The only difference is the predefined predictor-variable configuration used by each group.

## Expected input format

Each training dataset should include:

* `pointid`: anonymized sample identifier;
* `x`: anonymized relative x-coordinate;
* `y`: anonymized relative y-coordinate;
* predictor variables;
* `Label`: binary class label;
* `spatial_block_id`: anonymized spatial grouping identifier.

Each prediction dataset should include:

* `pointid`;
* anonymized relative `x` and `y` fields;
* the predictor variables required for the selected group;
* `spatial_block_id`.

The `spatial_block_id` field is used only for spatial validation and should not be used as a predictor variable.

## Important limitations

The public datasets are anonymized demonstration datasets prepared for reviewer verification and code execution.

They are not intended to disclose the exact study area, original coordinate network, original mineral occurrence coordinates, original non-occurrence coordinates, original CRS, UTM zone, coordinate origin, raw GIS layers, or final confidential prospectivity results.

The generated public outputs are example workflow outputs. They should not be interpreted as the final results reported in the manuscript.

## Results folder

Generated outputs are written locally to the `results/` directory when the scripts are executed.

Large GIS-ready output files are not included in this public repository because they may contain spatially sensitive information.

## Citation

If this repository is used, please cite the associated manuscript once published.

## Contact

For questions regarding the manuscript or reproducibility materials, please contact the corresponding author.
