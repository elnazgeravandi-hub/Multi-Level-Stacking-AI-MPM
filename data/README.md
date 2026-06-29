# Data folder

This folder contains anonymized tabular modeling datasets prepared to support reviewer verification, code execution, and reproducibility of the AI-based mineral prospectivity mapping workflow.

The original geospatial datasets contain sensitive mineral occurrence and non-occurrence locations and are therefore not publicly released in raw geospatial form.

The public datasets use anonymized relative local coordinates. The `x` and `y` fields should not be interpreted as real UTM coordinates. The original CRS, UTM zone, coordinate origin, raw mineral occurrence coordinates, raw non-occurrence coordinates, and original GIS layers are not included.

Spatial validation is preserved using the `spatial_block_id` field. This field is provided directly in the datasets and is used by the Python scripts for deterministic spatial block splitting and GroupKFold validation.

The `spatial_block_id` field must not be used as a predictor variable.

## Folder structure

```text
data/
├── README.md
├── G1/
│   ├── Train_data.xlsx
│   └── testnew.xlsx
└── G2/
    ├── Train_data.xlsx
    └── testnew.xlsx
```
