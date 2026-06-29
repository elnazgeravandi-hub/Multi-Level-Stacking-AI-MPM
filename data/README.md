# Data folder

This folder contains anonymized demonstration datasets prepared to support reviewer verification and execution of the AI-based mineral prospectivity mapping workflow.

The datasets included in this repository are not the original geospatial datasets used to generate the final manuscript results.

They are provided only to demonstrate the data structure, run the Python scripts, and verify the G1 and G2 modeling workflows.

## Data anonymization

The original geospatial datasets contain sensitive mineral occurrence and non-occurrence locations and are not publicly released.

The public datasets use anonymized relative coordinates.

The `x` and `y` fields should not be interpreted as real UTM coordinates or real-world locations.

The original CRS, UTM zone, coordinate origin, raw mineral occurrence coordinates, raw non-occurrence coordinates, original spatial layers, and final confidential GIS outputs are not included.

Spatial validation is preserved using the `spatial_block_id` field.

This field is provided directly in the datasets and is used by the Python scripts for spatial block splitting and spatially constrained validation.

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

## Intended use

These datasets are intended to allow reviewers and readers to:

* run the G1 workflow;
* run the G2 workflow;
* verify that the Python scripts execute correctly;
* inspect the input data structure;
* generate local validation metrics and GIS-ready output tables.

The outputs generated from these anonymized datasets are demonstration outputs only.

They should not be interpreted as the final confidential prospectivity results reported in the manuscript.
