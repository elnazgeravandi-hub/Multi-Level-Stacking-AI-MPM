# Results folder

This folder is reserved for output files generated when running the G1 and G2 workflows.

Generated result files are not included in this public repository because GIS-ready outputs may contain spatially sensitive information.

Users can run the provided Python scripts with the anonymized demonstration datasets to verify the workflow and generate example outputs locally.

## Expected output structure

After running the workflow, the following folders may be generated locally:

```text
results/
├── G1/
│   ├── metrics.csv
│   ├── roc_curve.png
│   ├── validation_predictions.csv
│   ├── full_prediction_output.csv
│   └── gis_ready_prediction_output.csv
└── G2/
    ├── metrics.csv
    ├── roc_curve.png
    ├── validation_predictions.csv
    ├── full_prediction_output.csv
    └── gis_ready_prediction_output.csv
```

## Important note

The outputs generated from the public anonymized demonstration datasets are example workflow outputs.

They should not be interpreted as the final confidential prospectivity results reported in the manuscript.
