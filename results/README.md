# Results folder

This folder is reserved for output files generated when running the G1 and G2 workflows.

Generated result files are not included in this public repository because GIS-ready outputs may contain spatially sensitive information.

Users can run the provided Python scripts with the anonymized demonstration datasets to verify the workflow and generate example outputs locally.

## Expected output structure

After running the workflow, the following folders and files may be generated locally:

```text
results/
├── G1/
│   ├── MultiStack_G1_validation_predictions.csv
│   ├── MultiStack_G1_spatial_metrics.csv
│   ├── MultiStack_G1_classification_report.txt
│   ├── ROC_MultiStack_G1_spatial.png
│   ├── MultiStack_G1_Test_FullOutput_spatial.csv
│   └── MultiStack_G1_For_ArcMap_spatial.csv
│
└── G2/
    ├── MultiStack_G2_validation_predictions.csv
    ├── MultiStack_G2_spatial_metrics.csv
    ├── MultiStack_G2_classification_report.txt
    ├── ROC_MultiStack_G2_spatial.png
    ├── MultiStack_G2_Test_FullOutput_spatial.csv
    └── MultiStack_G2_For_ArcMap_spatial.csv
```

## Example run check

The repository was tested using the anonymized demonstration datasets provided in the `data/` folder. Both G1 and G2 workflows executed successfully and generated local example outputs.

The following validation metrics were obtained from the demonstration datasets using the default settings:

| Workflow             |    AUC | Accuracy | Precision | Recall | F1-score |
| -------------------- | -----: | -------: | --------: | -----: | -------: |
| G1 demonstration run | 0.9853 |   0.9321 |    0.8846 | 0.9684 |   0.9246 |
| G2 demonstration run | 0.9764 |   0.9140 |    0.9222 | 0.8737 |   0.8973 |

## Important note

The outputs generated from the public anonymized demonstration datasets are example workflow outputs.

They are provided only to verify that the G1 and G2 pipelines execute correctly and that the multi-level stacking workflow can generate local outputs.

These metrics and outputs should not be interpreted as the final manuscript results or as exact reproduction of the confidential study-area outputs.
