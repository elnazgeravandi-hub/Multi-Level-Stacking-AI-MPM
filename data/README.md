# Data

This folder contains anonymized datasets used to reproduce the AI-based mineral prospectivity mapping workflow.

The datasets are provided for two modeling scenarios:

* `G1`: full evidential-layer dataset
* `G2`: selected/effective evidential-layer dataset

## Folder structure

```text
data/
├── G1/
│   ├── Train_data.xlsx
│   └── testnew.xlsx
│
└── G2/
    ├── Train_data.xlsx
    └── testnew.xlsx
```

## Dataset description

### G1

`G1/Train_data.xlsx` contains:

* `pointid`: anonymized sample identifier
* `x`: anonymized relative x-coordinate
* `y`: anonymized relative y-coordinate
* `Label`: binary class label
* all evidential predictor variables used in the G1 modeling scenario

`G1/testnew.xlsx` contains:

* `pointid`
* anonymized relative `x` and `y` coordinates
* all G1 predictor variables required for spatial prediction

### G2

`G2/Train_data.xlsx` contains:

* `pointid`: anonymized sample identifier
* `x`: anonymized relative x-coordinate
* `y`: anonymized relative y-coordinate
* `Label`: binary class label
* selected predictor variables used in the G2 modeling scenario

The G2 predictor variables are:

* `F1_clr`
* `F4_clr`
* `Fe_oxide`
* `Phylic`
* `argillic`
* `Microdiorite_Micromonzonite`
* `Granodiorite`
* `NS`
* `Fault_Densi`
* `NWSE`

`G2/testnew.xlsx` contains:

* `pointid`
* anonymized relative `x` and `y` coordinates
* the same G2 predictor variables required for spatial prediction

## Coordinate anonymization

The released `x` and `y` fields do not represent real-world coordinates.

To protect sensitive mineral occurrence locations, the original coordinate origin, CRS, UTM zone, and raw mineral occurrence coordinates are not released. The coordinate network was transformed into a relative local coordinate system before public release.

This anonymization preserves the internal spatial configuration required for spatial block validation and model reproducibility while preventing disclosure of exact mineral occurrence locations.

## Important note

Raw GIS layers, original occurrence coordinates, real-world coordinate origins, CRS information, and UTM zone information should not be uploaded to this public repository.
