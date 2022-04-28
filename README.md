# Metric Learning and Adaptive Boundary for Out-of-Domain Detection

---

## About
Source code for research paper [Metric Learning and Adaptive Boundary for Out-of-Domain Detection](https://arxiv.org/pdf/2204.10849.pdf) (accepted to [NLDB 2022](https://nldb2022.prhlt.upv.es)).

## Usage
Run `python3 code/run.py {clinc150,banking77}` for chosen dataset to replicate results.

## Dependencies
Run `pip install -r code/requirements.txt` to install dependencies.

## Datasets
Evaluated on [CLINC150](https://github.com/clinc/oos-eval) and [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets).

## Results
### Overall results
Accuracy and F1 score calculated for all classes (IND classes and OOD class).

|  | | 25% known ratio | | 50% known ratio | | 75% known ratio | |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset** | **Method** | **Accuracy** | **F1** | **Accuracy** | **F1** | **Accuracy** | **F1** |
| CLINC150 | MSP | 47.02 | 47.62 | 62.96 | 70.41 | 74.07 | 82.38 |
|  | DOC | 74.97 | 66.37 | 77.16 | 78.26 | 78.73 | 83.59 |
|  | OpenMax | 68.50 | 61.99 | 80.11 | 80.56 | 76.80 | 73.16 |
|  | DeepUnk | 81.43 | 71.16 | 83.35 | 82.16 | 83.71 | 86.23 |
| | ADB | 87.59 | 77.19 | 86.54 | 85.05 | 86.32 | 88.53 |
| | ODIST | 89.79 | UNK | 88.61 | UNK | 87.70 | UNK |
| | Our<sub>LMCL</sub> | **91.81** | **85.90** | 88.81 | 89.19 | **88.54** | **92.21** |
| | Our<sub>Triplet</sub> | 90.28 | 84.82 | **88.89** | **89.44** | 87.81 | 91.72 |
| BANKING77 | MSP | 43.67 | 50.09 | 59.73 | 71.18 | 75.89 | 83.60 |
|  | DOC | 56.99 | 58.03 | 64.81 | 73.12 | 76.77 | 83.34 |
|  | OpenMax | 49.94 | 54.14 | 65.31 | 74.24 | 77.45 | 84.07 |
|  | DeepUnk | 64.21 | 61.36 | 72.73 | 77.53 | 78.52 | 84.31 |
| | ADB | 78.85 | 71.62 | 78.86 | 80.90 | 81.08 | 85.96 |
| | ODIST | 81.69 | UNK | 80.90 | UNK | 82.79 | UNK |
| | Our<sub>LMCL</sub> | **85.71** | **78.86** | **83.78** | **84.93** | **84.40** | **88.39** |
| | Our<sub>Triplet</sub> | 82.71 | 70.02 | 81.83 | 83.07 | 81.82 | 86.94 |

### Class-specific results
F1 score calculated for IND classes and OOD class separately.

|  | | 25% known ratio | | 50% known ratio | | 75% known ratio | |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset** | **Method** | **F1 (OOD)** | **F1 (IND)** | **F1 (OOD)** | **F1 (IND)** | **F1 (OOD)** | **F1 (IND)** |
| CLINC150 | MSP | 50.88 | 47.53 | 57.62 | 70.58 | 59.08 | 82.59 |
|  | DOC | 81.98 | 65.96 | 79.00 | 78.25 | 72.87 | 83.69 |
|  | OpenMax | 75.76 | 61.62 | 81.89 | 80.54 | 76.35 | 73.13 |
|  | DeepUnk | 87.33 | 70.73 | 85.85 | 82.11 | 81.15 | 86.27 |
| | ADB | 91.84 | 76.80 | 88.65 | 85.00 | 83.92 | 88.58 |
| | ODIST | 93.42 | 79.69 | **90.62** | 86.52 | **85.86** | 89.33 |
| | Our<sub>LMCL</sub> | **94.5** | **85.6** | 88.9 | 89.2 | 78.4 | **92.3** |
| | Our<sub>Triplet</sub> | 93.3 | 84.6 | 89.0 | **89.4** | 76.6 | 91.8 |
| BANKING77 | MSP | 41.43 | 50.55 | 41.19 | 71.97 | 39.23 | 84.36 |
|  | DOC | 61.42 | 57.85 | 55.14 | 73.59 | 50.60 | 83.91 |
|  | OpenMax | 51.32 | 54.28 | 54.33 | 74.76 | 50.85 | 84.64 |
|  | DeepUnk | 70.44 | 60.88 | 69.53 | 77.74 | 58.54 | 84.75 |
| | ADB | 84.56 | 70.94 | 78.44 | 80.96 | 66.47 | 86.29 |
| | ODIST | 87.11 | 72.72 | 81.32 | 81.79 | 71.95 | 87.20 |
| | Our<sub>LMCL</sub> | **89.9** | **78.4** | **83.9** | **84.9** | **73.1** | **88.7** |
| | Our<sub>Triplet</sub> | 88.0 | 69.1 | 81.9 | 83.0 | 66.8 | 87.2 |

## Citation
If you like our work, please ⭐ this repository.

*Note: citation will be available soon*

## Acknowledgments
This research was partially supported by the Grant Agency of the Czech Technical University in Prague, grant
(SGS22/082/OHK3/1T/37).
