# Fracture Risk Prediction in Patients with Osteoporosis using traditional and Machine Learning Methods: A nationwide, prospective cohort study in Switzerland with Validation in the UK Biobank

## Introduction

The purpose of this README is to quickly give the reader an overview of the repository structure and the content of the files. The repository contains the code and data used for the analysis of the paper "Fracture Risk Prediction in Patients with Osteoporosis using traditional and Machine Learning Methods: A nationwide, prospective cohort study in Switzerland with Validation in the UK Biobank".

## Repository structure

The repository is structured as follows:

- `config/`: Contains configuration files with machine learning model parameters.
- `data/`: Contains the data used in the development and valiation process.
- `models/`: Contains the trained models (only available on request).
- `results/`: Contains the results of the analysis.
- `ukbiobank/notebooks/`: Contains the notebooks used for the validation on the UK Biobank. These notebooks cannot be run independently, as they require the data from the UK Biobank.
- `calculate_scores.ipynb`: Training + testing of the models + calculation of the performance metrics + calculation of p values.
- `heatmaps.ipynb`: The heatmaps in figure 1 are generated in this notebook.
- `plots.ipynb`: Notebook for creating various other plots presented in the paper.
- `preprocessing.ipynb`: In this notebook, all of the separate data sources are merged into one dataset. Data imputation and feature engineering is also done in this notebook.
- `table1.ipynb`: Notebook for generating Table 1.
- `utils.py`: Utility functions used across the project.
- `validation.ipynb`: Notebook for model validation on the UK Biobank.
- `xgb_model.py`: This file includes the train and test/evaluate functions for the XGBoost models.

## Reproducibility

- To reproduce the results presented in the paper, please run the notebook `create_scores.ipynb`.
- To reproduce the figures or tables presented in the paper, please run the corresponding notebook.

## Contact

If you have any questions, please contact the corresponding author of the paper: [Oliver Lehmann](mailto:lehmannoliver96@gmail.com)
