# A Flexible Framework for Incorporating Patient Preferences Into Q-Learning

## Overview
This repository contains the **simulation code** and **analysis scripts** associated with the manuscript:

**Title:** "A Flexible Framework for Incorporating Patient Preferences Into Q-Learning"
**Authors:** Joshua P. Zitovsky<sup>1</sup>, Yating Zou<sup>1</sup>, Leslie Wilson<sup>2</sup>, Michael R. Kosorok<sup>1</sup>
**Affiliation:** 1. University of North Carolina at Chapel Hill, 2. University of California, San Francisco
**Status:** Under Review

The repository includes code to **replicate simulation results** for the **BEST** and **CATIE** trials. All scripts are located in the `script2/` directory and should be run from within that directory.

---

## Table of Contents
1. [Requirements](#requirements)
2. [Instructions for BEST Trial Simulation](#instructions-for-best-trial-simulation)
3. [Instructions for CATIE Trial Simulation](#instructions-for-catie-trial-simulation)
4. [Generating Tables & Figures](#generating-tables-and-figures)
5. [License](#license)
6. [Contact](#contact)

---

## Requirements
- **Python Version**: 3.9.19
- **R Version**: 4.4.0
- **Conda Environment**: Specified in [environment.yml](environment.yml)

Install the required conda environment by running:

```bash
conda env create -f environment.yml
conda activate luql_env
```
Set up required packages for R by:

```bash
module add r/4.4.2
```

---

## Instructions for BEST Trial Simulation

All scripts below should be run from within the `script2/` directory.

1. **Run the simulation and estimation scripts**:
   - Run `python fitModel_n.py [seed] [n_sim] [n] [label]` to generate data and estimate preference model parameters for the varying-n scenarios.
   - Run `python fitModel_K.py [seed] [n_sim] [n] [K]` to generate data and estimate preference model parameters for the varying-K scenarios.
   - `label = op` runs the scenario with competing outcomes; `label = mis` runs the model misspecification case.
   - **Outputs**:
     - Simulated parameters and data → Stored in `./simData/`
     - Estimated parameters, estimation error, and computation time → Stored in `./estData/`

2. **Evaluate Q-learning algorithms**:
   - For varying-n scenarios, run with arguments `[seed] [n] [label]`:
     - `Rscript evalOurs_n.R [seed] [n] [label]` — LUQ-Learning
     - `Rscript evalNaive_n.R [seed] [n] [label]` — Naive approach
     - `Rscript evalSat_n.R [seed] [n] [label]` — Satisfaction-based approach
     - `Rscript evalKnown_n.R [seed] [n] [label]` — Oracle (preference known)
   - For varying-K scenarios, run with arguments `[seed] [K] [n]`:
     - `Rscript evalOurs_K.R [seed] [K] [n]`, `Rscript evalNaive_K.R [seed] [K] [n]`, `Rscript evalSat_K.R [seed] [K] [n]`, `Rscript evalKnown_K.R [seed] [K] [n]`
   - **Outputs**:
     - Observed and estimated value associated with the DTR obtained from each algorithm → Stored in `./evaData/`


## Instructions for CATIE Trial Simulation

All scripts below should be run from within the `script2/` directory.

1. **Run the simulation, estimation, and evaluation scripts**:
   - Run `fitButler.R [seed] [n]` for Butler's approach.
   - Run `fitButler_Z.R [seed] [n]` for LUQ-Learning and other baselines.
   - Data generation, parameter and DTR estimation, and policy evaluation are all handled in a single script.
   - **Outputs**:
     - Simulated parameters and data → Stored in `./simData/`
     - Estimated parameters, estimation error, and computation time → Stored in `./estData/`
     - Observed and estimated value associated with the DTR obtained from each algorithm → Stored in `./evaData/`

---

## Generating Tables and Figures
After the desired scenarios are run, use the corresponding `makeTable_*.R` or `makePlots_*.R` script to produce the tables and figures presented in the manuscript.

---

## License
This project is licensed under the [MIT License](https://mit-license.org/).

---

## Contact
For questions, contact [yating at live.unc.edu](mailto:yating@live.unc.edu)
