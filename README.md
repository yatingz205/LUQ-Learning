# A Flexible Framework for Incorporating Patient Preferences Into Q-Learning

## Overview
This repository contains the **simulation code** and **analysis scripts** associated with the manuscript:

**Title:** “A Flexible Framework for Incorporating Patient Preferences Into Q-Learning”  
**Authors:** Joshua P. Zitovsky<sup>1</sup>, Yating Zou<sup>1</sup>, Leslie Wilson<sup>2</sup>, Michael R. Kosorok<sup>1</sup>  
**Affiliation:** 1. University of North Carolina at Chapel Hill, 2. University of California, San Francisco  
**Status:** Under Review  

The repository includes code to **replicate simulation results** for the **BEST** and **CATIE** trials.

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
---

## Instructions for BEST Trial Simulation

1. **Run the simulation and estimation scripts**:
   - Run `fitModel_n.py [seed] [n] [label]` or `fitModel_K.py [seed] [K] [n]` to generate data and estimate parameters for the preference model for specified seed, n, and K.
   - label = op runs the scinario with competing outcomes; label = mis runs the model mis-specification case.
   - **Outputs**:
     - Simulated parameters and data → Stored in `./simData/`
     - Estimated parameters, estimation error, and computation time → Stored in `./estData/`

2. **Evaluate Q-learning algorithms**:
   - Run `evalOurs_n.py`, `evalNaive_n.py`, `evalSat_n.py`, `evalKnown_n.py` with the appropriate seed, n, and label parameters.
   - Or run `evalOurs_K.py`, `evalNaive_K.py`, `evalSat_K.py`, `evalKnown_K.py` with the appropriate seed, K, and n parameters.
   - **Outputs**:
     - Observed and estimated value associated with the DTR obtained from each type of Q-learning algorithm → Stored in `./evaData/`.


## Instructions for CATIE Trial Simulation

1. **Run the simulation, estimation, and evaluation scripts**:
   - Run `fitButler.R [seed] [n]` for Butler's approach.
   - Run `fitButler_Z.R [seed] [n]` for LUQ-Learning and other baselines.
   - Data generation, parameter and DTR estimation, and policy evaluation are in one place.
   - **Outputs**:
     - Simulated parameters and data → Stored in `./simData/`
     - Estimated parameters, estimation error, and computation time → Stored in `./estData/`
     - Observed and estimated value associated with the DTR obtained from each type of Q-learning algorithm  → Stored in `./evaData/`

---

## Generating Tables and Figures
After desired scenarios are run, use the makeTable or makePlots R script to produce tables and figures presented in the manuscript.

---

## License
This project is licensed under the [MIT License](https://mit-license.org/).

---

## Contact
For questions, contact [yating at live.unc.edu](mailto:yating@live.unc.edu)




