# Immune Response Model
This repository contains the codes and patient data used for generating simulation results for the coupled stochastic immune response model. More information about the numerical scheme and supporting information can be found in the main text.
## Requirements & Installation
The Immune Response Model requires Python (Python 3 recommended), Stochastic package, Matplotlib, Numpy and Pandas. Moreover, to make the plots interactive in the jupyter versions of the code the package ipympl can be used. The required packages are provided in the requirements text file and can be installed via running:

```
pip3 install -r requirements.txt  # Install all packages in requirements.txt
```
## Usage Instructions
The ImmuneResponseModel() class has methods to simulate instances of:
* Brownian Motion (Bm) in both T cell and Virus Populations. 
* Fractional Brownian Motion (fBm) in both T cell and Virus Populations.
* Bm in Virus and fBm in T cells.
* fBm in Virus and Bm in T cells.
To simulate each instances run the appropriate method. 
## Reproducing Plots from Manuscript
The plots of particular dataset fit can be obtained by using the .csv files that store the data for each patient and running the corresponding code for dataset fit.
## Reference
Manuscript title: Persistent correlation in cellular noise determines longevity of viral infection \
Authors: Abhilasha Batra, Shoubhik Banerjee and Rati Sharma
