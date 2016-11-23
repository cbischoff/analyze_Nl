"""
analyze_Nl.py
=============

Derives N_l curves for published CMB B-mode results, including:
  - BICEP2 / Keck 2014 results
  - ACTPol 1st year (Naess et al) and 2nd year (Louis et al)
  - SPTPol (Keisler et al / Crites et al)
  - POLARBEAR 2014 results
  - QUIET Q-band and W-band results

The method used to extract the N_l curves makes use of error bars
for both BB and EE spectra to solve for both the noise level and
degrees-of-freedom. It assumes that noise and filtering are 
symmetric for E and B modes, which could break down for the case 
of pure-B estimators. For more details, see the posting here: 
(link TBD)

Data releases from CMB experiments are not included here. You can
find them at the following links:
  - BICEP2 / Keck: http://www.bicepkeck.org/
  - ACTPol: https://lambda.gsfc.nasa.gov/product/act/actpol_prod_table.cfm
  - SPTPol: https://lambda.gsfc.nasa.gov/product/spt/sptpol_prod_table.cfm
  - POLARBEAR: https://lambda.gsfc.nasa.gov/product/polarbear/polarbear_prod_table.cfm
  - QUIET: https://lambda.gsfc.nasa.gov/product/quiet/index.cfm
See function documentation below for details on exactly which files 
to download.

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize


def read_BK14(prefix='BK14_cosmomc'):
    """
    Read results from BICEP2 / Keck Array 2014 data release.

    """
