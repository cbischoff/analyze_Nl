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
  - BICEP2/Keck: http://www.bicepkeck.org/
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
from os.path import join
from scipy.optimize import minimize


def BK14_data(band='BK14_150', prefix='BK14_cosmomc'):
    """
    Results from BICEP2 / Keck BK14 data release.
    
    This function reads data from the BK14 cosmomc data release, which can
    be downloaded from: 
    http://www.bicepkeck.org/BK14_datarelease/BK14_cosmomc.tgz

    Parameters
    ----------
    band : str, optional
        Specify which frequency band to use. Should be either "BK14_95" or
        "BK14_150". Defaults to "BK14_150".
    prefix : str, optional
        Path to directory containing BK14 cosmomc data release.

    Returns
    -------
    data : tuple, two elements
        Bandpower statistics for EE (first element) and BB (second element).

    """

    # Select 95 or 150 GHz.
    if band == 'BK14_95':
        (iEE, iBB) = (0, 1)
        fwhm = 43.0 / 60.0 # beamsize, in degrees
    elif band == 'BK14_150':
        (iEE, iBB) = (2, 3)
        fwhm = 30.0 / 60.0 # beamsize, in degrees
    else:
        print "ERROR: BK14 band selection is invalid"
        return None
        
    # Data structures for results.
    EE = {'nsplit': 1}
    BB = {'nsplit': 1}

    # Ell bins: nine bins, delta-ell = 35, starting from ell = 20.
    EE['bins'] = np.arange(20, 336, 35.0)
    BB['bins'] = EE['bins']

    # Loop over bandpower window functions.
    nbin = 9
    # Calculate mean ell of each bandpower. This will be used to convert
    # from Dl to Cl.
    EE['ell'] = np.zeros(nbin)
    BB['ell'] = np.zeros(nbin)
    # Also, calculate Bl^2 averaged across each bandpower.
    EE['Bl2'] = np.zeros(nbin)
    BB['Bl2'] = np.zeros(nbin)
    for i in range(nbin):
        # Read bandpower window function.
        filename = 'BK14_bpwf_bin{}.txt'.format(i + 1)
        bpwf = np.genfromtxt(join(prefix, 'data', 'BK14',
                                  'windows', filename))
        # Mean ell for each bandpower.
        EE['ell'][i] = np.average(bpwf[:,0], weights=bpwf[:,iEE])
        BB['ell'][i] = np.average(bpwf[:,0], weights=bpwf[:,iBB])
        # Mean value of Bl^2.
        sigma_rad = np.radians(fwhm / np.sqrt(8.0 * np.log(2)))
        Bl2 = np.exp(-1.0 * bpwf[:,0]**2 * sigma_rad**2)
        EE['Bl2'][i] = np.average(Bl2, weights=bpwf[:,iEE])
        BB['Bl2'][i] = np.average(Bl2, weights=bpwf[:,iBB])
        
    # Load bandpower covariance matrix.
    bpcm = np.genfromtxt(join(prefix, 'data', 'BK14',
                              'BK14_covmat_dust.dat'))
    # This matrix has shape (2277, 2277).
    # Nine ell bins times 253 different EE/BB/EB spectra.
    nmaps = 11 * 2
    nspec = nmaps * (nmaps + 1) / 2
    # Extract auto spectrum error bars for EE and BB.
    EE['sigma'] = np.sqrt(bpcm[range(iEE, nspec * 9, nspec),
                               range(iEE, nspec * 9, nspec)])
    BB['sigma'] = np.sqrt(bpcm[range(iBB, nspec * 9, nspec),
                               range(iBB, nspec * 9, nspec)])
    # Convert from Dl to Cl and multiply by Bl^2.
    EE['sigma'] = EE['sigma'] * 2.0 * np.pi / EE['ell'] / (EE['ell'] + 1.0)
    EE['sigma'] = EE['sigma'] * EE['Bl2']
    BB['sigma'] = BB['sigma'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)
    BB['sigma'] = BB['sigma'] * BB['Bl2']
    
    # Load bandpower expectation values for the LCDM+dust model that
    # corresponds to the bandpower covariance matrix.
    model = np.genfromtxt(join(prefix, 'data', 'BK14',
                               'BK14_fiducial_dust.dat'))
    # This array has shape (9, nspec+1). The first column is ell bin index.
    EE['expv'] = model[:, iEE+1]
    BB['expv'] = model[:, iBB+1]
    # Convert from Dl to Cl and multiply by Bl^2.
    EE['expv'] = EE['expv'] * 2.0 * np.pi / EE['ell'] / (EE['ell'] + 1.0)
    EE['expv'] = EE['expv'] * EE['Bl2']
    BB['expv'] = BB['expv'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)
    BB['expv'] = BB['expv'] * BB['Bl2']
    
    return (EE, BB)


def ACTpol_1yr_data(prefix='like_actpol_s1'):
    """
    Results from ACTPol first season data release.

    This function reads data from the ACTPol 2014 likelihood, which can be 
    downloaded from here:
    https://lambda.gsfc.nasa.gov/data/suborbital/ACT/act_pol/like_actpol_s1.tar.gz

    Parameters
    ----------
    prefix : str, optional
        Path to directory containing ACTPol first season data release.

    Returns
    -------
    data : tuple, two elements
        Bandpower statistics for EE (first element) and BB (second element).

    """
    
    # Data structures for results.
    EE = {'nsplit': 4}
    BB = {'nsplit': 4}
    
    return (EE, BB)


def ACTpol_2yr_data():
    # Data structures for results.
    EE = {'nsplit': 4}
    BB = {'nsplit': 4}
    
    return (EE, BB)


def SPTpol_data():
    # Data structures for results.
    EE = {'nsplit': 1}
    BB = {'nsplit': 1}
    
    return (EE, BB)


def POLARBEAR_data():
    # Data structures for results.
    EE = {'nsplit': 1}
    BB = {'nsplit': 1}
    
    return (EE, BB)


def QUIET_data():
    # Data structures for results.
    EE = {'nsplit': 1}
    BB = {'nsplit': 1}
    
    return (EE, BB)


def calculate_Nl(EE, BB):
    """
    Calculate Nl and effective fsky per bin.

    Calculation assumes that EE and BB have symmetric noise and filtering.
    Accounts for cross-spectrum based analysis and returns the Nl that 
    would be obtained if all data were coadded into a single map.

    Parameters
    ----------
    EE, BB : dict
        Data structures describing EE and BB bandpowers. Use the functions
        defined above to obtain these data for various experiments.
    
    Returns
    -------
    Nl : (N,) ndarray
        Nl values for all ell bins.
    fsky : (N,) ndarray
        Effective fsky for all ell bins, defined in a Knox formula sense.

    """

    # Number of data splits (should be same for EE and BB).
    if EE['nsplit'] == 1:
        x = 1.0
    else:
        x = float(EE['nsplit']) / (float(EE['nsplit']) - 1.0)
        
    # Quadratic expression for Nl
    a = (EE['sigma']**2 - BB['sigma']**2) * x
    b = 2.0 * (BB['expv'] * EE['sigma']**2 - EE['expv'] * BB['sigma']**2)
    c = (BB['expv'] * EE['sigma'])**2 - (EE['expv'] * BB['sigma'])**2
    Nl = (-1.0 * b + np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)

    # Calculate degrees-of-freedom as a function of ell.
    dof = BB['expv']**2 + 2.0 * BB['expv'] * Nl + x * Nl**2
    dof = 2.0 * dof / BB['sigma']**2
    # Convert to effective fsky based on ell bin definitions.
    fsky = dof / dof_fullsky(BB['bins'])

    return (Nl, fsky)


def dof_fullsky(bins):
    """
    Calculate degrees-of-freedom for the specified bin definitions 
    assuming a full sky experiment and tophat bins.

    Parameters
    ----------
    bins : (N+1,) ndarray
        List of bin edges. Length should be one more than the number of bins.
    
    Returns
    -------
    dof : (N,) ndarray
        Degrees-of-freedom per bin.

    """

    N = len(bins) - 1
    dof = np.zeros((N, ))
    # Just add up 2 * ell + 1 over the ell bin ranges.
    for i in range(N):
        dof[i] = np.sum(2.0 * np.arange(bins[i], bins[i+1]) + 1.0)

    return dof
    
