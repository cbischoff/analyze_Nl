"""
analyze_Nl.py
=============

Derives N_l curves for published CMB B-mode results, including:
  - BICEP2 / Keck 2014 results
  - ACTPol 1st year (Naess et al) and 2nd year (Louis et al)
  - POLARBEAR 2014 results
  - QUIET Q-band and W-band results

Data functions
--------------

Most of the code included here is just for reading in data from various 
CMB experiment data releases. These functions have names like "BK14_data" or
"ACTpol_2yr_data".

All of the data functions take an optional 'prefix' argument, which should
be a path to the directory containing the appropriate data release. Data releases from CMB experiments are not included; you can find them at the
following links:
  - BICEP2/Keck : http://www.bicepkeck.org/
  - ACTPol : https://lambda.gsfc.nasa.gov/product/act/actpol_prod_table.cfm
  - POLARBEAR : https://lambda.gsfc.nasa.gov/product/polarbear/polarbear_prod_table.cfm
  - QUIET : https://lambda.gsfc.nasa.gov/product/quiet/index.cfm
If you download and extract the tarball from those links in the same directory
as this code, then the default values of the prefix argument should work.

The data functions all return a pair of dict structures, which contain 
statistics for the EE and BB bandpowers. These data strutures always contain
the following (key, value) pairs:
  nsplit
        number of split maps used for power spectrum analysis, nsplit = 1
        implies auto-spectrum analysis
  bins
        ell values that define the edges of each ell bin (one more value than
        the number of bins)
  ell
      mean ell value of each ell bin, evaluated from bandpower window
      functions when available
  expv
      bandpower expectation values, expressed as C_l (not D_l) in uK^2
  sigma
      bandpower error bars, expressed as C_l (not D_l) in uK^2
  Bl2
      beam window function (squared)
Some data functions will also include the following (key, value) pair:
  Nl_actual
      N_l curve provided with the data release (as opposed to the one we 
      will calculate)

Analysis functions
------------------

The calculate_Nl function uses expectation values and error bars for both 
BB and EE spectra to simultaneously solve for the noise power and bandpower
degrees-of-freedom. It assumes that noise and filtering are symmetric for 
E and B modes, which could break down for the case of pure-B estimators. 
For more details, see the posting here: (link TBD)

Also provided is a function to fit the N_l curve to a functional form with
a white noise component plus 1/ell noise.

"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from os.path import join
from scipy.optimize import minimize


def BK14_data(band='BK14_150', prefix='BK14_cosmomc'):
    """
    Results from BICEP2 / Keck BK14 data release (Keck Array and BICEP2 
    Collaborations, 2016).
    
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
    EE, BB : dict
        Bandpower statistics for EE and BB.

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
        ell = bpwf[:,0]
        EE['ell'][i] = np.average(ell, weights=bpwf[:,iEE+1])
        BB['ell'][i] = np.average(ell, weights=bpwf[:,iBB+1])
        # Mean value of Bl^2.
        sigma_rad = np.radians(fwhm / np.sqrt(8.0 * np.log(2)))
        Bl2 = np.exp(-1.0 * ell * (ell + 1) * sigma_rad**2)
        EE['Bl2'][i] = np.average(Bl2, weights=bpwf[:,iEE+1])
        BB['Bl2'][i] = np.average(Bl2, weights=bpwf[:,iBB+1])
        
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
    # Convert from Dl to Cl.
    EE['sigma'] = EE['sigma'] * 2.0 * np.pi / EE['ell'] / (EE['ell'] + 1.0)
    BB['sigma'] = BB['sigma'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)
    
    # Load bandpower expectation values for the LCDM+dust model that
    # corresponds to the bandpower covariance matrix.
    model = np.genfromtxt(join(prefix, 'data', 'BK14',
                               'BK14_fiducial_dust.dat'))
    # This array has shape (9, nspec+1). The first column is ell bin index.
    EE['expv'] = model[:, iEE+1]
    BB['expv'] = model[:, iBB+1]
    # Convert from Dl to Cl.
    EE['expv'] = EE['expv'] * 2.0 * np.pi / EE['ell'] / (EE['ell'] + 1.0)
    BB['expv'] = BB['expv'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)

    # Read the actual N_l spectra, which are provided in the data release.
    Nl_actual = np.genfromtxt(join(prefix, 'data', 'BK14', 'BK14_noise.dat'))
    # This array has shape (9, nspec+1). The first column is ell bin index.
    EE['Nl_actual'] = Nl_actual[:, iEE+1]
    BB['Nl_actual'] = Nl_actual[:, iBB+1]
    # Convert from Dl to Cl.
    EE['Nl_actual'] = EE['Nl_actual'] * 2.0 * np.pi / EE['ell'] / (EE['ell'] + 1.0)
    BB['Nl_actual'] = BB['Nl_actual'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)
    
    return (EE, BB)


def ACTpol_1yr_data(prefix='like_actpol_s1',
                    camb_file='camb_72686694_lensedcls.dat'):
    """
    Results from ACTPol first season data release (Naess et al, 2014).

    This function reads data from the ACTPol 2014 likelihood, which can be 
    downloaded from here:
    https://lambda.gsfc.nasa.gov/data/suborbital/ACT/act_pol/like_actpol_s1.tar.gz

    Parameters
    ----------
    prefix : str, optional
        Path to directory containing ACTPol first season likelihood.
    camb_file : str, optional
        Path to CAMB .dat file containing lensed-LCDM theory spectra.

    Returns
    -------
    EE, BB : dict
        Bandpower statistics for EE and BB.

    """
    
    # Data structures for results.
    # ACT uses cross-spectrum analysis with data split into four maps
    # alternating by day.
    EE = {'nsplit': 4}
    BB = {'nsplit': 4}

    # Ell bins: 57 bins, starting from ell=126
    #   delta-ell = 50 for bins 1--38
    #   delta-ell = 100 for bins 39--43
    #   delta-ell = 200 for bins 44--46
    #   delta-ell = 400 for bins 47--54
    #   delta-ell = 800 for bins 55--57
    # But drop the first two bins, which aren't used for EE/BB spectra.
    # Include the upper limit of the last bin.
    bins = np.genfromtxt(join(prefix, 'data', 'data_act', 'Binning.dat'))
    nbin = bins.shape[0]
    EE['bins'] = np.zeros(nbin - 1)
    EE['bins'][0:nbin-2] = bins[2:,0]
    EE['bins'][nbin-2] = bins[-1,1]
    BB['bins'] = EE['bins']

    # Read bandpower error bars.
    EEpath = join(prefix, 'data', 'data_act', 'spectrum_EE.dat')
    EEdata = np.genfromtxt(EEpath)
    EE['sigma'] = EEdata[:,2] # sigma(C_l), in uK^2
    BBpath = join(prefix, 'data', 'data_act', 'spectrum_BB.dat')
    BBdata = np.genfromtxt(BBpath)
    BB['sigma'] = BBdata[:,2] # sigma(C_l), in uK^2
    
    # Read ACTPol bandpower window functions.
    bpwffile = join(prefix, 'data', 'data_act', 'BblMean_Pol.dat')
    bpwf = np.genfromtxt(bpwffile)
    # Drop the first two window functions, which correspond to ell=150 and
    # ell=200 bins, not included in EE/BB spectra.
    bpwf = bpwf[2:,:]
    # Get EE and BB theory spectra (generated with CAMB).
    (ell, EE_th, BB_th) = theory_spectra(camb_file)
    # Find common ell range between ACTPol bpwf and CAMB theory spectrum.
    # ACTPol bpwf are defined from ell = 1--9000.
    lmin = np.max([1, ell.min()])
    lmax = np.min([9000, ell.max()])
    i0_bpwf = int(lmin - 1)
    i1_bpwf = int(lmax)
    i0_th = int(np.where(ell == lmin)[0][0])
    i1_th = int(np.where(ell == lmax)[0][0] + 1)
    # Calculate mean ell for each bandpower, bandpower expectation values,
    # and beam window function.
    nbin = bpwf.shape[0]
    EE['ell'] = np.zeros(nbin)
    BB['ell'] = np.zeros(nbin)
    EE['expv'] = np.zeros(nbin)
    BB['expv'] = np.zeros(nbin)
    EE['Bl2'] = np.zeros(nbin)
    BB['Bl2'] = np.zeros(nbin)
    fwhm_deg = 1.3 / 60.
    sigma_rad = np.radians(fwhm_deg / np.sqrt(8.0 * np.log(2.0)))
    Bl2 = np.exp(-1.0 * ell * (ell + 1.0) * sigma_rad**2)
    for i in range(nbin):
        # Mean ell for bandpower.
        EE['ell'][i] = np.average(np.arange(i0_bpwf, i1_bpwf),
                                  weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['ell'][i] = EE['ell'][i]
        # Bandpower expectation values from theory spectra.
        EE['expv'][i] = np.average(EE_th[i0_th:i1_th],
                                   weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['expv'][i] = np.average(BB_th[i0_th:i1_th],
                                   weights=bpwf[i, i0_bpwf:i1_bpwf])
        # Beam window function.
        EE['Bl2'][i] = np.average(Bl2[i0_th:i1_th],
                                  weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['Bl2'][i] = EE['Bl2'][i]

    return (EE, BB)


def ACTpol_2yr_data(prefix='actpollite_s2_like',
                    camb_file='camb_72686694_lensedcls.dat'):    
    """
    Results from ACTPol second season data release (Louis et al, 2016).

    This function reads data from the ACTPol 2016 likelihood, which can be 
    downloaded from here:
    https://lambda.gsfc.nasa.gov/data/suborbital/ACT/actpol_2016/actpollite_s2_like.tar.gz
    It also reads BB spectrum data from a .dat file that is copied from 
    Table 5 of Louis et al. That .dat file is included in the git repo
    with this code.

    Parameters
    ----------
    prefix : str, optional
        Path to directory containing ACTPol second season likelihood.
    camb_file : str, optional
        Path to CAMB .dat file containing lensed-LCDM theory spectra.

    Returns
    -------
    EE, BB : dict
        Bandpower statistics for EE and BB.

    """

    # Data structures for results.
    EE = {'nsplit': 4}
    BB = {'nsplit': 4}

    # Read .dat file containing BB spectra from Table 5 of Louis et al.
    # Columns 1,2 contain ell bin boundaries.
    # Column 4 contains BB error bars, specified for D_l in uK^2.
    BBdata = np.genfromtxt('actpol_louis_2016_tbl5.dat')
    BB['sigma'] = BBdata[:,4]
    EE['bins'] = np.concatenate((BBdata[:,1], [BBdata[-1,2]]))
    BB['bins'] = EE['bins']

    # Read bandpower error bars for EE.
    EEpath = join(prefix, 'fullspectra', 'spectrum_EE.dat')
    EEdata = np.genfromtxt(EEpath)
    EE['sigma'] = EEdata[:,2] # sigma(C_l), in uK^2

    # Read ACTPol bandpower window functions.
    bpwffile = join(prefix, 'fullspectra', 'BblMean_EE.dat')
    bpwf = np.genfromtxt(bpwffile)
    # Get EE and BB theory spectra (generated with CAMB).
    (ell, EE_th, BB_th) = theory_spectra(camb_file)
    # Find common ell range between ACTPol bpwf and CAMB theory spectrum.
    # ACTPol bpwf are defined from ell = 1--9000.
    lmin = np.max([1, ell.min()])
    lmax = np.min([9000, ell.max()])
    i0_bpwf = int(lmin - 1)
    i1_bpwf = int(lmax)
    i0_th = int(np.where(ell == lmin)[0][0])
    i1_th = int(np.where(ell == lmax)[0][0] + 1)
    # Calculate mean ell for each bandpower, bandpower expectation values,
    # and beam window function.
    nbin = bpwf.shape[0]
    EE['ell'] = np.zeros(nbin)
    BB['ell'] = np.zeros(nbin)
    EE['expv'] = np.zeros(nbin)
    BB['expv'] = np.zeros(nbin)
    EE['Bl2'] = np.zeros(nbin)
    BB['Bl2'] = np.zeros(nbin)
    fwhm_deg = 1.3 / 60.
    sigma_rad = np.radians(fwhm_deg / np.sqrt(8.0 * np.log(2.0)))
    Bl2 = np.exp(-1.0 * ell * (ell + 1.0) * sigma_rad**2)
    for i in range(nbin):
        # Mean ell for bandpower.
        EE['ell'][i] = np.average(np.arange(i0_bpwf, i1_bpwf),
                                  weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['ell'][i] = EE['ell'][i]
        # Bandpower expectation values from theory spectra.
        EE['expv'][i] = np.average(EE_th[i0_th:i1_th],
                                   weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['expv'][i] = np.average(BB_th[i0_th:i1_th],
                                   weights=bpwf[i, i0_bpwf:i1_bpwf])
        # Beam window function.
        EE['Bl2'][i] = np.average(Bl2[i0_th:i1_th],
                                  weights=bpwf[i, i0_bpwf:i1_bpwf])
        BB['Bl2'][i] = EE['Bl2'][i]

    # Convert BB error bars from D_l to C_l.
    BB['sigma'] = BB['sigma'] * 2.0 * np.pi / BB['ell'] / (BB['ell'] + 1.0)

    # Read N_l for various fields from ACTpol data release.
    Nl_actual = np.genfromtxt(join(prefix, 'fullspectra', 'noise_EE.dat'))
    # This file contains ell=300 bin, not present in other files?
    Nl_actual = Nl_actual[1:,:]
    # Four N_l spectra:
    #   1 = D5 deep field
    #   2 = D6 deep field
    #   3 = D56 wide field, PA1 detector array
    #   4 = D56 wide field, PA2 detector array
    EE['Nl_actual_D5'] = Nl_actual[:,1]
    EE['Nl_actual_D6'] = Nl_actual[:,2]
    EE['Nl_actual_D56_PA1'] = Nl_actual[:,3]
    EE['Nl_actual_D56_PA2'] = Nl_actual[:,4]
    
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


def theory_spectra(camb_file):
    """
    Read CMB theory spectra from CAMB .dat file

    Parameters
    ----------
    camb_file : str
        Path to CAMB .dat file (not .fits!)

    Returns
    -------
    ell : (N,) ndarray
        Ell values for EE, BB spectra. Usually starts from ell=2.
    EE : (N,) ndarray
        EE spectrum C_l, in uK^2
    BB : (N,) ndarray
        BB spectrum C_l, in uK^2

    """

    camb = np.genfromtxt(camb_file)
    ell = camb[:,0]
    # For EE and BB, convert from D_l to C_l.
    EE = camb[:,2] * 2.0 * np.pi / ell / (ell + 1.0)
    BB = camb[:,3] * 2.0 * np.pi / ell / (ell + 1.0)

    return (ell, EE, BB)


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
    
