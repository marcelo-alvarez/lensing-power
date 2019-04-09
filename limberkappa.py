import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = '/Users/marcelo/Work/src/CAMB'
sys.path.insert(0,camb_path)
import camb
from camb import model, initialpower
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

omb    = 0.049
omc    = 0.261
h      = 0.68
ns     = 0.965
sigma8 = 0.81
Asnorm = 2e-9

def docamb():    

    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    H0 = 100*h; ombh2 = omb * h**2; omch2 = omc * h**2; mnu=0.001; 
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0, tau=0.06)
    pars.InitPower.set_params(As=Asnorm, ns=ns, r=0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    
    #calculate results for these parameters
    results = camb.get_results(pars)

    #Now get matter power spectra and sigma8 at redshift 0
    pars.set_matter_power(redshifts=[0.], kmax=2.0)
    
    #Linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    sigma8norm = np.array(results.get_sigma8())[0]
    
    #Normalize and reset InitPower
    As = Asnorm * (sigma8 / sigma8norm)**2
    print('Using As = ',As)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    
    #calculate results for normalized power spectrum
    results = camb.get_results(pars)
    
    #check if sigma8 is now what we want
    kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
    sigma8out = np.array(results.get_sigma8())[0]
    print('sigma8out = %f sigma8in = %f\n'%(sigma8out,sigma8))
    
    return pars,results

def getLimberKappa(pars,zmax,nonlinear):    

    #For calculating large-scale structure and lensing results yourself, get a power spectrum
    #interpolation object. In this example we calculate the CMB lensing potential power
    #spectrum using the Limber approximation, using PK=camb.get_matter_power_interpolator() function.
    #calling PK(z, k) will then get power spectrum at any k and redshift z in range.
    
    nz = 100 #number of steps to use for the radial/redshift integration
    kmax=40  #kmax to use
    
    #For Limber result, want integration over \chi (comoving radial distance), from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(pars)
    chistar = results.conformal_time(0)- results.tau_maxvis
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    zstar = zs[-1]

    #select out only values for z<zmax
    chis = chis[zs<zmax]
    zs   =   zs[zs<zmax]
    
    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #Get the matter power spectrum interpolation object (based on RectBivariateSpline). 
    #Here for lensing we want the power spectrum of the Weyl potential.
    PK = camb.get_matter_power_interpolator(pars, nonlinear=nonlinear, 
                                            hubble_units=False, k_hunit=False, kmax=kmax,
                                            var1=model.Transfer_Weyl,var2=model.Transfer_Weyl, zmax=zs[-1])

    #Get lensing window function (flat universe)
    win = ((chistar-chis)/(chis**2*chistar))**2

    #Do integral over chi
    ls = np.arange(2,5000+1, dtype=np.float64)
    cl_kappa=np.zeros(ls.shape)
    w = np.ones(chis.shape) #this is just used to set to zero k values out of range of interpolation
    for i, l in enumerate(ls):
        k=(l+0.5)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**4)

    cl_kappa*= (ls*(ls+1))**2

    return ls,cl_kappa
