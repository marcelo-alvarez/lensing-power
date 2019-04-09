import sys, platform, os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from limberkappa import *
import healpy as hp
import scipy as sp
from scipy.interpolate import *
from scipy.optimize    import * 

# set these to false to calculate the cls from the map and re-do the limber calculation, respectivly
readmapcls=True
readlimbcls=False
mapfile='websky-v0.0'

chisum = sp.special.chdtr # chdtr(n,x) is cumulative chi-squared dist of order n

sigmahigh  = 0.5+sp.special.erf(1./np.sqrt(2.))/2
sigmalow   = 0.5-sp.special.erf(1./np.sqrt(2.))/2
sigmahigh2 = 0.5+sp.special.erf(2./np.sqrt(2.))/2
sigmalow2  = 0.5-sp.special.erf(2./np.sqrt(2.))/2

def chisumhigh(x,n):
        return chisum(n,x)-sigmahigh
def chisumlow(x,n):
        return chisum(n,x)-sigmalow
def chisumhigh2(x,n):
        return chisum(n,x)-sigmahigh2
def chisumlow2(x,n):
        return chisum(n,x)-sigmalow2
def chi2sigma(n):
        chilow  = brentq(chisumlow,  0,10*n,args=[n,])
        chihigh = brentq(chisumhigh, 0,10*n,args=[n,])
        chilow2 = brentq(chisumlow2, 0,10*n,args=[n,])
        chihigh2= brentq(chisumhigh2,0,10*n,args=[n,])
        return chilow/n, chihigh/n, chilow2/n, chihigh2/n

chi2sigma = np.vectorize(chi2sigma)
    
def getclfrommap(fname,nside):
    if readmapcls:
        data = np.load(fname+'_cls.npz')
        return data['ell'],data['clh'],data['clf'],data['clt']
    else:
        f=hp.read_map(fname+'.fits')
        f=hp.ud_grade(f,nside)
        cl=hp.anafast(f)[1:]
        ell=np.arange(len(cl))+1
        np.savez(fname+'_cls.npz',ell=ell,cl=cl)
        return ell,cl

def getlimbcls():
    if readlimbcls:
        data=np.load('limber_cls.npz')
        return data['ls_low'],data['cl_kappa_low'],data['cl_kappa_lowlin'],data['ls'],data['cl_kappa'],data['cl_kappalin'],data['cl_camb']
    else:
        #Compare with CAMB's calculation:
        #note that to get CAMB's internal calculation accurate at the 1% level at L~2000, 
        #need lens_potential_accuracy=2. Increase to 4 for accurate match to the Limber calculation here
        pars,results = docamb()
        pars.set_for_lmax(2500,lens_potential_accuracy=2)
        results = camb.get_results(pars)
        cl_camb=results.get_lens_potential_cls(2500)
        #cl_camb[:,0] is phi x phi power spectrum (other columns are phi x T and phi x E)za
        ls_low,cl_kappa_low = getLimberKappa(pars,4.5,True)
        ls,cl_kappa = getLimberKappa(pars,1500.,True)
        ls_low,cl_kappa_lowlin = getLimberKappa(pars,4.5,False)
        ls,cl_kappalin = getLimberKappa(pars,1500.,False)
        np.savez('limber_cls.npz',ls_low=ls_low,cl_kappa_low=cl_kappa_low,cl_kappa_lowlin=cl_kappa_lowlin,
                 ls=ls,cl_kappa=cl_kappa,cl_kappalin=cl_kappalin,cl_camb=cl_camb)
        return ls_low,cl_kappa_low,cl_kappa_lowlin,ls,cl_kappa,cl_kappalin,cl_camb

def diffcomp(ell,clf,clm):
    return clm/clllf(ell) - 1

ls_map,clh_map,clf_map,clt_map = getclfrommap(mapfile,2048)
ls_low,cl_limber_low,cl_limber_lowlin,ls,cl_limber,cl_limberlin,cl_camb = getlimbcls()

#Make plot. Expect difference at very low-L from inaccuracy in Limber approximation, and
#very high L from differences in kmax (lens_potential_accuracy is only 2, though good by eye here)

cl_camb = 2*np.pi*cl_camb/4 # convert [l(l+1)]^2C_phi/2pi (what cl_camb is) to cl_kappa

siglow, sighigh, siglow2, sighigh2 = chi2sigma(2*ls_low+1)
clll_low   = siglow   * cl_limber_low
clll_high  = sighigh  * cl_limber_low
clll_low2  = siglow2  * cl_limber_low
clll_high2 = sighigh2 * cl_limber_low

clllf = interp1d(ls,cl_limber_low,bounds_error=False)

grid = plt.GridSpec(2,1,hspace=0.0,wspace=0.0,height_ratios=[4,1])

ax1= plt.subplot(grid[0,0])
ax2= plt.subplot(grid[1,0])
ax3= plt.subplot(grid[1,0])

dotc=(0.2,0.2,0.2,0.1)

dclh_map     = diffcomp(ls_map,clllf,clh_map)
dclf_map     = diffcomp(ls_map,clllf,clf_map)
dclt_map     = diffcomp(ls_map,clllf,clt_map)
dcl_limber  = diffcomp(ls,    clllf,cl_limber_lowlin)
dclll_low   = diffcomp(ls_low,clllf,clll_low)
dclll_low2  = diffcomp(ls_low,clllf,clll_low2)
dclll_high  = diffcomp(ls_low,clllf,clll_high)
dclll_high2 = diffcomp(ls_low,clllf,clll_high2)

lr=(1.0,0.5,0.5); lg=(0.5,1.0,0.5); lb=(0.5,0.5,1.0); lk=(0.5,0.5,0.5)

dots = 1.0
ax2.semilogx(ls_map,dclh_map,'o',markersize=dots,c=lr)
ax2.semilogx(ls_map,dclf_map,'o',markersize=dots,c=lb)
ax2.semilogx(ls_map,dclt_map,'o',markersize=dots,c=lk)
ax2.semilogx(ls_low,0*cl_limber_low, color='k')

ax2.fill_between(ls_low,dclll_low2,dclll_high2,facecolor='r',alpha=0.2)
ax2.fill_between(ls_low,dclll_low ,dclll_high ,facecolor='y',alpha=0.2)

ax1.loglog(ls_map,clh_map,'o',markersize=dots,c=lr)
ax1.loglog(ls_map,clf_map,'o',markersize=dots,c=lb)
ax1.loglog(ls_map,clt_map,'o',markersize=dots,c=lk)
ax1.loglog(ls_low,cl_limber_low, color='k')
ax1.loglog(ls,cl_limber_lowlin, color='k',ls=':')

ax1.legend(['WebSky Halos','WebSky Field','WebSky Total','Halofit','Linear'])

ax1.fill_between(ls_low,clll_low2,clll_high2,facecolor='r',alpha=0.2)
ax1.fill_between(ls_low,clll_low,clll_high,facecolor='y',alpha=0.2)

ax1.set_xlim([1,8000])
ax2.set_xlim([1,8000])
ax3.set_xlim([1,8000])

ax1.set_ylim([3e-11,4e-7])
#ax2.set_ylim([-1.2,0.6])
ax2.set_ylim([-0.2,0.2])
ax1.set_ylabel(r'$C_\ell^{\kappa\kappa}$')
ax2.set_xlabel(r'$\ell$')

ax3.semilogx(ls,dcl_limber, color='grey',ls=':')
ax3.tick_params(top=True,which='both',direction='inout')

ax1.set_xticklabels([])
plt.savefig(mapfile+'.pdf',bbox_inches='tight')

