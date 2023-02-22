import numpy as np
import treecorr
from scipy import stats
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import treecorr
import pandas as pd
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import threading
ncores = threading.active_count()
from scipy.special import binom, comb
from itertools import combinations
import time
"""Code to compute gamma_t and wg+ for lightcones with treecorr"""
class lensingPCF():
    def __init__(self,*args,**qwargs):
        """Init : First catalog: clustering catalog, second catalog source catalog"""
        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)
                  
        binsfile = qwargs['binsfile']
        self.computation = qwargs['computation']
        if self.computation in ['WGP','WGG']:
            self.bins1 = binsfile[0]
            self.bins2 = binsfile[1]
        else:
            self.bins1 = binsfile[0]

        self.min_sep = np.min(self.bins1)
        self.max_sep = np.max(self.bins1)
        self.nbins = len(self.bins1)

        self.units = qwargs['units']
        self.npatch = qwargs['npatch']
        self.upatches = np.linspace(0,self.npatch-1,self.npatch,dtype='int')

        self.npatch_random = self.npatch
        self.patch_RR = 'False'
        self.RR_calc = None
        
        if 'bin_slop' in qwargs:
            self.bin_slop = qwargs['bin_slop']
        else:
            self.bin_slop = 0.
        
        if self.npatch == 1:
            self.var_method = "shot"
        else:
            self.var_method = "jackknife"
       
        self.rand2 = None
        self.cov = None
        
        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        len1 = len(ra)
        
        self.z =  qwargs['Z']
        dc = self.cosmo.comoving_distance(self.z).value

        ra2 =  qwargs['RA2']
        dec2 =  qwargs['DEC2']
        w2 =  qwargs['W2'] 
        g1 = qwargs['g1']
        g2 = qwargs['g2']
        len2 = len(ra2)
        
        self.z2 =  qwargs['Z2']
        dc2 = self.cosmo.comoving_distance(self.z2).value           
 
        self.data1 = treecorr.Catalog(ra=ra,dec=dec,r=dc,w=w,ra_units=self.units,\
            dec_units=self.units,npatch=self.npatch)
    
        self.data2 = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,w=w2,g1=g1,g2=g2,ra_units=self.units,\
            dec_units=self.units,npatch=self.npatch,patch_centers=self.data1.patch_centers)    
        
        self.varg = treecorr.calculateVarG(self.data2)

        posC = np.column_stack([ra,dec])
        pos = np.column_stack([ra2,dec2])
        nrows, ncols = pos.shape
        dtype={'names':['f{}'.format(i) for i in range(ncols)],
               'formats':ncols * [posC.dtype]}
        
        uniquelen = np.maximum(len(np.unique(posC,axis=0)),len(np.unique(pos,axis=0)))
        C,xind,yind = np.intersect1d(posC.view(dtype), pos.view(dtype),return_indices=True)
        self.w_int1 = self.data1.w[xind]
        self.w_int2 = self.data2.w[yind]        
        
        if (len(self.w_int1) > 0) & (len(self.w_int1) < uniquelen):
            self.corr = 'subsample'
        elif len(self.w_int1)==0:
            self.corr = 'cross'
        elif len(self.w_int1)==uniquelen:
            self.corr  = 'auto'

        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']        
        z_r =  qwargs['Z_r']

        dc_r = self.cosmo.comoving_distance(z_r).value
        self.rand1 = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,w=w_r,ra_units=self.units,\
            dec_units=self.units,npatch=self.npatch_random,patch_centers=self.data1.patch_centers,is_rand=1)
            

        if 'RA_r2' in qwargs:
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']        
            z_r2 =  qwargs['Z_r2']
            dc_r2 = self.cosmo.comoving_distance(z_r2).value
            self.rand2 = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,ra_units=self.units,\
                dec_units=self.units,npatch=self.npatch_random,patch_centers=self.data1.patch_centers,is_rand=1)


    def compute_norm(self):
        
        if self.computation == "WGP" and self.corr =="cross" and self.rand2 is None:
            raise ValueError("You must provide at least two random catalogs for wg+ cross estimation.")

        self.rgnorm = np.zeros(self.npatch)
        self.rrnorm = np.zeros(self.npatch)
        self.ngnorm = np.zeros(self.npatch)
        patchD = np.unique(self.data1.patch)

        if self.npatch == 1:
            self.rgnorm = self.rand1.sumw*self.data2.sumw
            if self.corr == "auto":
                self.ngnorm = (self.data1.sumw)**2 - np.sum((self.data1.w)**2)
                self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
                
            elif self.corr == "subsample":
                self.ngnorm = self.data1.sumw*self.data2.sumw - np.sum(self.w_int1*self.w_int2)
                
            elif self.corr == "cross":
                self.ngnorm = self.data1.sumw*self.data2.sumw
                if self.computation =="WGP":
                    self.rrnorm = self.rand1.sumw*self.rand2.sumw
                else:
                    self.rrnorm = (self.rand1.sumw)**2 - np.sum((self.rand1.w)**2)
            
        
        elif self.npatch > 1:
            for i in range(0,len(patchD)):
                cond1 = self.data1.patch == patchD[i]
                cond2 = self.data2.patch == patchD[i]
                cond3 = self.rand1.patch == patchD[i]
                wd1 = self.data1.w[~cond1]
                wd2 = self.data2.w[~cond2]
                wr1 = self.rand1.w[~cond3]
            
                if self.corr =='auto':
                    self.ngnorm[i]= (np.sum(wd1)**2 - np.sum(wd1**2))
                    self.rrnorm[i]=(np.sum(wr1)**2 - np.sum(wr1**2))
                    self.rgnorm[i] = np.sum(wr1)*np.sum(wd1)

                elif self.corr =='subsample':
                    ra_t = np.hstack([self.data1.patches[x].ra for x in self.upatches if x!=i])
                    dec_t = np.hstack([self.data1.patches[x].dec for x in self.upatches if x!=i])
                    ra_t2 = np.hstack([self.data2.patches[x].ra for x in self.upatches if x!=i])
                    dec_t2 = np.hstack([self.data2.patches[x].dec for x in self.upatches if x!=i])
                    w_t = np.hstack([self.data1.patches[x].w for x in self.upatches if x!=i])
                    w_t2 = np.hstack([self.data2.patches[x].w for x in self.upatches if x!=i])
                    
                    posC = np.column_stack([ra_t,dec_t])
                    pos = np.column_stack([ra_t2,dec_t2])
                    nrows, ncols = pos.shape
                    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                           'formats':ncols * [posC.dtype]}

                    uniquelen = len(np.unique(pos,axis=0))
                    C,xind,yind = np.intersect1d(posC.view(dtype), pos.view(dtype),return_indices=True)
                    w_int1 = w_t[xind]
                    w_int2 = w_t2[yind]
                    self.ngnorm[i]= np.sum(wd1)*np.sum(wd2) - np.sum(w_int1*w_int2)
                    self.rgnorm[i]= np.sum(wr1)*np.sum(wd2)
                    
                    if self.rand2 is None : 
                        self.rrnorm[i] = self.rrnorm[i]=(np.sum(wr1)**2 - np.sum(wr1**2))
                    else :
                        ra_t = np.hstack([self.rand1.patches[x].ra for x in self.upatches if x!=i])
                        dec_t = np.hstack([self.rand1.patches[x].dec for x in self.upatches if x!=i])
                        ra_t2 = np.hstack([self.rand2.patches[x].ra for x in self.upatches if x!=i])
                        dec_t2 = np.hstack([self.rand2.patches[x].dec for x in self.upatches if x!=i])
                        w_t = np.hstack([self.rand1.patches[x].w for x in self.upatches if x!=i])
                        w_t2 = np.hstack([self.rand2.patches[x].w for x in self.upatches if x!=i])
                        posC = np.column_stack([ra_t,dec_t])
                        pos = np.column_stack([ra_t2,dec_t2])
                        nrows, ncols = pos.shape
                        dtype={'names':['f{}'.format(i) for i in range(ncols)],
                               'formats':ncols * [posC.dtype]}

                        uniquelen = len(np.unique(pos,axis=0))
                        C,xind,yind = np.intersect1d(posC.view(dtype), pos.view(dtype),return_indices=True)
                        w_int1 = w_t[xind]
                        w_int2 = w_t2[yind]
                        self.rrnorm[i] = np.sum(wr1)*np.sum(wr2) - np.sum(w_int1*w_int2)
                    
                elif self.corr == 'cross':
                    self.ngnorm[i]= np.sum(wd1)*np.sum(wd2)
                    self.rgnorm[i]= np.sum(wr1)*np.sum(wd2)
                    if self.computation == 'WGP':
                        cond4 = np.where(self.rand2.patch == patchD[i])[0]
                        wr2 = self.rand2.w[cond4]
                        self.rrnorm[i] = np.sum(wr1)*np.sum(wr2)  
                    else:
                        self.rrnorm[i]=(np.sum(wr1)**2 - np.sum(wr1**2))

                      
    """Routines to compute wg+/gammat/wgg for a single patch"""
    def combine_pairs_DS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[1].weight)*(self.rgnorm/self.ngnorm) - corrs[1].xi

    def combine_pairs_RS(self,corrs):
        return corrs[0].xi*(corrs[0].weight/corrs[2].weight)*(self.rrnorm/self.ngnorm) - \
        corrs[1].xi*(corrs[1].weight/corrs[2].weight)*(self.rrnorm/self.rgnorm)
    
    
    def combine_pairs_RS_clustering(self,corrs):
        return (corrs[0].weight/corrs[2].weight)*(self.rrnorm/self.ngnorm) - \
            2*(corrs[1].weight/corrs[2].weight)*(self.rrnorm/self.rgnorm) + 1.
    

    def combine_pairs_RS_proj(self,corrs):
        xirppi_t = np.zeros((len(self.bins2)-1,self.nbins-1))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        rr = corrs[2*int(len(corrs)/3):len(corrs)]
            
        for i in range(0,len(self.bins2)-1):
            corrs = [ng[i],rg[i],rr[i]]
            xirppi_t[i] = self.combine_pairs_RS(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    
    def combine_pairs_RS_proj_clustering(self,corrs):
        xirppi_t = np.zeros((len(self.bins2)-1,self.nbins-1))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        rr = corrs[2*int(len(corrs)/3):len(corrs)]

        for i in range(0,len(self.bins2)-1):
            corrs = [ng[i],rg[i],rr[i]]
            xirppi_t[i] = self.combine_pairs_RS_clustering(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    
    
    def combine_pairs_DS_proj(self,corrs):
        xirppi_t = np.zeros((len(self.bins2)-1,self.nbins-1))
        ng = corrs[0:int(len(corrs)/3)]
        rg = corrs[int(len(corrs)/3):2*int(len(corrs)/3)]
        for i in range(0,len(self.bins2)-1):
            corrs = [ng[i],rg[i]]
            xirppi_t[i] = self.combine_pairs_DS(corrs)
        xirppi_t = np.sum(xirppi_t*self.dpi,axis=0)
        return xirppi_t
    
    """Routines to compute wg+/gammat for multiple patch"""
    def get_rp_pairs(self,corrs):
        return corrs[0].weight
    
    def get_xi(self,corrs):
        return corrs[0].xi
    
    def get_jackknife_rp_pairs_clustering(self,corrNG,corrRG,corrRR):
        out_DD=treecorr.build_multi_cov_design_matrix([corrNG], 'jackknife', func=self.get_rp_pairs)
        out_DR=treecorr.build_multi_cov_design_matrix([corrRG], 'jackknife', func=self.get_rp_pairs)
        out_RR=treecorr.build_multi_cov_design_matrix([corrRR], 'jackknife', func=self.get_rp_pairs)
        
        pairs_DD,weight1_ = zip(*out_DD)
        pairs_DR,weight2_ = zip(*out_DR)
        pairs_RR,weight3_ = zip(*out_RR)

        pairs_DD = np.array(pairs_DD)
        pairs_DR = np.array(pairs_DR)
        pairs_RR = np.array(pairs_RR)

        pairs_DD_norm = pairs_DD/self.ngnorm[:,np.newaxis]
        pairs_DR_norm = pairs_DR/self.rgnorm[:,np.newaxis]
        pairs_RR_norm = pairs_RR/self.rrnorm[:,np.newaxis] 
        
        return pairs_DD_norm,pairs_DR_norm,pairs_RR_norm
    
    def get_jackknife_rp_pairs(self,corrNG,corrRG,corrRR):
        
        out_xiDD = treecorr.build_multi_cov_design_matrix([corrNG], 'jackknife', func=self.get_xi)
        out_xiDR = treecorr.build_multi_cov_design_matrix([corrRG], 'jackknife', func=self.get_xi)

        xi_DD,_ = zip(*out_xiDD)
        xi_DR,_ = zip(*out_xiDR)

        xi_DD = np.array(xi_DD)
        xi_DR = np.array(xi_DR)

        DD,DR,RR = self.get_jackknife_rp_pairs_clustering(corrNG,corrRG,corrRR)
        
        return xi_DD,xi_DR,DD,DR,RR

    
    def get_jackknife_rppi_pairs_clustering(self,dictNG,dictRG,dictRR):
        
        out_DD=[treecorr.build_multi_cov_design_matrix([x], 'jackknife', func=self.get_rp_pairs) for x in dictNG.values()]
        out_DR=[treecorr.build_multi_cov_design_matrix([x], 'jackknife', func=self.get_rp_pairs) for x in dictRG.values()]
        out_RR=[treecorr.build_multi_cov_design_matrix([x], 'jackknife', func=self.get_rp_pairs) for x in dictRR.values()]
        
        pairs_DD,weight1_ = zip(*out_DD)
        pairs_DR,weight2_ = zip(*out_DR)
        pairs_RR,weight3_ = zip(*out_RR)

        pairs_DD = np.array(pairs_DD)
        pairs_DR = np.array(pairs_DR)
        pairs_RR = np.array(pairs_RR)

        pairs_DD_norm = np.array([x/self.ngnorm[:,np.newaxis] for x in pairs_DD])
        pairs_DR_norm = np.array([x/self.rgnorm[:,np.newaxis] for x in pairs_DR])
        pairs_RR_norm = np.array([x/self.rrnorm[:,np.newaxis] for x in pairs_RR])
        
        return pairs_DD_norm,pairs_DR_norm,pairs_RR_norm
    

    def get_jackknife_rppi_pairs(self,dictNG,dictRG,dictRR):
        
        out_xiDD = [treecorr.build_multi_cov_design_matrix([x], 'jackknife', func=self.get_xi) for x in dictNG.values()]
        out_xiDR = [treecorr.build_multi_cov_design_matrix([x], 'jackknife', func=self.get_xi) for x in dictRG.values()]

        xi_DD,_ = zip(*out_xiDD)
        xi_DR,_ = zip(*out_xiDR)

        xi_DD = np.array(xi_DD)
        xi_DR = np.array(xi_DR)

        DD,DR,RR = self.get_jackknife_rppi_pairs_clustering(dictNG,dictRG,dictRR)
        
        return xi_DD,xi_DR,DD,DR,RR
    

    def combine_jack_pairs_rp(self,corrNG,corrRG,corrRR):

        NG,RG,wNG,wRG,wRR = self.get_jackknife_rp_pairs(corrNG,corrRG,corrRR)

        xi = (NG*wNG - RG*wRG)/wRR
        xi_mean = np.mean(xi,axis=0)
        xi = xi - xi_mean
        C = (1.-1./self.npatch)*np.dot(xi.T,xi)
        return xi_mean,C
    

    
    def combine_jack_pairs_rppi(self,dictNG,dictRG,dictRR):

        NG,RG,wNG,wRG,wRR = self.get_jackknife_rppi_pairs(dictNG,dictRG,dictRR)
        
        xirppi = (NG*wNG - RG*wRG)/wRR
        wgp = np.sum(xirppi*self.dpi,axis=0)
        wgp_mean = np.mean(wgp,axis=0)
        wgp = wgp - wgp_mean
        C = (1.-1./self.npatch)*np.dot(wgp.T,wgp)
        
        return wgp_mean,C

    
    def combine_jack_pairs_rppi_clustering(self,dictNG,dictRG,dictRR):
  
        NG,RG,RR = self.get_jackknife_rppi_pairs_clustering(dictNG,dictRG,dictRR)
    
        xirppi = NG/RR - 2*RG/RR + 1.
        wgg = np.sum(xirppi*self.dpi,axis=0)
        wgg_mean = np.mean(wgg,axis=0)
        wgg = wgg - wgg_mean
        C = (1.-1./self.npatch)*np.dot(wgg.T,wgg)
        return wgg_mean,C

    
    def compute_gammat(self,min_rpar=0):
        min_rpar = min_rpar
        self.compute_norm()
        """Distances sources > Distances lens + x Mpc """
     
        ng = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,
                                      max_sep=self.max_sep,min_rpar=min_rpar,metric="Rlens",bin_slop=self.bin_slop,var_method=self.var_method)

        rg = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,
                                      max_sep=self.max_sep,min_rpar=min_rpar,metric="Rlens",bin_slop=self.bin_slop,var_method=self.var_method)
        
        rr = treecorr.NNCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,
                                      max_sep=self.max_sep,min_rpar=min_rpar,metric="Rlens",bin_slop=self.bin_slop,var_method=self.var_method)

        ng.process(self.data1,self.data2)
        rg.process(self.rand1,self.data2)
        rr.process(self.rand1,self.rand2)

        corrs=[ng,rg,rr]
        if self.var_method =="shot":
            xi = self.combine_pairs_RS(corrs)
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm)) 
        else:
            xi,cov = self.combine_jack_pairs_rp(ng,rg,rr)
            self.xi = xi
            self.cov = cov
            err = np.sqrt(np.diag(cov))
            
        rnorm = ng.rnom
        meanr = rg.meanr
        meanlogr =  rg.meanlogr

        return rg.rnom,meanr,meanlogr,xi,err


    def compute_wgp(self):
        
        self.compute_norm()
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""
        self.dpi = abs(self.bins2[1] - self.bins2[0])
        dictNG = {}
        dictRG = {}
        dictRR = {}
        
        for i in range(0,len(self.bins2)-1):
            pi_min = self.bins2[i]
            pi_max = self.bins2[i+1]
            dictNG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)
            dictRG[i] = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)
            dictRR[i] = treecorr.NNCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)

            dictNG[i].process(self.data1,self.data2)
            dictRG[i].process(self.rand1,self.data2)
            if self.rand2 is None:
                dictRR[i].process(self.rand1,self.rand1)
            else :
                dictRR[i].process(self.rand1,self.rand2)           
                
        catNG = list(dictNG.values())
        catRG =  list(dictRG.values())
        catRR = list(dictRR.values())
        corrs = catNG + catRG + catRR
        
        rg = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,metric="Rperp")
        rg.process(self.rand1,self.data2)
        meanr = rg.meanr
        meanlogr =  rg.meanlogr
        
        if self.var_method =="shot":
            xi_mean = self.combine_pairs_RS_proj(corrs)
            #Wrong
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))    
        else:
            xi_mean,cov = self.combine_jack_pairs_rppi(dictNG,dictRG,dictRR)
            self.xi = xi_mean            
            self.cov = cov
            err = np.sqrt(np.diag(cov))

        return dictNG[0].rnom,meanr,meanlogr,xi_mean,err
    
    
    def compute_wgg(self):        
        self.compute_norm()
        """Treecorr provides 2D counts computation but only for linear bins here we compute xi(rp,pi) by looping over radial distance instead"""

        dictNN = {}
        dictRN = {}
        dictRR = {}
        self.dpi = abs(self.bins2[1] - self.bins2[0])        
        
        for i in range(0,len(self.bins2)-1):
            pi_min = self.bins2[i]
            pi_max = self.bins2[i+1]
            dictNN[i] = treecorr.NNCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)
            dictRN[i] = treecorr.NNCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)
            dictRR[i] = treecorr.NNCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop,var_method=self.var_method)

            dictNN[i].process(self.data1,self.data1)
            dictRN[i].process(self.rand1,self.data1)
            dictRR[i].process(self.rand1,self.rand1)

        catNN = list(dictNN.values())
        catRN = list(dictRN.values())
        catRR = list(dictRR.values())
        corrs = catNN + catRN + catRR
        
        rg = treecorr.NGCorrelation(bin_type='Log',nbins=self.nbins-1,min_sep=self.min_sep,max_sep=self.max_sep,metric="Rperp")
        rg.process(self.rand1,self.data2)
        meanr = rg.meanr
        meanlogr =  rg.meanlogr
        
        if self.var_method =="shot":
            xi_mean = self.combine_pairs_RS_proj_clustering(corrs)        
            err = np.sqrt(self.varg/rg.weight*(self.rgnorm/self.ngnorm))    
        else:
            xi_mean,cov = self.combine_jack_pairs_rppi_clustering(dictNN,dictRN,dictRR)
            self.xi = xi_mean            
            self.cov = cov
            err = np.sqrt(np.diag(cov))

        return dictNN[0].rnom,meanr,meanlogr,xi_mean,err 
    
    def get_cov(self):
        return self.cov
    
    def get_measurements(self):
        if self.computation == 'WGP':
            r_,meanr_,meanlogr_,xi_mean,err = self.compute_wgp()
        elif self.computation == 'WGG':
            r_,meanr_,meanlogr_,xi_mean,err = self.compute_wgg()
        else:
            r_,meanr_,meanlogr_,xi_mean,err = self.compute_gammat()
            
        if self.npatch == 1: 
            return meanr_,xi_mean,err
        elif self.npatch > 1:
            cov = self.get_cov()
            return meanr_,xi_mean,cov

    def combine_measurements(self,corr):
        self.xi_tot = np.column_stack((self.xi,corr.xi))
        self.xi_mean = np.mean(self.xi_tot,axis=0)
        xi = self.xi_tot - self.xi_mean
        Cov = (1.-1./self.npatch)*np.dot(xi.T,xi)
        return Cov

    
    def get_cov(self):
        return self.cov
    
