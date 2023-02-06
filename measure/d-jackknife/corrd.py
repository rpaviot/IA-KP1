import numpy as np
from scipy import stats
import treecorr
from astropy.cosmology import FlatLambdaCDM
from scipy.special import binom, comb
from itertools import combinations
from jackknife import jackknife

class deleted_jackknife(jackknife):
    
    def __init__(self,*args,**qwargs):
        
        if 'bin_slop' in qwargs:
            self.bin_slop = qwargs['bin_slop']
        else:
            self.bin_slop = 0.
    
        self.jackpairs = []    
        self.read_input(**qwargs)
        self.get_combinations()
        super().__init__(**qwargs)


    def dist(self,x,y):
        return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)


    def get_combinations(self):
        
        """get a list of all d-deleted jackknife combinations"""

        self.comb = int(binom(self.Ns,self.Nd))
        comb = [",".join(map(str, comb)) for comb in combinations(self.upatches, self.Nd)]        
        list_comb = list(map(eval,comb))
        self.array_comb = np.array(list_comb)

        
        """get the total numbers of jackknife pairs"""

        self.npairs = int((self.Ns*(self.Ns-1))/2)        

        """Get all one by one combinations between unique patches"""

        for i in range(0,len(self.upatches)):
            centers = self.data1_._centers
            dist_cent = np.array([self.dist(centers[i],centers[j]) for j in range(0,len(centers))])
            dist_cent = dist_cent/np.min(dist_cent[dist_cent !=0])
            cond = ((dist_cent > 0.99) & (dist_cent < 3.0))
            link = self.upatches[cond]
            self.jackpairs.append([(i,link[j]) for j in range(0,len(link))])           
            
        self.jackpairs = np.vstack(self.jackpairs)

    def read_input(self,**qwargs):

        self.Om =  qwargs['Om0']
        self.cosmo = FlatLambdaCDM(Om0=self.Om,H0=100)

        self.Ns = qwargs['Ns']
        self.Nd = qwargs['Nd']
        self.computation = qwargs['computation']
        binsfile = qwargs['binsfile']
        self.upatches = np.linspace(0,self.Ns-1,self.Ns,dtype='int')

        if self.computation in ['WGP','WGG','XIL']:
            self.twoD=True
            self.bins1 = binsfile[0]
            self.bins2 = binsfile[1]
            self.pimax = np.max(self.bins2)
            self.du = abs(self.bins2[1]-self.bins2[0])
        else:
            self.twoD=False
            self.bins1 = binsfile[0]

        """read data/random"""

        ra =  qwargs['RA']
        dec =  qwargs['DEC']
        w =  qwargs['W']
        z =  qwargs['Z']
        dc = self.cosmo.comoving_distance(z)
        
        g1 = np.zeros(len(ra))
        g2 = np.zeros(len(ra))
        g12 = np.zeros(len(ra))
        g22 = np.zeros(len(ra))
        
        
        if 'g1' in qwargs:
            g1 = qwargs['g1']
            g2 = qwargs['g2']
            
        self.data1_ = treecorr.Catalog(ra=ra,dec=dec,r=dc,g1=g1,g2=g2,w=w,ra_units='degree',\
            dec_units='degree',npatch=self.Ns)


        if 'RA2' in qwargs:
            self.corr = 'cross'
            
            ra2 =  qwargs['RA2']
            dec2 =  qwargs['DEC2']
            w2 =  qwargs['W2'] 
            z2 = qwargs['Z2']
            dc2 = self.cosmo.comoving_distance(z2).value
            
            if 'g12' in qwargs:
                g12 = qwargs['g1']
                g22 = qwargs['g2']
            
            self.data2_ = treecorr.Catalog(ra=ra2,dec=dec2,r=dc2,w=w2,g1=g12,g2=g22,ra_units='degree',\
                    dec_units='degree',npatch=self.Ns,patch_centers=self.data1_.patch_centers)

        else :
            self.corr = 'auto'
            self.data2_ = self.data1_

        ra_r =  qwargs['RA_r']
        dec_r =  qwargs['DEC_r']
        w_r =  qwargs['W_r']
        z_r =  qwargs['Z_r']        
        dc_r = self.cosmo.comoving_distance(z_r).value

        self.rand1_ = treecorr.Catalog(ra=ra_r,dec=dec_r,r=dc_r,w=w_r,ra_units='degree',\
            dec_units='degree',npatch=self.Ns,patch_centers=self.data1_.patch_centers,is_rand=1)

        
        if 'RA_r2' in qwargs:
            
            ra_r2 =  qwargs['RA_r2']
            dec_r2 =  qwargs['DEC_r2']
            w_r2 =  qwargs['W_r2']
            z_r2 =  qwargs['Z_r2']        
            dc_r2 = self.cosmo.comoving_distance(z_r2).value

            self.rand2_ = treecorr.Catalog(ra=ra_r2,dec=dec_r2,r=dc_r2,w=w_r2,ra_units='degree',\
                dec_units='degree',npatch=self.Ns,patch_centers=self.data1_.patch_centers,is_rand=1)
            
        else:
            self.rand2_ = self.rand1_
        
        
        
