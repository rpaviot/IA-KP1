from pycorr.corrfunc import CorrfuncTwoPointCounter
from corrd import *
import threading
ncores = threading.active_count()


class compute_pairs():
    
    def __init__(self,bins1,bins2=None,bin_slop = 0.):
        self.bins1 = bins1
        self.bins2 = bins2
        self.min_sep = np.min(self.bins1)
        self.max_sep = np.max(self.bins1)
        self.bin_slop = bin_slop

    def compute_clustering(self,catalog1,catalog2=None):

        """treecorr automatically converted (ra,dec) in radian, convert it back to degree"""

        ra = np.rad2deg(catalog1.ra)
        dec = np.rad2deg(catalog1.dec)
        r = catalog1.r
        w = catalog1.w
        pos = np.column_stack([ra,dec,r]).T

        if catalog2 is not None:
            ra2 = np.rad2deg(catalog2.ra)
            dec2 = np.rad2deg(catalog2.dec)
            r2 = catalog2.r
            w2 = catalog2.w
            pos2 = np.column_stack([ra2,dec2,r2]).T
            DD =self.compute_pairs(pos,w,pos2=pos2,weights2=w2)
        else :
            DD = self.compute_pairs(pos,w)
        return DD

    def compute_pairs(self,pos,weight,pos2=None,weights2=None):
            pycorr = CorrfuncTwoPointCounter('rppi', [self.bins1,self.bins2], positions1=pos,positions2=pos2, \
                                     position_type='rdd', weights1=weight,weights2=weights2 ,nthreads=ncores,compute_sepsavg=False)
            return pycorr.wcounts

    def compute_sepavg(self,pos,weight,pos2,weights2):
            pycorr = CorrfuncTwoPointCounter('rppi', [self.bins1,self.bins2], positions1=pos,positions2=pos2, \
                                     position_type='rdd', weights1=weight,weights2=weights2 ,nthreads=ncores,compute_sepsavg=True)
            return pycorr.sepavg()
        
        
    def compute_IA(self,catalog1,catalog2):
        DS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        wDS = np.zeros((len(self.bins1)-1,len(self.bins2)-1))
        for i in range(0,len(self.bins2)-1):
            pi_min = self.bins2[i]
            pi_max = self.bins2[i+1]
            ng = treecorr.NGCorrelation(bin_type='Log',nbins=len(self.bins1)-1,min_sep=self.min_sep,max_sep=self.max_sep,\
                min_rpar=pi_min,max_rpar=pi_max,metric="Rperp",bin_slop=self.bin_slop)
            ng.process(catalog1,catalog2)
            DS[:,i] = ng.xi*ng.weight

        return DS

    def compute_IA_sepavg(self,catalog1,catalog2):
        ng = treecorr.NGCorrelation(bin_type='Log',nbins=len(self.bins1)-1,min_sep=self.min_sep,max_sep=self.max_sep,metric="Rperp")
        ng.process(catalog1,catalog2)
        return ng.meanr

        
    def compute_clustering_rp(self,catalog1,catalog2):
        ra = np.rad2deg(catalog1.ra)
        dec = np.rad2deg(catalog1.dec)
        r = catalog1.r
        w = catalog1.w
        pos = np.column_stack([ra,dec,r]).T

        ra2 = np.rad2deg(catalog2.ra)
        dec2 = np.rad2deg(catalog2.dec)
        r2 = catalog2.r
        w2 = catalog2.w
        pos2 = np.column_stack([ra2,dec2,r2]).T
        rp=self.compute_sepavg(pos,w,pos2=pos2,weights2=w2)       
        return rp 

    def compute_IA_rp(self,catalog1,catalog2):
        rp = self.compute_IA_sepavg(catalog1,catalog2)
        return rp