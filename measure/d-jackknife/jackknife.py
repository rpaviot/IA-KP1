from corrd import *
from compute_pairs import compute_pairs


class jackknife():

    def __init__(self,**qwargs):
        self.define_matrice()
        self.set_engine()
        self.pairs_counter = compute_pairs(self.bins1,self.bins2,self.bin_slop)
        #self.compute_all_pairs()
    
    def define_matrice(self):
        if self.twoD is True:
            self.NN_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1,len(self.bins2) - 1))
            self.NR_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1,len(self.bins2) - 1))
            self.RR_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1,len(self.bins2) - 1))
            self.xi = np.zeros((self.comb,len(self.bins1) - 1,len(self.bins2) - 1))
        else:
            self.NN_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1))
            self.NR_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1))
            self.RR_ = np.zeros((self.npairs,self.npairs,len(self.bins1) - 1))
            self.xi = np.zeros((self.comb,len(self.bins1) - 1))

                
    def set_engine(self):
        if self.computation =='WGG':
            self.pipeline = 'pycorr'
        elif self.computation == 'WGP':
            self.pipeline = 'treecorr'
            
    def compute_all_pairs(self):
        self.compute_clustering_pairs()
        if self.pipeline == 'treecorr':
            self.compute_ng_pairs()

    def compute_clustering_pairs(self):
        self.compute_auto_clustering_pairs()
        self.compute_cross_clustering_pairs()

    def compute_ng_pairs(self):
        self.compute_auto_ng_pairs()
        self.compute_cross_ng_pairs()

    def compute_auto_clustering_pairs(self,calc_RR=True):

        for i in range(0,len(self.upatches)):
            patchi = self.upatches[i]
            data1i = self.data1_.patches[patchi]
            rand1i = self.rand1_.patches[patchi]
            rand2i = self.rand2_.patches[patchi]

            if self.pipeline =="pycorr":
                DD = self.pairs_counter.compute_clustering(data1i)
                DR = self.pairs_counter.compute_clustering(data1i,rand1i)
                RR = self.pairs_counter.compute_clustering(rand1i)
                self.NN_[i][i]=DD
                self.NR_[i][i]=DR
                self.RR_[i][i]=RR
            else:
                patchi = self.upatches[i]
                rand1i = self.rand1_.patches[patchi]
                RR = self.pairs_counter.compute_clustering(rand1i,rand2i)
                self.RR_[i][i]=RR

    def compute_cross_clustering_pairs(self,calc_RR=True):

        for patches in self.jackpairs:
            patchi,patchj = patches

            data1i = self.data1_.patches[patchi]
            data1j = self.data1_.patches[patchj]
            
            rand1i =  self.rand1_.patches[patchi]
            rand1j =  self.rand1_.patches[patchj]
            rand2j = self.rand2_.patches[patchj]

            if self.pipeline =="pycorr":
                self.NR_[patchi][patchj] = self.pairs_counter.compute_clustering(data1i,rand1j)
                self.RR_[patchi][patchj] = self.pairs_counter.compute_clustering(rand1i,rand1j)
                self.NN_[patchi][patchj] = self.pairs_counter.compute_clustering(data1i,data1j)                    
            else:
                self.RR_[patchi][patchj] = self.pairs_counter.compute_clustering(rand1i,rand2j)
    

    def compute_auto_ng_pairs(self):
        for i in range(0,len(self.upatches)):
            patchi = self.upatches[i]
            data1i = self.data1_.patches[patchi]
            data2i = self.data2_.patches[patchi]
            rand1i = self.rand1_.patches[patchi]
            NN = self.pairs_counter.compute_IA(data1i,data2i)
            NR = self.pairs_counter.compute_IA(rand1i,data2i)
            self.NN_[i][i]=NN
            self.NR_[i][i]=NR

    def compute_cross_ng_pairs(self):
        for patches in self.jackpairs:
            patchi,patchj = patches

            data1i = self.data1_.patches[patchi]
            data1j = self.data1_.patches[patchj]
            
            data2i = self.data2_.patches[patchi]
            data2j = self.data2_.patches[patchj]
            
            rand1i =  self.rand1_.patches[patchi]
            rand1j =  self.rand1_.patches[patchj]
            
            NN = self.pairs_counter.compute_IA(data1i,data2j)                                
            NR = self.pairs_counter.compute_IA(rand1i,data2j)  

            self.NN_[patchi][patchj]=NN
            self.NR_[patchi][patchj]=NR

    def combine_all_clustering(self):
        for i in range(0,len(self.array_comb)):
            catg = []
            catr = []
            indices = self.array_comb[i]
            comb = [",".join(map(str, comb)) for comb in combinations(indices, 2)]        
            array_comb = np.array(list(map(eval,comb)))

            DDpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            DRpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            RRpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))

            DDpairs += np.sum([self.NN_[x,x] for x in indices],axis=0)
            DDpairs += np.sum([self.NN_[x,y]+self.NN_[y,x] for x,y in array_comb],axis=0)
            DRpairs += np.sum([self.NR_[x,x] for x in indices],axis=0)
            DRpairs += np.sum([self.NR_[x,y]+self.NR_[y,x] for x,y in array_comb],axis=0)
            RRpairs += np.sum([self.RR_[x,x] for x in indices],axis=0)
            RRpairs += np.sum([self.RR_[x,y]+self.RR_[y,x] for x,y in array_comb],axis=0)
            
            catg = [np.concatenate([catg,self.data1_.patches[x].w]) for x in indices.astype('int')]
            catg = np.hstack(catg)
            catr = [np.concatenate([catr,self.rand1_.patches[x].w]) for x in indices.astype('int')]
            catr = np.hstack(catr)
                
            normDD = np.sum(catg)**2 - np.sum(catg**2)
            normDR = np.sum(catg)*np.sum(catr)
            normRR = np.sum(catr)**2 - np.sum(catr**2)
            xi = (DDpairs/RRpairs)*(normRR/normDD) - 2*(DRpairs/RRpairs)*(normRR/normDR) + 1.
            self.xi[i] = xi


    def combine_all_IA(self):
        for i in range(0,len(self.array_comb)):  
            catg = []
            catr = []
            indices = self.array_comb[i]
            comb = [",".join(map(str, comb)) for comb in combinations(indices, 2)]        
            array_comb = np.array(list(map(eval,comb)))

            NGxi = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            RGxi = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            NGpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            RGpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
            RRpairs = np.zeros((len(self.bins1) - 1,len(self.bins2) - 1))
        
            NGxi += np.sum([self.NN_[x,x] for x in indices],axis=0)
            NGxi += np.sum([self.NN_[x,y] + self.NN_[y,x] for x,y in array_comb],axis=0)
            RGxi += np.sum([self.NR_[x,x] for x in indices],axis=0)
            RGxi += np.sum([self.NR_[x,y] + self.NR_[y,x] for x,y in array_comb],axis=0)
            RRpairs += np.sum([self.RR_[x,x] for x in indices],axis=0)
            RRpairs += np.sum([self.RR_[x,y]+self.RR_[y,x] for x,y in array_comb],axis=0)
            
            catg = [np.concatenate([catg,self.data1_.patches[x].w]) for x in indices.astype('int')]
            catg = np.hstack(catg)

            catr = [np.concatenate([catr,self.rand1_.patches[x].w]) for x in indices.astype('int')]
            catr = np.hstack(catr)

            if self.corr == 'auto':  
                normDD = np.sum(catg)**2 - np.sum(catg**2)
                normDR = np.sum(catg)*np.sum(catr)
                normRR = np.sum(catr)**2 - np.sum(catr**2)
            elif self.corr == 'cross':
                catg2 = [np.concatenate([catg2,self.data2_.patches[x].w]) for x in indices.astype('int')]
                catg2 = np.hstack(catg2)
                catr2 = [np.concatenate([catr2,self.rand2_.patches[x].w]) for x in indices.astype('int')]
                catr2 = np.hstack(catr2)        
                normDD = np.sum(catg)*np.sum(catg2)
                normDR = np.sum(catg)*np.sum(catr)
                normRR = np.sum(catr)*np.sum(catr2)
                
            xi = NGxi/RRpairs*(normRR/normDD) - RGxi/RRpairs*(normRR/normDR) 

            self.xi[i] = xi


    def get_measurements(self):
        self.compute_all_pairs()
        if self.pipeline == 'pycorr':
            self.combine_all_clustering()
        else:
            self.combine_all_IA()
        if self.twoD is True:
            self.xi = np.sum(self.xi*self.du,axis=2)

        self.xi_mean = np.mean(self.xi,axis=0)
        xi = self.xi - self.xi_mean
        Cov = (self.Ns - self.Nd)/(self.Nd*self.comb)*np.dot(xi.T,xi)
        rp = self.get_meanr()
        return rp,self.xi_mean,Cov


    def get_meanr(self):
        if self.corr == 'pycorr':
            rp = self.pairs_counter.compute_clustering_rp(self.data1_,self.rand1_)
        else:
            rp = self.pairs_counter.compute_IA_rp(self.rand1_,self.data2_)
        return rp
    
    def set_random_pairs(self,corr):
        self.RR_ = corr.RR_

    def combine_measurements(self,corr):
        self.xi_tot = np.column_stack((self.xi,corr.xi))
        self.xi_mean = np.mean(self.xi_tot,axis=0)
        xi = self.xi_tot - self.xi_mean
        Cov = (self.Ns - self.Nd)/(self.Nd*self.comb)*np.dot(xi.T,xi)
        return Cov