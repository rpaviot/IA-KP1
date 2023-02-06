import numpy as np
import emcee
import iminuit
from iminuit import Minuit
from models import twopt_model

class fit(twopt_model):
    def __init__(self):
        super().__init__()
        self.Ktapv = np.vectorize(self.Ktap)
    
    def set_cosmo(self,*args):
        super().set_cosmology(*args)

    def set_data(self,r1,data1,Cov,pimax=100,data2=None,r2=None):
        self.pimax = pimax
        if data2 is None:
            self.fit='simple'
        else:
            self.fit='combined'
        self.r1 = r1
        if r2 is None:
            self.r2 = r1
        else:
            self.r2 = r2
        self.data1 = data1
        self.data2 = data2
        rr = np.hstack([self.r1,self.r2])
        rx,ry = np.meshgrid(rr,rr)
        dist = np.abs(ry - rx)
        self.cov = self.Ktapv(dist,15)*Cov

    def set_cut(self,rmin,rmax,rmin2=None,rmax2=None):
        if self.fit=='simple':
            cond = np.where((self.r1 > rmin)&(self.r2 < rmax))
            self.rfit = self.r1[cond]
            self.data = self.data1[cond]
            r1,r2 = np.meshgrid(self.r1,self.r1)
            cond = (r1 > rmin)&(r1 <rmax) & (r2 > rmin)&(r2 <rmax)
            self.Cov = self.cov[cond]
            size = int(np.sqrt(len(self.Cov)))
            self.Cov = self.Cov.reshape((size,size))
            self.invcov = np.linalg.inv(self.Cov)
        else:
            if rmin2 is None:
                rmin2 = rmin
                rmax2 = rmax
            cond1 = np.where((self.r1 > rmin)&(self.r2 < rmax))
            cond2 = np.where((self.r2 > rmin2)&(self.r2<rmax2))
            self.data1 = self.data1[cond1]
            self.data2 = self.data2[cond2]
            self.data = np.hstack([self.data1,self.data2])
            self.rfit = self.r1[cond1]
            self.rfit2  = self.r2[cond2]
            rr = np.hstack([self.r1,self.r2])
            r1,r2 = np.meshgrid(rr,rr)
            cond = (r1 > rmin)&(r1 <rmax) & (r2 > rmin2)&(r2 <rmax2)
            self.Cov = self.cov[cond]
            size = int(np.sqrt(len(self.Cov)))
            self.Cov = self.Cov.reshape((size,size))
            self.invcov = np.linalg.inv(self.Cov)
            
    def Ktap(self,x,Tp):
        if x < Tp:
            f = (1 - x/Tp)**4*(4*x/Tp + 1)
        else:
            f = 0
        return f   

    def set_model(self,model,config=None):
        self.model = model
        self.config = config

        if 'WGP' in self.model:
            self.predictions = super().projected_IA_power

        if 'WGG' and 'WGP' in self.model:
            self.predictions = super().projected_power

    def get_likelihood(self,p):
        self.xi = self.predictions(self.rfit,p,self.config)
        diff = self.xi - self.data
        chi = np.dot(diff,np.dot(self.invcov,diff))
        return 2*chi

    def minimize(self):
        if self.predictions in [super().projected_power,super().projected_IA_power]:
            def lnprob(b1,b2,bs,a1,a2,ad):
                p = np.array([b1,b2,bs,a1,a2,ad])
                lnprob = self.get_likelihood(p)
                return lnprob

            m = Minuit(lnprob,b1=1.5,b2=0.0,bs=0.0,a1=1.0,a2=0.0,ad=0.0)
            m.errordef = Minuit.LIKELIHOOD
            m.limits = [(0.1,4.0),(-2,2),(-2,2),(0,10),(-2,2),(-2,2)]

            if self.config == 'NLA':
                m.fixed["b2"] = True
                m.fixed["bs"] = True
                m.fixed["a2"] = True
                m.fixed["ad"] = True

            m.migrad()
            b1 = m.values['b1']
            b2 = m.values['b2']
            bs = m.values['bs']
            a1 = m.values['a1']
            a2 = m.values['a2']
            ad = m.values['ad']

            self.pmax = np.hstack([b1,b2,bs,a1,a2,ad])
            self.chi2 = m.fval