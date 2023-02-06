import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pyccl.background as pb
import fastpt.HT as HT
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import splrep,splev,interp1d
from scipy.integrate import quad

"""CLASS to model wg+ and wgg given ccl(fast-pt) predictions"""

def L2(x):
    return 1./2*(3*x**2-1.)

def L4(x):
    return 1./8*(35.*x**4 - 30*x**2 + 3.)

def derivative(func, x, h):
	return (-func(x+2*h)+8*func(x+h)-8*func(x-h)+func(x-2*h))/(12*h)


class twopt_model:

    def __init__(self):
        self.ptc = None
        self.unique_z = True
        #self.models_list =  {'WGG':self.projected_IA_power,'WGP':self.projected_clustering_power,'xil':None,'DSigma':None}

    def set_cosmology(self,Om0,Omb,As,ns,h,zeff):

        self.Om0 = Om0
        self.Omb = Omb
        self.As = As
        self.ns = ns
        self.h = h
        self.zeff = zeff
        self.Omc = Om0 - Omb
        self.cosmo = ccl.Cosmology(Omega_c=self.Omc, Omega_b=self.Omb, h=self.h, A_s=self.As, n_s=self.ns)
        self.f = pb.growth_rate(self.cosmo,1./(1+self.zeff))
        
    def set_nz(self,zedge,z,dn):
        self.unique_z = False
        self.z_sample = zedge
        self.sf_sample = 1./(1+zedge)
        self.f = np.array([pb.growth_rate(self.cosmo,a) for a in self.sf_sample])

        z_min = zedge[0]
        z_max = zedge[len(zedge)-1]
        self.dz = zedge[1] - zedge[0]
        z_chi = np.linspace(0,10,10000)
        sf = 1./(1+z_chi)
        
        """Get by linear extrapolation n(z) on the boundaries"""
        spline = splrep(z,dn, k=1, s=0)
        dn = splev(zedge, spline)
        dn = dn/np.sum(dn)
        
        dist = ccl.comoving_radial_distance(self.cosmo, sf)
        spline_chi = interp1d(z_chi,dist,kind="cubic",fill_value = (0,0),bounds_error=False)
        chi = spline_chi(zedge)
        dchi = derivative(spline_chi,zedge,0.001)
        
        Wz = dn*dn/(chi**2*dchi)
        spline_Wz = interp1d(zedge,Wz,kind="cubic",fill_value = (0,0),bounds_error=False)
        norm = quad(spline_Wz,z_min,z_max)[0]
        self.Wz = Wz/norm
        
    def set_fastpt(self):
        nk = 1024
        kmin = -5
        kmax = 2
        self.ks = np.logspace(kmin,kmax,nk)
        self.k = self.ks/self.h 
        self.z = np.linspace(self.zeff-0.01,self.zeff+0.01,200)
        self.gz = ccl.growth_factor(self.cosmo, 1./(1+self.z))

        self.ptc = pt.PTCalculator(with_NC=True, with_IA=True,
                      log10k_min=-5, log10k_max=2, nk_per_decade=20)

        
        """compute projected clustering in redshift space"""
    def compute_projectedR(self,rp,pk,b1,f):
        beta = f/b1
        P0 = (1+2./3.*beta+1./5*beta**2)*pk
        P2 = (4./3.*beta+4./7*beta**2)*pk
        P4 = (8./35*beta**2)*pk

        r,xi0 =  HT.k_to_r(self.k,P0,alpha_k=1.5, beta_r=-1.5,mu=0.5)
        r,xi2 = HT.k_to_r(self.k,P2,1.5,-1.5,mu=2.5)
        r,xi4 = HT.k_to_r(self.k,P4,1.5,-1.5,mu=4.5)
        xi0s = CS(r,xi0)
        xi2s = CS(r,xi2)
        xi4s = CS(r,xi4)

        dpi = 0.1
        npt = int(2.*self.pimax/dpi) + 1
        pi = np.linspace(-self.pimax,self.pimax,npt)
        s = np.zeros(np.outer(rp,pi).shape)
        xi_rppi = np.zeros(np.outer(rp,pi).shape)
        x,y=np.indices(xi_rppi.shape)
        s[x,y]= np.sqrt(rp[x]**2+pi[y]**2)
        mu = pi/s
        xi_rppi[x,y]=xi0s(s)-xi2s(s)*L2(mu)+xi4s(s)*L4(mu)
        wgg_ = np.sum(xi_rppi*dpi,axis=1)
        
        return wgg_
        
    def projected_IA_power(self,rp,params,config,do_rsd=True):
        if self.ptc is None:
            self.set_fastpt()

        b1,b2,bs,a1,a2,ad = params
        c1,cd,c2 = pt.translate_IA_norm(self.cosmo, self.z, a1=a1, a1delta=ad, a2=a2, Om_m2_for_c2 = False)

        ptt_g = pt.PTNumberCountsTracer(b1=b1, b2=b2, bs=bs)

        if config == 'TATT':
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(self.z,c1), c2=(self.z,c2), cdelta=(self.z,cd)) 
            pk_gi= pt.get_pt_pk2d(self.cosmo, ptt_g, tracer2=ptt_i, ptc=self.ptc)
        else:
            ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(self.z,c1))
            pk_gi = pt.get_pt_pk2d(self.cosmo, ptt_g, tracer2=ptt_i, ptc=self.ptc)

        if self.unique_z == True:
            p_gI = self.h**3*pk_gi.eval(self.ks,1./(1+self.zeff),self.cosmo)
            
            if do_rsd is True:
                beta = self.f/b1
                P0 = (1+1./3.*beta)*p_gI
                P2 = (2./3.*beta)*p_gI

                r,xi0 =  HT.k_to_r(self.k,P0,alpha_k=1., beta_r=-1.,mu=0.5,pf=-1./(2*np.pi))
                r,xi2 = HT.k_to_r(self.k,P2,alpha_k=1., beta_r=-1.,mu=2.5,pf=-1./(2*np.pi))

                xi0s = CS(r,xi0)
                xi2s = CS(r,xi2)

                pi = np.linspace(-self.pimax,self.pimax,2*self.pimax+1)
                s = np.zeros(np.outer(rp,pi).shape)
                xi_rppi = np.zeros(np.outer(rp,pi).shape)
                x,y=np.indices(xi_rppi.shape)
                s[x,y]= np.sqrt(rp[x]**2+pi[y]**2)
                mu = pi/s
                xi_rppi[x,y]=xi0s(s)#-xi2s(s)*L2(mu)
                wgp_ = np.sum(xi_rppi,axis=1)

            else:
                r,wgp=HT.k_to_r(self.k,p_gI,1.,-1.,2., -1./(2*np.pi))
                spline_wgp = CS(r,wgp)
                wgp_ = spline_wgp(rp)
            
        else:
            p_gI = np.array([self.h**3*pk_gi.eval(self.ks,a,self.cosmo) for a in self.sf_sample])
            if do_rsd is True:
                print("do something bro please")
            else:
                r,_ = HT.k_to_r(self.k,p_gI[0],1.,-1.,2., -1./(2*np.pi))
                wgp = [HT.k_to_r(self.k,Pk,1.,-1.,2., -1./(2*np.pi))[1] for Pk in p_gI]
                wgp = np.sum(wgp*self.Wz[:,None],axis=0)*self.dz
                spline_wgp = CS(r,wgp)
                wgp_ = spline_wgp(rp)
            
        return wgp_


    def projected_clustering_power(self,rp,params,config,do_rsd=True):
        if self.ptc is None:
            self.set_fastpt()

        b1,b2,bs= params[0:3]
        ptt_g = pt.PTNumberCountsTracer(b1=b1, b2=b2, bs=bs)
        pk_gg = pt.get_pt_pk2d(self.cosmo, ptt_g, ptc=self.ptc)
        
        if self.unique_z == True:
            Pkgg = self.h**3*pk_gg.eval(self.ks,1./(1+self.zeff),self.cosmo)
            
            if do_rsd is True:
                wgg_ = self.compute_projectedR(rp,Pkgg,b1,self.f)
            else :
                r,wgg=HT.k_to_r(self.k,Pkgg,1.,-1.,0., 1./(2*np.pi))
                spline_wgg = CS(r,wgg)
                wgg_ = spline_wgg(rp)
        else:
            Pkgg = np.array([self.h**3*pk_gg.eval(self.ks,a,self.cosmo) for a in self.sf_sample])
            if do_rsd is True:
                wgg = [self.compute_projectedR(rp,Pk,b1,growth) for Pk,growth in zip(Pkgg,self.f)]
                wgg_ = np.sum(wgg*self.Wz[:,None],axis=0)*self.dz
            else:
                r,_ = HT.k_to_r(self.k,Pkgg[0],1.,-1.,0., 1./(2*np.pi))
                wgg = [HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(2*np.pi))[1] for Pk in Pkgg]
                wgg = np.sum(wgg*self.Wz[:,None],axis=0)*self.dz
                spline_wgg = CS(r,wgg)
                wgg_ = spline_wgg(rp)
        return wgg_

    def projected_power(self,rp,params,config,rp2=None):
        wgg = self.projected_clustering_power(rp,params,config)
        if rp2 is None:
            wgp = self.projected_IA_power(rp,params,config)
        else:
            wgp = self.projected_IA_power(rp2,params,config)
        xi = np.hstack([wgg,wgp])
        return xi

  
            