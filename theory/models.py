import numpy as np
import pyccl as ccl
import pyccl.nl_pt as pt
import pyccl.background as pb
import fastpt.HT as HT
from scipy.interpolate import CubicSpline as CS
from scipy.interpolate import splrep,splev,interp1d
from scipy.integrate import quad,simps
from numpy import arctan as atan
# from numba import vectorize,njit,prange,jit
from scipy import integrate
from scipy.stats import binned_statistic

"""CLASS to model wgg,wgp,wpp and W_+ given ccl(fast-pt) predictions"""

def legendre2(x):
    return (3*x**2-1.)/2

def L4(x):
    return 1./8*(35.*x**4 - 30*x**2 + 3.)

def derivative(func, x, h):
	return (-func(x+2*h)+8*func(x+h)-8*func(x-h)+func(x-2*h))/(12*h)

### 4 pi G / c**2 in Msun/pc###
const_dsigma = 6.013530900921833e-13

class twopt_model:

    def __init__(self,cosmology,config,computation=[],do_rsd=True,pimax=100,dpi=10,do_lensing=False,include_B=True,FoG=False):
        
        """Initialisation of the models
        
        Inputs : 
        -cosmology : specifies the fidicual cosmology for the modelling. A list in format of [Omc,Omb,Omnu,As,ns,h,zeff]
        - config : either NLA or TATT model
        -computation : a list that specifies which statitics will be computed by the models, in [wgg,wgp,wpp,W_+]
        -do_rsd : either to apply redshift-space distortions to wgg. Default to True.
        -pimax : When doing RSD, needs to provive a value of pimax (the one used to do the measurements.)
        -do_lensing : either to apply lensing contributions to wgp and wgp. Default to False (cannot apply lensing without giving some n(z)s)
        -include_B: either to include B-modes for wpp and W_+ estimation. Default to True

        """

        self.unique_z = True
        self.FoG = FoG
        self.Omc = cosmology[0]
        self.Omb = cosmology[1]
        self.mnu = cosmology[2]
        self.As = cosmology[3]
        self.ns = cosmology[4]
        self.h = cosmology[5]
        self.zeff = cosmology[6]
        self.dpi = dpi
        
        self.cosmo = ccl.Cosmology(Omega_c=self.Omc, Omega_b=self.Omb,m_nu=self.mnu, h=self.h, A_s=self.As, n_s=self.ns)
        
        self.f = pb.growth_rate(self.cosmo,1./(1+self.zeff))
        self.zpt = np.linspace(self.zeff-0.01,self.zeff+0.01,100)
        self.apt = np.sort(1./(1+self.zpt))

        self.config=config
        self.computation=computation
        self.do_rsd = do_rsd
        self.pimax = pimax
        self.do_lensing = do_lensing
        self.include_B = include_B
        self.models_list =  {'WGG':self.projected_clustering_power,'WGP':self.projected_gI_power,\
                            'WPP':self.projected_II_power, 'W_+':self.projected_comb_II_power}
        
        npt = int(self.pimax/self.dpi) + 1
        self.pi_data = np.linspace(0,self.pimax,npt)

        
        self.computation = computation
        self.sigma_inv_ = np.vectorize(self.sigma_inv)
        self.set_fastpt()
        
    def set_fastpt(self):
        nk = 512
        kmin = -5
        kmax = 2
        self.ks = np.logspace(kmin,kmax,nk)
        self.k = self.ks/self.h 
            
        self.ptc = pt.PTCalculator(with_NC=True, with_IA=True,
                      log10k_min=kmin, log10k_max=kmax, nk_per_decade=20)
        


    def set_nz(self,zedge,z,dn1,dn2):
            
        """ This function calculate the window functions, and variables for lensing contribution"""
        
        self.unique_z = False
        self.z_hist = zedge
        self.dz = zedge[1] - zedge[0]

        z_min = zedge[0]
        z_max = zedge[len(zedge)-1]
        self.zmin = z_min
        self.zmax = z_max
        
        """Get by linear extrapolation n(z) on the boundaries"""
        spline = splrep(z,dn1, k=3, s=0)
        dn1 = splev(zedge, spline,ext=1)
        dn1_ = dn1/np.trapz(dn1,zedge)
        
        spline = splrep(z,dn2, k=3, s=0)
        dn2 = splev(zedge, spline,ext=1)
        dn2_ = dn2/np.trapz(dn2,zedge)
        
        self.spline_nz1 = CS(self.z_hist,dn1_)
        self.spline_nz2 = CS(self.z_hist,dn2_)

        self.dz2 = 0.005
        npt = int((self.zmax - self.zmin)/self.dz2 + 1)
        self.z_sample = np.linspace(self.zmin,self.zmax,npt)
        self.zpt = np.linspace(self.zmin-self.dz2,self.zmax+self.dz2,npt+2)
        self.apt = np.sort(1./(1+self.zpt))
        # self.dz2 = np.diff(self.z_sample)[0]
        self.sf = np.sort(1./(1+self.z_sample))
        self.f = np.array([pb.growth_rate(self.cosmo,a) for a in self.sf])
        
        z_chi = np.linspace(0,10,10000)
        sf = 1./(1+z_chi)
        dist = ccl.comoving_radial_distance(self.cosmo, sf)

        spline_chi = interp1d(z_chi,dist,kind="cubic",fill_value = (0,0),bounds_error=False)
        chi = spline_chi(self.z_sample)
        dchi = derivative(spline_chi,self.z_sample,0.001)
        
        Wz = self.spline_nz1(self.z_sample)*self.spline_nz2(self.z_sample)/(chi**2*dchi)
        spline_Wz = interp1d(self.z_sample,Wz,kind="cubic",fill_value = (0,0),bounds_error=False)
        norm = quad(spline_Wz,z_min,z_max)[0]
        self.Wz = Wz[::-1] /norm## to match scale factor sorting.

        Wz = self.spline_nz1(self.z_sample)*self.spline_nz1(self.z_sample)/(chi**2*dchi)
        spline_Wz = interp1d(self.z_sample,Wz,kind="cubic",fill_value = (0,0),bounds_error=False)
        norm = quad(spline_Wz,z_min,z_max)[0]
        self.Wz_clustering = Wz[::-1]/norm
        
        Wz = self.spline_nz2(self.z_sample)*self.spline_nz2(self.z_sample)/(chi**2*dchi)
        spline_Wz = interp1d(self.z_sample,Wz,kind="cubic",fill_value = (0,0),bounds_error=False)
        norm = quad(spline_Wz,z_min,z_max)[0]
        self.Wz_source = Wz[::-1]/norm
                
        
    def set_pks2D(self,params):
        """This function computes once all 1-loop quantities given a set of params"""
        b1,b2,bs,b3nl,a1,a2,bTA = params

        if self.config == 'TATT':
            c1,cd,c2 = pt.translate_IA_norm(self.cosmo, self.zpt, a1=a1, a1delta=a1*bTA, a2=a2, Om_m2_for_c2 = False)
        else :
            c1,cd,c2 = pt.translate_IA_norm(self.cosmo, self.zpt, a1=a1, a1delta=0., a2=0., Om_m2_for_c2 = False)


        #ptt_m = pt.PTMatterTracer()
        ptt_g = pt.PTNumberCountsTracer(b1=b1, b2=b2, bs=bs,b3nl=b3nl)
        ptt_i = pt.PTIntrinsicAlignmentTracer(c1=(self.zpt,c1), c2=(self.zpt,c2), cdelta=(self.zpt,cd)) 
        
        # self.pk_mi = pt.get_pt_pk2d(self.cosmo, ptt_m, tracer2=ptt_i, ptc=self.ptc)
        self.pk_gi = pt.get_pt_pk2d(self.cosmo, ptt_g, tracer2=ptt_i, ptc=self.ptc,a_arr=self.apt)
        # self.pk_mm = pt.get_pt_pk2d(self.cosmo, ptt_m, ptc=self.ptc)
        self.pk_gg = pt.get_pt_pk2d(self.cosmo, ptt_g, ptc=self.ptc,a_arr=self.apt)
        #self.pk_gm = pt.get_pt_pk2d(self.cosmo, ptt_g, tracer2=ptt_m, ptc=self.ptc,a_arr=self.zpt)
        self.pk_ii_ee, self.pk_ii_bb = pt.get_pt_pk2d(self.cosmo, ptt_i, ptc=self.ptc,return_ia_ee_and_bb=True,a_arr=self.apt)


    def weighted_nz_signal(self,rp,pk2d,Wz,correlation,pk2d_=None):
            
        pk_ = np.array([self.h**3*pk2d.eval(self.ks,a,self.cosmo) for a in self.sf])
        r,_ = HT.k_to_r(self.k,pk_[0],1.,-1.,2., -1./(2*np.pi))
        
        if correlation == 'wgg' and self.do_rsd==True:
            fftlog = np.array([self.compute_projectedR(rp,Pk,self.params[0],growth) for Pk,growth in zip(pk_,self.f)])
            
        elif correlation == 'wgg' and self.do_rsd==False :
            fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(2*np.pi))[1] for Pk in pk_])
            
        elif correlation == 'wgp':
            fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,2., -1./(2*np.pi))[1] for Pk in pk_])
            
        elif correlation == 'wpp':
            if self.include_B == True:
                pk2_ = np.array([self.h**3*pk2d_.eval(self.ks,a,self.cosmo) for a in self.sf])
                pki = pk_ + pk2_
                pkj = pk_ - pk2_
                fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(4*np.pi))[1] for Pk in pki]) + np.array([HT.k_to_r(self.k,Pk,1.,-1.,4., 1./(4*np.pi))[1] for Pk in pkj])
            else:
                fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(4*np.pi))[1] for Pk in pk_]) + np.array([HT.k_to_r(self.k,Pk,1.,-1.,4., 1./(4*np.pi))[1] for Pk in pk_])
               
        elif correlation == 'W_+':
            if self.include_B == True:
                pk2_ = np.array([self.h**3*pk2d_.eval(self.ks,a,self.cosmo) for a in self.sf])
                pki = pk_ + pk2_
                fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(2*np.pi))[1] for Pk in pki])
            else:
                fftlog =  np.array([HT.k_to_r(self.k,Pk,1.,-1.,0., 1./(2*np.pi))[1] for Pk in pk_])
                
        
        if correlation == 'wgg' and self.do_rsd==True:
            xi_ = np.sum(fftlog*Wz[:,None],axis=0)*self.dz2
            return xi_
        else:
            xi_ = np.sum(fftlog*Wz[:,None],axis=0)*self.dz2
            spline_xi = CS(r,xi_)
            xi_ = spline_xi(rp)
            return xi_
        
    def projected_power(self,rp,params):
        
        """This is the callable function of the class thar return the desired quantities"""

        b1,b2,a1,a2,bTA=params
        
        if self.config == 'TATT':
            bs = -(4./7.)*(b1 - 1.)
            b3nl = b1 - 1.
            
        else :
            bs = 0.
            b3nl = 0.           
            # bs = -(4./7.)*(b1 - 1.)
            # b3nl = b1 - 1.
            
        params = [b1,b2,bs,b3nl,a1,a2,bTA]
        self.params = params
        self.set_pks2D(self.params)
        
        func = self.models_list[self.computation[0]]
        xi =  func(rp)

        if len(self.computation) > 1:
            for i in range(1,len(self.computation)):
                func = self.models_list[self.computation[i]]
                xi2 = func(rp)
                xi = np.column_stack([xi,xi2])                
        return xi.T
    
    
    def compute_binavg(self,rp,rpbins,stats):
        intg,_,_=binned_statistic(rp, stats, statistic='mean',
                       bins=rpbins)
        return intg
        
    def projected_power_binavg(self,rpbins,params):
        size = len(rpbins)
        size_new = size*30
        rp = np.geomspace(rpbins[0],rpbins[-1],size_new)
        
        """This is the callable function of the class thar return the desired quantities"""

        b1,b2,a1,a2,bTA=params
        
        if self.config == 'TATT':
            bs = -(4./7.)*(b1 - 1.)
            b3nl = b1 - 1.
            
        else :
            bs = 0.
            b3nl = 0.           
            # bs = -(4./7.)*(b1 - 1.)
            # b3nl = b1 - 1.
            
        params = [b1,b2,bs,b3nl,a1,a2,bTA]
        self.params = params
        self.set_pks2D(self.params)
        
        func = self.models_list[self.computation[0]]
        xi =  func(rp)
        xi = self.compute_binavg(rp,rpbins,xi)
        if len(self.computation) > 1:
            for i in range(1,len(self.computation)):
                func = self.models_list[self.computation[i]]
                xi2 = func(rp)
                xi2 = self.compute_binavg(rp,rpbins,xi2)
                xi = np.column_stack([xi,xi2])                
        return xi.T
    
        
    def projected_gI_power(self,rp):
        
        """This function computes wgp"""

        if self.unique_z == True:
            p_gI = self.h**3*self.pk_gi.eval(self.ks,1./(1+self.zeff),self.cosmo)
            r,wgp=HT.k_to_r(self.k,p_gI,1.,-1.,2., -1./(2*np.pi))
            spline_wgp = CS(r,wgp)
            wgp_ = spline_wgp(rp)
        else:
            wgp_ = self.weighted_nz_signal(rp,self.pk_gi,self.Wz,'wgp')
                
        if self.do_lensing==True:
            gammat = self.get_lensing_wgp_contribution(rp)
            return wgp_ - gammat 
        else:
            return wgp_

        
    def projected_II_power(self,rp):
        
        """This function computes wpp"""
    
        if self.unique_z == True:
            p_II_e = self.h**3*self.pk_ii_ee.eval(self.ks,1./(1+self.zeff),self.cosmo)
            if self.include_B == True:
                p_II_B = self.h**3*self.pk_ii_bb.eval(self.ks,1./(1+self.zeff),self.cosmo)
                pki = p_II_e + p_II_B
                pkj = p_II_e - p_II_B
                r,wpp=HT.k_to_r(self.k,pki,1.,-1.,0., 1./(4*np.pi))
                r,wpp2=HT.k_to_r(self.k,pkj,1.,-1.,4., 1./(4*np.pi))
                wpp = wpp + wpp2
            else:
                r,wpp=HT.k_to_r(self.k,p_II_e,1.,-1.,0., 1./(4*np.pi))
                r,wpp2=HT.k_to_r(self.k,p_II_e,1.,-1.,4., 1./(4*np.pi))
                wpp = wpp + wpp2
                
            spline_wpp = CS(r,wpp)
            wpp_ = spline_wpp(rp)
        else:
            wpp_ = self.weighted_nz_signal(rp,self.pk_ii_ee,self.Wz_source,'wpp',pk2d_=self.pk_ii_bb)
                
        return wpp_
    
        
    def projected_comb_II_power(self,rp):
        
        """This function computes W_+"""
    
        if self.unique_z == True:
            p_II_e = self.h**3*self.pk_ii_ee.eval(self.ks,1./(1+self.zeff),self.cosmo)

            if self.include_B == True:
                p_II_B = self.h**3*self.pk_ii_bb.eval(self.ks,1./(1+self.zeff),self.cosmo)
                pki = p_II_e + p_II_B
                r,wp=HT.k_to_r(self.k,pki,1.,-1.,0., 1./(2*np.pi))
                #wp = 2*wp
            else:
                r,wp=HT.k_to_r(self.k,p_II_e,1.,-1.,0., 1./(2*np.pi))
                #wp = 2*wp
                
            spline_wp = CS(r,wp)
            wp_ = spline_wp(rp)

        else:
            wp_ = self.weighted_nz_signal(rp,self.pk_ii_ee,self.Wz_source,'W_+',pk2d_=self.pk_ii_bb)
        return wp_


    def projected_clustering_power(self,rp):
        
        """This function computes wgg"""
        
        if self.unique_z == True:
            Pkgg = self.h**3*self.pk_gg.eval(self.ks,1./(1+self.zeff),self.cosmo)
            if self.do_rsd is True:
                wgg_ = self.compute_projectedR(rp,Pkgg,self.params[0],self.f)
            else :
                r,wgg=HT.k_to_r(self.k,Pkgg,1.,-1.,0., 1./(2*np.pi))
                spline_wgg = CS(r,wgg)
                wgg_ = spline_wgg(rp)
                
        else:
            wgg_ = self.weighted_nz_signal(rp,self.pk_gg,self.Wz_clustering,'wgg')
            
        return wgg_
    

    def compute_projectedR(self,rp,pk,b1,f):
        
        """compute wgg(rp) in redshift space"""

        beta = f/b1
        
        if self.FoG is False:
            P0 = (1+2./3.*beta+1./5*beta**2)*pk
            P2 = (4./3.*beta+4./7*beta**2)*pk
            P4 = (8./35*beta**2)*pk
            
        elif self.FoG is True:
            P0 = (1./2)*pk*(self.K10 + 2*beta*self.K20 + beta**2*self.K30)
            P2 = (5./2)*pk*(2*beta*self.K22 + beta**2*self.K32)
            P4 = (9./2)*2*pk*beta**2*self.K34
            
            
        r,xi0 =  HT.k_to_r(self.k,P0,alpha_k=1.5, beta_r=-1.5,mu=0.5)
        r,xi2 = HT.k_to_r(self.k,P2,1.5,-1.5,mu=2.5)
        r,xi4 = HT.k_to_r(self.k,P4,1.5,-1.5,mu=4.5)
        xi0s = CS(r,xi0)
        xi2s = CS(r,xi2)
        xi4s = CS(r,xi4)
        
        dpi = 0.2
        npt = int(self.pimax/dpi) + 1
        pi = np.linspace(0,self.pimax,npt)
        s = np.zeros(np.outer(rp,pi).shape)
        xi_rppi = np.zeros(np.outer(rp,pi).shape)
        x,y=np.indices(xi_rppi.shape)
        s[x,y]= np.sqrt(rp[x]**2+pi[y]**2)
        mu = pi/s
        xi_rppi[x,y]=xi0s(s)-xi2s(s)*legendre2(mu)+xi4s(s)*L4(mu)
        # xi = self.compute_binavg(pi,self.pi_data,xi_rppi)
        wgg_ = 2*np.sum(xi_rppi*dpi,axis=1)
        #wgg_ = 2*simps(xi_rppi,pi,axis=1)

        
        return wgg_
    
    def sigmacrit(self,z1,z2):
        if z1 > z2:
            sigmacrit = 0.
        else:
            D1 = self.spline_chi_ang(z1)
            D2 = self.spline_chi_ang(z2)
            D12 = (D2 - D1)
            sigmacrit = (1.+z2)**2*const_dsigma*D12*D1/D2
        return sigmacrit
    
    def sigma_inv(self,z1,z2):
        result = self.spline_nz1(z1)*self.spline_nz2(z2)*self.sigmacrit(z1,z2)
        return result    
    
    def get_lensing_wgp_contribution(self,rp):
        
        ### This function get the prediction of gammat that will be injected into wgp
        
        Pkgm = self.h**3*self.pk_gm.eval(self.ks,1./(1+self.zeff),self.cosmo)
        r,dsigma=HT.k_to_r(self.k,Pkgm,1.,-1.,2., 1./(2*np.pi))
        gammat = dsigma/self.sigmacrit_eff
        spline_gammat = CS(r,gammat)
        gammat_ = spline_gammat(rp)
    
        return gammat_
    

    
