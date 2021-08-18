import uproot
import pandas as pd
import mplhep as hep
import numpy as np
import awkward
import ROOT
import scipy.stats as stats #this one used to do fits
import matplotlib.pyplot as plt
import awkward as ak
import MyHelpers as mh
from tqdm import tqdm #this is a fancy feature to make a progress bar as the loop proceed


#to make the plots in CMS style execute this line
plt.style.use([hep.style.ROOT, hep.style.firamath])
plt.style.use(hep.style.CMS)

sqrt_s = 14000
zpps = 2.34e4 #in cm
c = 29.9792#in cm/ns 

def fit_mass(sig):
    data = (sig['mpp'].values/sig['mll'].values)-1
    data = data[ abs(data) < 0.05]
    (mu_m, sigma_m) = stats.norm.fit(data)
    return(sigma_m)

def fit_y(sig):
    new_data = (sig['ypp'].values - sig['yll'].values)/sig['yll'].values
    new_data = new_data[~np.isnan(new_data) &(abs(new_data) < 0.1) ]
    (mu_y, sigma_y) = stats.norm.fit(new_data)
    return sigma_y

def fit_vz(sig,res):
    data = (-(sig['pr1_'+str(res)+'_t'] - sig['pr2_'+str(res)+'_t'])*c/2 - sig['pr_vtx_z'].values)
    if res == 50:
        data = data[~np.isnan(data) & (abs(data) < 1.6) ]
    if res == 20:
        data = data[~np.isnan(data) & (abs(data) < 1) ]
    (mu_vz, sigma_vz) = stats.norm.fit(data)
    return sigma_vz
    
def fit_t(sig,res):
    tmu =  ( sig['mu1_t'].values  + sig['mu2_t'].values )/2
    tpp =  (( sig['pr1_'+str(res)+'_t'].values + sig['pr2_'+str(res)+'_t'].values ) - 2*zpps/c)/2
    data = (tpp-tmu)
    data = data[~np.isnan(data) & ~np.isneginf(data)]
    (mu_t, sigma_t) = stats.norm.fit(data)
    return sigma_t  
    
def full_selection(data,res, sigma_m, sigma_y,sigma_vz,sigma_t):
    t = []
    vz = []
    y = []
    m = []
    acc = 0
    #kinematics
    data_m = (data['mpp']-data['mll']) 
    data_y = abs(data['ypp']-data['yll'])
    #timing
    p1_t = data['pr1_'+str(res)+'_t'].values
    p2_t = data['pr2_'+str(res)+'_t'].values    
    mu1_t = data['mu1_t'].values 
    mu2_t = data['mu2_t'].values 
    t_pp = (( p1_t + p2_t) - 2*zpps/c)/2
    data_t = abs(t_pp -  (mu1_t+mu2_t)/2 )
    #position
    pp_vz = - (data['pr1_'+str(res)+'_t'].values - data['pr2_'+str(res)+'_t'].values)*c/2 
    vz_4D = data['pr_vtx_z'].values
    data_vz = abs(pp_vz - vz_4D)
    #selection
    for i in range(0,len(data_t)):
        if data_t[i] < abs(2*sigma_t):
            if data_vz[i] < abs(2*sigma_vz):           
                if data_y[i] < abs(2*sigma_y*data['yll'][i]):
                    if data_m[i] < abs(2*sigma_m*data['mll'][i]):
                        t = np.append(t, t_pp[i])
                        vz = np.append(vz, pp_vz[i])
                        y = np.append(y, data['ypp'][i])
                        m = np.append(m, data['mpp'][i])
                        acc = acc+1
    acc_rate = acc/len(data_t)
#    print('number of unselected events:')
#    print(len(data_t))
#    print('number of selected evets:')
#    print(acc)
#    print('acceptance rate:')
#    print(acc_rate)
    return acc, len(data_t), t, vz, y, m

    
def create_frame(muons,protons,vertices, event_info, bg_muons,bg_protons, bg_vertices, bg_event_info):
    #define data array for signal
    sig_data = {}
    mh.InitData(sig_data)
    #loop over all events for the signal
    N = len(muons)
    for i in range(N): 
        #find index of two highest pT muons with pT > 25 GeV
        mu=muons[i]
        mu1_idx, mu2_idx = mh.SelMu(mu)            
        #if found less than 2 muons, skip the event:
        if mu1_idx<0 or mu2_idx<0: continue 
        mu1, mu2 = mh.GiveMu(mu, mu1_idx, mu2_idx)
        #exclude muons with dummy values
        if (mu.pfcand_t[mu1_idx]<-80 or mu.pfcand_t[mu2_idx]<-80): continue
        xi_dimu_plus = ((mu1.Pt()*np.exp(mu1.Rapidity())+mu2.Pt()*np.exp(mu2.Rapidity())) / sqrt_s) 
        xi_dimu_minus =((mu1.Pt()*np.exp(-mu1.Rapidity())+mu2.Pt()*np.exp(-mu2.Rapidity())) / sqrt_s)
        #additional cut scenario 1-3
        #0.0189−0.0095
        if((xi_dimu_plus<0.01) == True & (xi_dimu_minus<0.01) == True): continue 
        #additional cut scenario 1-4
        #if((xi_dimu_plus<0.0032) == True & (xi_dimu_minus<0.0032) == True): continue   
        # find two signal protons:
        pr=protons[i]
        # smearing and selecting protons
        pr1_idx, pr2_idx = mh.SelProtons(pr,mu1,mu2, xi_dimu_plus, xi_dimu_minus)
        if pr1_idx<0 or pr2_idx<0: continue
        vx = vertices[i]
        ev = event_info[i]
        #Filling muon and proton events
        mh.Fill_mu(sig_data, mu, mu1, mu2, mu1_idx,mu2_idx)
        mh.Fill_pr(sig_data,pr,pr1_idx,pr2_idx,vx,ev)
        mh.Sig_time(sig_data, ev, pr, pr1_idx, pr2_idx)
        #Add smeared proton times
        mh.Fill_smeared_pr_t(sig_data)
    sig = pd.DataFrame(data=sig_data)

    #define data array for Background
    bg_data = {}
    mh.InitData(bg_data)
    #loop over all events for the background
    n = len(bg_muons)
    for i in range(n): 
        #find index of two highest pT muons with pT > 25 GeV
        mu=bg_muons[i]
        mu1_idx, mu2_idx = mh.SelMu(mu)    
        #if found less than 2 muons, skip the event:
        if mu1_idx<0 or mu2_idx<0: continue 
        mu1, mu2 = mh.GiveMu(mu, mu1_idx, mu2_idx)
        #exclude muons with dummy values
        if (mu.pfcand_t[mu1_idx]<-80 or mu.pfcand_t[mu2_idx]<-80): continue
        xi_dimu_plus = ((mu1.Pt()*np.exp(mu1.Rapidity())+mu2.Pt()*np.exp(mu2.Rapidity())) / sqrt_s) 
        xi_dimu_minus =((mu1.Pt()*np.exp(-mu1.Rapidity())+mu2.Pt()*np.exp(-mu2.Rapidity())) / sqrt_s)
        #additional cut scenario 1-3
        #0.0189−0.0095
        if((xi_dimu_plus<0.01) == True & (xi_dimu_minus<0.01) == True): continue 
        #additional cut scenario 1-4
        #if((xi_dimu_plus<0.0032) == True & (xi_dimu_minus<0.0032) == True): continue 
        # find two signal protons:
        pr=bg_protons[i]
        # smearing and selecting protons
        pr1_idx, pr2_idx = mh.SelProtons(pr,mu1,mu2, xi_dimu_plus, xi_dimu_minus)
        if pr1_idx<0 or pr2_idx<0: continue
        vx = bg_vertices[i]
        ev = bg_event_info[i]
        #Filling muon and proton events
        mh.Fill_mu(bg_data,mu, mu1, mu2, mu1_idx,mu2_idx)
        mh.Fill_pr(bg_data,pr,pr1_idx,pr2_idx,vx,ev) 
        mh.Bg_time(bg_data, pr, pr1_idx, pr2_idx)
        #Add smeared proton times
        mh.Fill_smeared_pr_t(bg_data)
    bg = pd.DataFrame(data=bg_data)    
    return(sig,bg)

 
