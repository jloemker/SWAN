import uproot
import pandas as pd
import mplhep as hep
import numpy as np
import awkward
import ROOT
import scipy.stats as stats #this one used to do fits
import matplotlib.pyplot as plt
from tqdm import tqdm #this is a fancy feature to make a progress bar as the loop proceed
import awkward as ak
import math as m
#to make the plots in CMS style execute this line
plt.style.use([hep.style.ROOT, hep.style.firamath])
plt.style.use(hep.style.CMS)

def InitData(data):
    for k in ['pt','eta','phi','m','t','vz']:
        data['mu1_'+k]=[]
        data['mu2_'+k]=[]
    for k in ['vz','xi']:
        data['pr1_'+k]=[]
        data['pr2_'+k]=[]
    for k in ['pt2','t','z']:
        data['pr_vtx_'+k]=[]
    # add event kinematics
        data['mll']=[]
        data['yll']=[]
        data['mpp']=[]
        data['ypp']=[]
        data['evt_t0']=[]
    return data

def SelMu(muons):
    mu = muons
    mu1_idx=-1; mu2_idx=-1; 
    for i_mu, pt in enumerate(mu.pfcand_pt):
        if pt<25: continue
        if mu1_idx<0 or pt>mu.pfcand_pt[mu1_idx]:
            mu2_idx=mu1_idx
            mu1_idx=i_mu
        elif mu2_idx<0 or pt>mu.pfcand_pt[mu2_idx]:
            mu2_idx=i_mu
    return mu1_idx, mu2_idx

def SelProtons(proton_from_event):
    pr=proton_from_event
    proton_neg_idx=np.where(ak.to_numpy((pr.genproton_pz<0) & (pr.genproton_pz>-6999) ))[0]
    proton_pos_idx=np.where(ak.to_numpy((pr.genproton_pz>0) & (pr.genproton_pz< 6999) ))[0]
    #create list of all possible pair combinations between list 1 and list 2:
    proton_pairs_idx=np.array(np.meshgrid(proton_pos_idx,proton_neg_idx)).T.reshape(-1,2)    
    proton_idx1, proton_idx2 = proton_pairs_idx[0]
    #proton_idx1 = proton_pos_idx[0]   
    #proton_idx2 = proton_neg_idx[0]
    return proton_idx1, proton_idx2

def Fill_mu(data,muons,mu1_idx,mu2_idx):
    mu = muons
    mu1=ROOT.Math.PtEtaPhiMVector(mu.pfcand_pt[mu1_idx],
                                  mu.pfcand_eta[mu1_idx],
                                  mu.pfcand_phi[mu1_idx],
                                  mu.pfcand_mass[mu1_idx])
    mu2=ROOT.Math.PtEtaPhiMVector(mu.pfcand_pt[mu2_idx],
                                  mu.pfcand_eta[mu2_idx],
                                  mu.pfcand_phi[mu2_idx],
                                  mu.pfcand_mass[mu2_idx])   
    data['mu1_pt'].append(mu1.Pt())
    data['mu1_eta'].append(mu1.Eta())
    data['mu1_phi'].append(mu1.Phi())
    data['mu1_m'].append(mu1.M())
    data['mu1_t'].append(mu.pfcand_t[mu1_idx])
    data['mu1_vz'].append(mu.pfcand_vz[mu1_idx])
    data['mu2_pt'].append(mu2.Pt())
    data['mu2_eta'].append(mu2.Eta())
    data['mu2_phi'].append(mu2.Phi())
    data['mu2_m'].append(mu2.M())
    data['mu2_t'].append(mu.pfcand_t[mu2_idx])
    data['mu2_vz'].append(mu.pfcand_vz[mu2_idx])
    #calculate invariant mass from two muons:
    data['mll'].append((mu1+mu2).M())
    data['yll'].append((mu1+mu2).Rapidity())
    
def Fill_pr(data,protons,pr1_idx,pr2_idx,vertex,event):
    pr=protons
    #add proton information
    xi1=pr.genproton_xi[pr1_idx]
    xi2=pr.genproton_xi[pr2_idx]
    data['pr1_xi'].append(xi1)
    data['pr1_vz'].append(pr.genproton_vz[pr1_idx])
    data['pr2_xi'].append(xi2)
    data['pr2_vz'].append(pr.genproton_vz[pr2_idx])
    #calculate invariant mass from two muons:
    data['mpp'].append(14000.*np.sqrt(xi1*xi2))
    data['ypp'].append((1/2)*np.log(xi1/xi2))
    #add primary vertex info
    vx=vertex
    data['pr_vtx_pt2'].append(vx.vtx4D_pt2[0])
    data['pr_vtx_t'].append(vx.vtx4D_t[0])
    data['pr_vtx_z'].append(vx.vtx4D_z[0])
    #add event info variables:
    ev=event
    data['evt_t0'].append(ev.genvtx_t0)