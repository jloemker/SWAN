import uproot
import pandas as pd
import mplhep as hep
import numpy as np
import awkward
import ROOT
import scipy.stats as stats #this one used to do fits
import matplotlib.pyplot as plt
import awkward as ak

sqrt_s = 14000

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

def GiveMu(muons, mu1_idx, mu2_idx):
    mu = muons
    mu1=ROOT.Math.PtEtaPhiMVector(mu.pfcand_pt[mu1_idx],
                                  mu.pfcand_eta[mu1_idx],
                                  mu.pfcand_phi[mu1_idx],
                                  mu.pfcand_mass[mu1_idx])
    mu2=ROOT.Math.PtEtaPhiMVector(mu.pfcand_pt[mu2_idx],
                                  mu.pfcand_eta[mu2_idx],
                                  mu.pfcand_phi[mu2_idx],
                                  mu.pfcand_mass[mu2_idx]) 
    return mu1, mu2


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
        mu1, mu2 = GiveMu(mu, mu1_idx, mu2_idx)
        xi_dimu_plus = ((mu1.Pt()*np.exp(mu1.Rapidity())+mu2.Pt()*np.exp(mu2.Rapidity())) / sqrt_s) 
        xi_dimu_minus =((mu1.Pt()*np.exp(-mu1.Rapidity())+mu2.Pt()*np.exp(-mu2.Rapidity())) / sqrt_s)
    #additional cut
        if((xi_dimu_plus<0.0032) == True & (xi_dimu_minus<0.0032) == True): continue
    return mu1_idx, mu2_idx


def SmearPosProtonMomentum(proton_from_event):
    XI_RES=0.02 # use 2% for now
    pr=proton_from_event
    xi_smear = np.random.normal(0,abs(ak.to_numpy(pr.genproton_xi))*XI_RES)
    pr.genproton_xi = pr.genproton_xi + xi_smear
    pr.genproton_pz = pr.genproton_pz + 7000*xi_smear         
    #return corrected array of protons
    return pr

def SmearNegProtonMomentum(proton_from_event):
    XI_RES=0.02 # use 2% for now
    pr=proton_from_event
    xi_smear = np.random.normal(0,abs(ak.to_numpy(pr.genproton_xi))*XI_RES)
    pr.genproton_xi = pr.genproton_xi - xi_smear
    pr.genproton_pz = pr.genproton_pz - 7000*xi_smear         
    #return corrected array of protons
    return pr

def SelProtons(proton_from_event, mu1, mu2):
    PZ_MIN=4990; PZ_MAX=6999.99
    pr=proton_from_event
    # accepted proton indices
    proton_neg_idx_acc=np.where(ak.to_numpy((pr.genproton_pz<-PZ_MIN) & (pr.genproton_pz>-PZ_MAX) ))[0]
    proton_pos_idx_acc=np.where(ak.to_numpy((pr.genproton_pz>PZ_MIN) & (pr.genproton_pz< PZ_MAX) ))[0]
    # accepted protons
    proton1 = pr[proton_pos_idx_acc]
    proton2 = pr[proton_neg_idx_acc]
    # smearing proton momenta
    SmearPosProtonMomentum(proton1)
    SmearNegProtonMomentum(proton2)    
    # di-muon kinematics
    xi_dimu_plus = ((mu1.Pt()*np.exp(mu1.Rapidity())+mu2.Pt()*np.exp(mu2.Rapidity())) / sqrt_s) 
    xi_dimu_minus =((mu1.Pt()*np.exp(-mu1.Rapidity())+mu2.Pt()*np.exp(-mu2.Rapidity())) / sqrt_s)
    #additional cut
    if((xi_dimu_plus>0.0032) == True & (xi_dimu_minus>0.0032) == True):
    #get protons with closest xi values to the reconstructed muons from the list of accepted protons
        proton_idx1_acc = ak.to_numpy(abs(proton1.genproton_xi-xi_dimu_plus)).argmin()
        proton_idx2_acc = ak.to_numpy(abs(proton2.genproton_xi-xi_dimu_minus)).argmin()
    # get the proton index for the full list of protons:
        proton_idx1 = proton_pos_idx_acc[proton_idx1_acc]   
        proton_idx2 = proton_neg_idx_acc[proton_idx2_acc]
    else:
        proton_idx1 = -1
        proton_idx2 = -1
    # return proton indices   
    return proton_idx1, proton_idx2



def Fill_mu(data, mu, mu1, mu2,mu1_idx,mu2_idx):
    mu = mu
    data['mu1_pt'].append(mu1.Pt())
    data['mu1_eta'].append(mu1.Rapidity())
    data['mu1_phi'].append(mu1.Phi())
    data['mu1_m'].append(mu1.M())
    data['mu1_t'].append(mu.pfcand_t[mu1_idx])
    data['mu1_vz'].append(mu.pfcand_vz[mu1_idx])
    data['mu2_pt'].append(mu2.Pt())
    data['mu2_eta'].append(mu2.Rapidity())
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
    
