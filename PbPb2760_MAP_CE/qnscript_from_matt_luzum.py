#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:39:47 2019

@author: mluzum
"""

#%%
#import matplotlib
#matplotlib.use('Agg')

import os

#import PCA
#import jackknife as jk
#import mapp

#from pathlib import Path
import numpy as np


# Input data format, for reading raw event results
from configurations import *
from calculations_file_format_single_event import return_result_dtype


masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
ptlist = (ptcuts[1:]+ptcuts[:-1])/2

#%%

# Read simulation results from all events in a single design point
# design_dir is the directory containing the event files 0.dat to 499.dat 
# (upper limit now determined by input Nevents, to accomodate XeXe with 400 events)
# skiplist is the list of unwanted problematic events 
# output format will be:
# alldata[event][observable(4=Qn)][delta_f?][pid][Q0 or Qn][pt]
# with maximum indices
# alldata[499][4][3][4][1][55]
def readevents(design_dir,Nevents,skiplist,STAR_or_ALICE):
    # result_dtype = return_result_dtype('ALICE')
    result_dtype = return_result_dtype(STAR_or_ALICE)
    # This will store all results from all events at 1 design point.
    # From this we can pull any desired info, such as the differential Q vectors
    # filename = design_dir + '0.dat'
    alldata = np.array([],dtype=result_dtype)
    for event in range(Nevents):
        if event not in skiplist:
            filename = design_dir + str(event) + '.dat'
            eventdata = np.fromfile(filename, dtype=result_dtype)
            alldata = np.append(alldata,eventdata)
    return alldata

# functions to extract specific quantities from the full dataset
# Q0[event, pt] for identified particle with index pid 
# 0=pion, 1=kaon, 2=proton, 3=Sigma, 4=Xi
def get_Q0(alldata,delta_f_index,pid):
    # Q0 = np.array([],dtype=int)
    Q0 = [alldata[0][4][delta_f_index][pid][0]]
    for event in range(1,len(alldata)):
        Q0event = alldata[event][4][delta_f_index][pid][0]
        # Q0 = np.append(Q0, Q0event, axis=0)
        Q0 = np.concatenate((Q0, [Q0event]), axis=0)
    return Q0

# number of smash events in each hydro event
def get_nsamples(alldata,delta_f_index):
    #nsamples = np.array([],dtype=int)
    nsamples = np.array([],dtype=float)
    for event in range(len(alldata)):
        nsamples_event = alldata[event][3][delta_f_index][0]
        nsamples = np.append(nsamples, nsamples_event)
    # replace any zeros with inf. Per-sample averages will therefore be zero.
    nsamples[nsamples==0]=np.inf
    return nsamples

# result is Qn[event,pt] for a particular identified particle
def get_Qn(alldata,delta_f_index,pid):
    # Qn = np.array([], dtype=np.complex128)
    Qn = [alldata[0][4][delta_f_index][pid][1]]
    for event in range(1,len(alldata)):
        Qnevent = alldata[event][4][delta_f_index][pid][1]
        Qn = np.concatenate((Qn, [Qnevent]),axis=0)
        # Qn = np.append(Qn, Qnevent,axis=0)
    return Qn



# error estimate from jackknife resampling.
# function should ask for an argument which is
# a numpy array with each row representing an event
# with output the final event-averaged observable
def jackknifeerror(function, arg):
    # direct calculation of observable with all events
    val = function(arg)
    nevents = len(arg)
    jkdist = np.array([])
    for i in range(nevents):
        # remove i'th event and calculate observable,
        # appending to list jkdist
        samplearg = np.delete(arg,i,axis=0)
        jkdist = np.append(jkdist, function(samplearg))
#        jkdist = np.concatenate(jkdist, function(samplearg), axis=0)
    # jackknife error is propoortional to the 
    # standard deviation of this distribution
    error = np.sqrt(nevents - 1)*np.std(jkdist, axis=0)
    # estimate of bias
#    bias = (nevents - 1)*(np.mean(jkdist) - val)
    # bias-corrected jackknife estimate
    corrval = nevents*val - (nevents-1)*np.mean(jkdist, axis=0)
    return [corrval, error]

#
## pt integrated observables, only requiring integrated Q_n vectors
#
## function to calculate integrated v_n{2} from
## a numpy array of intregrated Q vectors "Qn", one per event
## plus the particle number "Q0"
## concatoneted so that Qn is the first column of the argumen
## and Q0 the next
## See Eq. 3.56 in Ante Bilandic's thesis
## Must wight Q0 properly to take into account the rapidity->pseudorapidity transformation!
#def integratedvn2(QnQ0):
#    Qn = QnQ0[:,0]
#    Q0 = QnQ0[:,1]
#    # Subtract self-correlations
#    temp = np.absolute(Qn)**2 - Q0
#    # Sum over events and normalize to construct the 2-particle correlation matrix c_n{2}
#    temp = np.sum(temp)
#    temp /= np.sum(Q0*(Q0-1))
#    # Take square root to compute v_n{2}, retaining the sign of c_n{2}
#    temp = np.sign(temp)*np.sqrt(np.absolute(temp))
#    return temp


# have to fix for rapidity->pseudorapidity conversion. 
# Self correlation corrections are more complicated!
# See Appendix B of Ante Bilandic's thesis.
# QnQ0dQ0 = np.c_[Qn,Q0,dQ0]
def integratedvn2(QnQ0dQ0):
    Qn = QnQ0dQ0[:,0] # ordinary Q vector \sum w e^{-in\phi}
    Q0 = QnQ0dQ0[:,1].astype(float) # multiplicity -- single weighted Q0 \sum w
    dQ0 = QnQ0dQ0[:,2].astype(float) # doubly weighted Q0 \sum w^2
    # Subtract self-correlations
    temp = np.absolute(Qn)**2 - dQ0
    # Sum over events and normalize to construct the 2-particle correlation matrix c_n{2}
    temp = np.sum(temp)
    temp /= np.sum(Q0*Q0 - dQ0)
    # Take square root to compute v_n{2}, retaining the sign of c_n{2}
    temp = np.sign(temp)*np.sqrt(np.absolute(temp))
    return temp





## function to calculate integrated v_n{4} from
## a numpy array of intregrated Q vectors "Qn", one per event
## plus the double harmonic Q2n for subtracting self-correlations 
## (probably not important, but might as well use the exact expressions.
##  input array of zeros if not wanted or if not available.)
## plus particle number Q0
## See Eq. 3.56 in Ante Bilandic's thesis
#  See also Appendix B of the same thesis.
#  The jacobian factor and the normalization of SMASH events
#  can both be thought of as particle weights in the averages,
#  and we must use these more complicated expressions for the self-correlation corrections.
# If zeros are supplied for everything except Qn and Q0, self-correlation corrections will be neglected,
# but there may be no significant difference, especially with oversampled events.
# See Eq.  B16 and B.17 of Ante's thesis
def integratedvn4(arg):
    # split into 1-dimensional arrays (1 value per event)
    Qn = arg[:,0]
    dQn = arg[:,1]  # double weighting of jacobian and nsamples
    tQn = arg[:,2] # triple weighting
    dQ2n = arg[:,3] # doubly weighted double harmonic (2n)
    Q0 = arg[:,4].astype(float)
    dQ0 = arg[:,5].astype(float)
    tQ0 = arg[:,6].astype(float)
    qQ0 = arg[:,7].astype(float) # quad weighting
    # First contruct 2- and 4- particle correlation, subtracting self correlations
    c4 = np.absolute(Qn)**4 + np.absolute(dQ2n)**2
    c4 += -2*np.real(dQ2n*np.conjugate(Qn)*np.conjugate(Qn))
    #c4 += -4*dQ0*np.absolute(Qn)**2 + 8*np.absolute(dQn)**2 + 2*dQ0*dQ0-6*qQ0
    c4 += -4*dQ0*np.absolute(Qn)**2 + 8*np.real(tQn*np.conjugate(Qn)) + 2*dQ0*dQ0-6*qQ0
    c4 = np.sum(c4)
    #c4 /= np.sum(Q0*(Q0-1)*(Q0-2)*(Q0-3))
    c4 /= np.sum(np.power(Q0,4) - 6*dQ0*Q0*Q0 + 3*dQ0*dQ0 + 8*Q0*tQ0 - 6*qQ0)
    c2 = np.absolute(Qn)**2 - dQ0
    c2 = np.sum(c2)
    c2 /= np.sum(Q0*Q0-dQ0)
    # The final observable is the 4th root of a combination of c4 and c2
    vn4 = 2*c2**2 - c4 # = -cn{4} in the literature
    vn4 = np.sign(vn4)*np.power(np.abs(vn4),0.25)
    return vn4


# total multiplicity
# expects array of particle number in each event
# already normalized by the number of SMASH samples
def multiplicity(Q0):
    mult = np.mean(Q0)
    return mult

# observables requiring pt-differential Q vectors
    
# pt spectra
# each row of Q0 should be the pt spectum of each event,
# already normalized by the number of SMASH events
def ptspectra(Q0):
    return np.mean(Q0,axis=0)


# pt and Q0 must be 2D arrays in event+pt space
# (Probably should just define pt list externally.  Fix later.)
#def meanpt(ptQ0):
#    pt = ptQ0[:,0]
#    Q0 = ptQ0[:,1]
#    temp = pt*Q0
#    temp = np.sum(temp)
#    temp /= np.sum(Q0)
#    return np.mean(temp)



def meanpt(Q0):
    temp = ptlist*Q0
    temp = np.sum(temp)
    temp /= np.sum(Q0)
    return np.mean(temp)


# I have to find the paper where they measure meanpt fluctuations.
# I have no idea if this function is right....
#def ptflucs(Q0dQ0):
#    Q0 = Q0dQ0[:,0]
#    dQ0 = Q0dQ0[:,1]
#    meanpt = np.mean(ptlist*Q0)
#    meanptsq = np.mean(ptlist*ptlist*Q0)
#    var = 
#    


# differential flow vn{2}(pt)
# argument is multidimensional array of integrated Qn vector
# differential Qn vector and differential spectra (Q0)
# plus doubly weighted differential spectra.
# See section starting with Eq. (3.63) in Ante's thesis
# Easiest to call this for each pt bin individually.  
# Maybe can revise later to call once for entire pt range.
def diffvn2(intQndiffQnintQ0diffQ0):
    Qn = intQndiffQnintQ0diffQ0[:,0]  # Q-vector of particles in large pt range
    pn = intQndiffQnintQ0diffQ0[:,1]  # q-vector of particles in pt bin
    Q0 = intQndiffQnintQ0diffQ0[:,2]  # (single weighted) n of particles in entire range
    mq = intQndiffQnintQ0diffQ0[:,3]  # singly weighted n of particle in pt bin
    dmq = intQndiffQnintQ0diffQ0[:,4] # doubly weighted number of particles in pt bin
    dQ0 = intQndiffQnintQ0diffQ0[:,5] # doubly weighted number of particles in large pt range
    dn = pn*np.conjugate(Qn) - dmq
    dn /= (mq*Q0 - dmq)
    dn = np.real(np.mean(dn, axis=0))
    vn2 = np.absolute(Qn)**2 - dQ0
    # Sum over events and normalize to construct the 2-particle correlation matrix c_n{2}
    vn2 = np.sum(temp)
    vn2 /= np.sum(Q0*Q0 - dQ0)
    # Take square root to compute v_n{2}, retaining the sign of c_n{2}
    vn2 = np.sign(temp)*np.sqrt(np.absolute(temp))
#    dn /= vn2
    # Subtract self-correlations
    return dn/vn2




# Higher level operations
    
#  Analysis should be done independantly for each choice of delta f.
# Let's start by creating a multidmensional numpy array for a single delta f
# [event, pid, harmonic, pt]    anything else?
#  We want to keep this stored in memory for the entire analysis:

# Step 1 is centrality selection.  Create a single event vector (numpy array)
# with desired quantity (e.g., ch. multiplicity in some pseudorapidity range), 
# and perform a sort.  




# Gives list of event indices, in order of largest Q0 to smallest.
# We'll keep this in memory and always use it to generate results
# for a specific centrality bin.
def centsort(Q0):
    srt = Q0.argsort()[::-1]
    return np.array(srt)

# Centrange should be an array of the form [A,B] in percent
#  E.g., for 10-20% centrality,  centrange = [10-20]
# Returns a list of event indices belonging to this centrality bin
def centbin(arglist, centrange):
    Nevents = len(arglist)
    Nmin = np.floor_divide(int(centrange[0]*Nevents),100)
    Nmax = np.floor_divide(int(centrange[1]*Nevents),100)
    return np.array(arglist[Nmin:Nmax])


#def integrated_jacobian(m, pT, etaCut):
#
#    m2=m*m
#    pT2=pT*pT
#    cosh2EtaCut=np.cosh(2*etaCut)
#    sinhEtaCut=np.sinh(etaCut)
#
#    return np.log((np.sqrt(2*m2 + pT2 + pT2*cosh2EtaCut) + np.sqrt(2)*pT*sinhEtaCut)/(np.sqrt(2*m2 + pT2 + pT2*cosh2EtaCut) - np.sqrt(2)*pT*sinhEtaCut)) #/(2.*etaCut)


    
# Returns a matrix conversion factor dN/dy -> dN/deta = jac*dN/Y
# with dimensions [pid,pt]
    

from scipy import integrate


#masslist = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
#ptcuts = np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
#ptlist = (pTcuts[1:]+pTcuts[:-1])/2
def jac(eta, mass, pt):
#    pTcuts=np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.])
#    NpT = len(pTcuts)-1
#    pt = (pTcuts[1:]+pTcuts[:-1])/2
#    NpT = len(pt)
#    Qn_species = [
#        ('pion', 211),
#        ('kaon', 321),
#        ('proton', 2212),
#        ('Sigma', 3222),
#        ('Xi', 3312),
#        ]
#    mass = np.array([0.13957, 0.49368, 0.93827, 1.18937, 1.32132])
#    Nspecies = len(mass)
    return pt*np.cosh(eta)/np.sqrt(mass*mass + pt*pt*np.cosh(eta)*np.cosh(eta))

def intjac(etarange, mass, pt):
    etamin = etarange[0]
    etamax = etarange[1]
    return integrate.quad(jac,etamin,etamax,args=(mass,pt))
    
# Returns a matrix conversion factor dN/dy -> dN/deta = jac*dN/Y
# with dimensions [pid,pt]
def jaclist(etarange):
    nmass = len(masslist)
    npt = len(ptlist)
    jaclst = np.zeros((nmass,npt))
    for i in  range(npt):
        for j in range(nmass):
            jaclst[j,i] = intjac(etarange,masslist[j],ptlist[i])[0]
    return jaclst
    
#jac = lambda mass, pt, eta:pt*np.cosh(eta)/np.sqrt(mass*mass + pt*pt*np.cosh(eta)*np.cosh(eta))

# Multiply by Jacobian factor to take into account pseudorapidity cut
# At the same time, normalize by the number of SMASH events in each hydro event.
#  Apply multiple times (i.e., nested) to get multiply-weighted Q-vectors for use with 
# cumulant observables.
def cutQn(Qn, etarange, nsamples):
    list = jaclist(etarange)
    #list = list.astype(complex)
    Qn /= 2 # Qn was calculated in a range of 2 units of rapidity.
    Qn *= list[None,:,None,:] # per-rapidity to per-pseudorapidity
    Qn /= nsamples[:,None,None,None] # per SMASH sample
#    for pt in range(len(ptlist)):
#        # list[None,:,None,pt] *= Qn[:,pt] 
#        Qn[:,:,:,pt] *= list[None,:,None,pt]
    #return jaclist(etarange)*Qn
    return Qn

# Take the results with pseudorapidity cut (cutQn above) and rebin in pt.
# pt limits must match one of those from ptcuts.  
def cutpt(Qn, ptbin):
    temp = np.argwhere(ptcuts == ptbin[0])
    if len(temp)!=0: 
        minbin = temp[0,0]
    else:
        print('Lower pt cut not valid')
    temp = np.argwhere(ptcuts == ptbin[1])
    if len(temp)!=0: 
        maxbin = temp[0,0]
    else:
        print('Upper pt cut not valid')
#    minbin = np.argwhere(ptcuts == ptbin[0])[0,0]
#    maxbin = np.argwhere(ptcuts == ptbin[1])[0,0]
    return np.sum(Qn[:,:,:,minbin:maxbin], axis=3)





# Output all observables for single design point and single delta f
        
# start with centrality selection   
#  Input is full Qn list for [event, particle, harmonic, pt]
# def sortevents(Qn, sevents, etarange, ptrange):
def sortevents(Qn, nsamples,  etarange, ptrange):
#    etarange = [-1,1]
    # Convert all per-rapidity Qn to per-pseudorapidity Qn in given range
    # and per-sample.
    Qn = cutQn(Qn,etarange,nsamples)
    # Combine desired pt range
    Qn = cutpt(Qn,ptrange) # [event,particle,harmonic]
    Q0 = Qn[:,:,0] # [event,particle]
    #Q0 /= sevents[:,None]
    # Charged hadron Q0
    Q0ch = 2*np.sum(Q0, axis = 1)
#    diffQ0ch = Qnch[:,0]
#    ptrange = [0,10]
    # Q0ch = cutpt(diffQ0ch, ptrange)
    # List of events sorted by multiplicity, highest to lowest.
    # Use this list for all centrality selections 
#    eventlist = centsort(Q0ch)
    Q0ch = Q0ch.astype(float)
    eventlist = Q0ch.argsort()[::-1]
    return eventlist
    
#    centbins = np.array([[0,10],[10,20],[20,30],[30,40],[40,50]]) # for testing
        
#    return eventlist
#    # First loop is over centrality.  Everything is done independently 
#    for cent in centbins:
#        Qnlistbinned = Qnlist[centbins]
#        jac = jaclist(etarange)
#        for event in centbin(eventlist, cent):
#            
#            
#        
#    
#    for n in range(1,6):
#        diffQnch = Qnch[:,n]
#        intQnch = cutpt(diffQ0ch,ptrange)
    
 

def finalmultiplicity (Qn, sevents, eventlist, etarange, ptrange):
    Q0 = Qn[:,0,:,:]
    Q0bin = Qn[eventlist]
    Q0eta = cutQn(Qnbin, etarange)
    Q0etapt = cutpt(Q0eta, ptrange)
    Q0etapt /= sevents
#    return multplicity(Q0etapt)  # without uncertainty
    return jackknifeerror(multiplicity, Q0etapt)
        
def combineQn(alldata,delta_f_index, Nparticles, Nevents, Npt, Nharmonics):
    Qn = np.full((Nparticles, Nevents, Npt, Nharmonics+1),0+0j)
    for pid in range(Nparticles):
        Q0pid = get_Q0(alldata,delta_f_index,pid)
        Qnpid = get_Qn(alldata,delta_f_index,pid)
        Qnnpid = np.insert(Qnpid,0,Q0pid,axis=2)
        Qn[pid] = Qnnpid
    # Change axes so the array is [event,harmonic, particle, pt]
    # All functions will expect this common format.
    Qn = np.swapaxes(Qn,0,1)
    Qn = np.swapaxes(Qn,2,3)
    return Qn

# For a single design point and single delta_f, compute all observables, including centrality selection
#def allobservables(Q0, Qn, sevents):
def allobservables(design_dir, delta_f_index,  Nevents, STAR_or_ALICE):
     Nparticles = len(masslist)
     #Nparticles = 5
     Npt = 56
     Nharmonics = 5
     # extract raw data from files
     alldata = readevents(design_dir,Nevents,[],STAR_or_ALICE)
#
     # extract useful information and format as numpy arrays
     # number of SMASH events for each hydro event
     nsamples = get_nsamples(alldata,delta_f_index)
#
     # Putting all harmonics together changes Q0 to a complex number.  We'll see if this is a problem.
#     # To convert Q0 back to integer from complex:   Q0.astype(int)
     Qn = combineQn(alldata,delta_f_index, Nparticles, Nevents, Npt, Nharmonics)
#     Qn = np.full((Nparticles, Nevents, Npt, Nharmonics+1),0+0j)
#     for pid in range(Nparticles):
#         Q0pid = get_Q0(alldata,delta_f_index,pid)
#         Qnpid = get_Qn(alldata,delta_f_index,pid)
#         Qnnpid = np.insert(Qnpid,0,Q0pid,axis=2)
#         Qn[pid] = Qnnpid
#     # Change axes so the array is [event,harmonic, particle, pt] 
#     # All functions will expect this common format.
#     Qn = np.swapaxes(Qn,0,1)
#     Qn = np.swapaxes(Qn,2,3)
#     #Qn = np.moveaxis(Qn, 0, 2) # pid
#     # We start every analysis with a pseudorapidity cut on this full array.
#     # Only then can we integrate pt and/or combine charged hadrons.
#     
     # order events for centrality selection.
     # Use charged hadron multiplicity in the following phase space
     etarange = [-1,1]
     ptrange = [0,10]
#     
     # Qnpersample = Qn/nsamples[:,None,None,None]
     # List of events indices, sorted by latest to smallest multiplicity
     eventlist = sortevents(Qn, nsamples, etarange, ptrange)
#     
#     
     # Arrange calculations by common centrality binning
     centbins = np.array([[0,10],[10,20],[20,30],[30,40],[40,50]])
     ncentbins = len(centbins)
     Nch = np.zeros(ncentbins)  # integrated charged hadron multiplicity
     Nchpt = np.zeros(ncentbins,nptbins) # charged hadron spectra
     pionspectra = np.zeros(ncentbins,nptbins)
#     Qnlist /= sevents
#
#
     ptrange = [0.3,3]
     Qncut = cutQn(Qn, etarange, nsamples)
     Qndcut = cutQn(Qncut, etarange, nsamples) # doubly weighted
     Qntcut = cutQn(Qndcut, etarange, nsamples) # triple weighting
     Qnqcut = cutQn(Qntcut, etarange, nsamples) # quadruple weighting
     Qnch = 2*np.sum(cutpt(Qncut, ptrange), axis=1)
     Qndch = 2*np.sum(cutpt(Qndcut, ptrange), axis=1)
     Qntch = 2*np.sum(cutpt(Qntcut, ptrange), axis=1)
     Qnqch = 2*np.sum(cutpt(Qnqcut, ptrange), axis=1)
     Qnpersamplecut = cutQn(Qnpersample,etarange)
     Qndpersamplecut = cutQn(Qnpersamplecut/nsamples[:,None,None,None],etarange) # double jacobian factor, for self correlation subtraction 
     Qntpersamplecut = cutQn(Qndpersamplecut/nsamples[:,None,None,None],etarange) 
     Qnqpersamplecut = cutQn(Qntpersamplecut/nsamples[:,None,None,None],etarange) # quad jacobian factor, for self correlation subtraction 
     Qnpersamplecut = cutpt(Qnpersamplecut,ptrange)
     Qndpersamplecut = cutpt(Qndpersamplecut,ptrange)
     Qnch = 2*np.sum(Qnpersamplecut, axis=1)
     Qndch = 2*np.sum(Qndpersamplecut, axis=1)
     Qnqch = 2*np.sum(cutpt(Qnqpersamplecut,ptrange), axis=1)

     Q2 = Qnch[:,2]
     Q0 = Qnch[:,0].astype(float)
     dQ0 = Qndch[:,0].astype(float)
     qQ0 = Qnqch[:,0].astype(float)
#     arg = np.append([Q2],[Q0], axis=0)
#     arg = np.append(arg,[dQ0], axis=0)
#     arg = np.swapaxes(arg,0,1)
     arg = np.c_[Q2,Q0,dQ0]
     dQ2 = Qndch[:,2]
     dQ4 = Qndch[:,4]
     arg4 = np.c_[Q2,dQ2,dQ4,Q0,dQ0,qQ0]


     Q0cut = cutQn(Q0/sevents,etarange)
#     Qnch = 2*np.sum(Qncut, axis = 2)
     Q0ch = 2*np.sum(Q0cut, axis = 2)
     for c in range(ncentbins):
         # Total charged hadron multiplicity 
         etarange = [-1,1]
         ptrange = [0.3,3]
         elist = centbin(eventlist,centbins[c])
         Nch[c] = finalmultiplicity(Qnch, sevents, elist, etarange, ptrange)
         
         ptranges = [[0.25,0.5],[0.5,1],[1,1.5],[1.5,2],[2,3],[3,5]]
         nptbins = len(ptranges)
         for p in range(len(ptranges)):
             ptrange = ptranges[p]
             Nchpt[c,p] = finalmultiplicity(Qnch, sevents, elist, etarange, ptrange)
             pionspectra[c,p] = finalmultiplicity(Qn[:,0], sevents, elist, etarange, ptrange)
    
    

#
## Cut in pseudorapidity must be made first, at the level of 
## the fully differential Q vector of identified particles.
## This function takes the differential Q vector (per unit rapidity)
## and returns the pt-differential Q vector for the given rapidity range
## Single event?  Entire event array?
## Maybe better to do everything at the same time.  
## Will usually use all charged hadrons anyway, which wil lreqire everything
##  So Qn should have 4 dimensions:  event, particle id, harmonic, pt?
#def etacut(Qn, etarange):
#    
#    
#        
#
## Teka a fully differential Q vector and return 
## the vector in a single given pt bin and pseudorapidity range
#def cut(Qn, ptrange, etarange):
#    nspecies = len(Qn_species)
#    
#    
#
#
## Take fully differential Q vectors representing a singla parameter set, 
## perform centrality selection, and output various observables
## for the given design point.
#def observables(Qn):
#    # First define the centrality selection criterion,
#    # since this affects all observables.
#    # In this case we choose total multiplicity in the kinematic range:
#    centptrange = [0.3,5]
#    centetarange = [2,5]
#    
#    nevents = len(Qn)
#    
#    for i in range(nevents):
#        
#
#    
#    
#    
##Qn_species = [
##        ('pion', 211),
##        ('kaon', 321),
##        ('proton', 2212),
##        ('Sigma', 3222),
##        ('Xi', 3312),
##]
##
### Qn_diff_pT_cuts=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.]
##Qn_diff_pT_cuts=[0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.,3.2,3.4,3.6,3.8,4.,10.]
##Qn_diff_NpT = len(Qn_diff_pT_cuts)-1
##Nharmonic = 8
##Nharmonic_diff = 5
##        
##        
### Q vector, diff-flow, identified charged hadrons
##('d_flow_pid', [(name, [('N', int_t, Qn_diff_NpT),
##                                                                ('Qn', complex_t, [Qn_diff_NpT, Nharmonic_diff])], 1)
##                                for (name,_) in Qn_species      ], number_of_viscous_corrections),
##]
##
##"""
##NpT = 10
##Nharmonic = 8
##Nharmonic_diff = 5

        


#%%
