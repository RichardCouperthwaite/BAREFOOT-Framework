# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:24:19 2020

@author: richardcouperthwaite
"""

import numpy as np
from scipy.optimize import fsolve

def isostrain_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    s0F = np.zeros((mm,1))
    s0M = np.zeros((mm,1))
    sF = np.zeros((mm,1))
    sM = np.zeros((mm,1))
    stress = np.zeros((mm,10001))
    str_ = np.zeros((mm,1))
    dsde = np.zeros((mm,1))
    cc = np.zeros((mm,1))
    index = np.zeros((mm,90))
    
    for ii in range(mm):
        # yield strength of the phases
        s0F[ii]=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M[ii]=400+1000*((100*x_C[ii])**(1/3))
       
        kF=2200
        kM=450
        nF=0.5
        nM=0.06
        strain=np.linspace(0,1,10001,endpoint=True)
        for i in range(10001):
            sF[ii]=s0F[ii]+kF*strain[i]**nF
            sM[ii]=s0M[ii]+kM*strain[i]**nM
            stress[ii,i]=((1-f[ii])*sF[ii])+(f[ii]*sM[ii])
 
        index[ii,:] = np.array(np.nonzero(strain <= ep))
        str_[ii]=stress[ii,int(np.max(index[ii,:]))]
        dsde[ii]=(stress[ii,int(np.max(index[ii,:]))+1]-stress[ii,int(np.max(index[ii,:]))-1])/(2*(strain[int(np.max(index[ii,:]))+1]-strain[int(np.max(index[ii,:]))]))
        cc[ii]=dsde[ii]/str_[ii]
    
    if single_calc:
        return cc[0]
    else:
        return cc

def isostress_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    s0F = np.zeros((mm,1))
    s0M = np.zeros((mm,1))
    # sF = np.zeros((mm,1))
    # sM = np.zeros((mm,1))
    # stress = np.zeros((mm,10001))
    str_ = np.zeros((mm,1))
    dsde = np.zeros((mm,1))
    cc = np.zeros((mm,1))
        
    for ii in range(mm):
        # yield strength of the phases
        s0F[ii]=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M[ii]=400+1000*((100*x_C[ii])**(1/3))
        vf=f[ii]

        kF=2200
        kM=450
        nF=0.5
        nM=0.06
    
        # Overall Stress
        stress=np.linspace(170,1900,173000,endpoint=True)
        l=len(stress)
        strain = np.zeros((l,1))
        for i in range(l):
            if (stress[i] < s0F[ii]):
                epF=0;
            else:
                epF=((stress[i]-s0F[ii])/kF)**(1/nF)

            if (stress[i] < s0M[ii]):
                epM=0
            else:
                epM=((stress[i]-s0M[ii])/kM)**(1/nM);

            strain[i]=((1-vf)*epF)+(vf*epM);
        
        index = np.array(np.nonzero(strain <= ep))
        str_=stress[np.max(index)];
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]))
        
        cc[ii]=dsde/str_
    if single_calc:
        return cc[0]
    else:
        return cc

def isowork_IS(x,ep):
    beta_Si = 732.7676
    beta_Mn = 213.4494
    beta_C = 7507.582
    
    single_calc = False
    mm = x.shape[0]
    if x.shape[0] == 4:
        try:
            a = x.shape[1]
            x = x.transpose()
        except IndexError:
            x = np.array([[x[0],x[1],x[2],x[3]],[0,0,0,0]])
            single_calc = True
            mm = 1
    f=x[:,0]
    x_C = x[:,1]
    x_Mn = x[:,2]
    x_Si = x[:,3]
    
    cc = np.zeros((mm,1))
        
    for ii in range(mm):
        # yield strength of the phases
        s0F=200 + beta_Mn*((x_Mn[ii])**0.5) + beta_Si*((x_Si[ii])**0.5)
        s0M=400+1000*((100*x_C[ii])**(1/3))
        vf=f[ii]
        kF=2200
        kM=450
        nF=0.5
        nM=0.06
        # strain increment in ferrite
        depF=0.0001
        epF=np.zeros((10000,1))
        epM=np.zeros((10000,1))
        sF=np.ones((10000,1))*s0F
        sM=np.ones((10000,1))*s0M
        sT=np.zeros((10000,1))
        epT=np.zeros((10000,1))
        SS = np.zeros((10000,2))
        for k in range(9999): #i=2:(1/depF)
            i = k+1
            epF[i]=epF[i-1]+depF
            sF[i]=s0F+kF*epF[i]**nF
            wF=sF[i]*depF
            temp=epM[i-1]
            isow =  lambda wF,s0M,kM,nM,temp,depM : wF-((s0M+kM*(temp+depM)**nM)*depM)
            # isow=@(wF,s0M,kM,nM,temp,depM) wF-((s0M+kM*(temp+depM)^nM)*depM)
            fun = lambda depM : isow(wF,s0M,kM,nM,temp,depM)
            # fun=@(depM) isow(wF,s0M,kM,nM,temp,depM)
            depM=fsolve(fun,depF) # depF is initial guess
            epM[i]=epM[i-1]+depM
            sM[i]=s0M+kM*epM[i]**nM
            sT[i]=((1-vf)*sF[i])+(vf*sM[i])
            epT[i]=((1-vf)*epF[i])+(vf*epM[i])
            SS[i,0]=epT[i]
            SS[i,1]=sT[i]
        
        strain=np.zeros((10000,1))
        stress=np.zeros((10000,1))
        
        for iii in range(10000):
            strain[iii]=SS[iii,0]
            stress[iii]=SS[iii,1]
        
        index = np.array(np.nonzero(strain <= ep))
        str_=stress[np.max(index)];
        dsde=(stress[np.max(index)+1]-stress[np.max(index)-1])/(2*(strain[np.max(index)+1]-strain[np.max(index)]))
        
        cc[ii]=dsde/str_
        
    if single_calc:
        return cc[0]
    else:
        return cc

if __name__ == "__main__":
    print(isostrain_IS(np.array([0.5,0.1,0.1,0.1]),0.009))
    
    print(isostress_IS(np.array([0.5,0.1,0.1,0.1]),0.009))
    
    print(isowork_IS(np.array([0.5,0.1,0.1,0.1]),0.009))