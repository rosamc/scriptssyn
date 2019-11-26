import numpy as np
import pandas as pd
import re

parenthesis=re.compile(r'(\(.*\))')

def get_exp_matrix(df,TFnames,affinities,matnames=False):
    cGFP=['GFPfa1','GFPfa2','GFPfa3']
    naf=len(affinities)
    nTFs=len(TFnames)
    mat_fc=np.zeros((naf*nTFs+1,naf*nTFs)) #additional row on top is to compare to nothing on TF2
    if matnames:
        names_short=[x[0] if x!='HSF1m' else x[-1] for x in TFnames]
        affinities_short=[parenthesis.findall(x)[0] for x in affinities]
        mat_names=np.empty_like(mat_fc,dtype='object')

    for n1 in range(nTFs):
        TF1=TFnames[n1]
        for n1_ in range(naf):
            af1=affinities[n1_]
            for n2 in range(n1,nTFs):
                TF2=TFnames[n2]
                for n2_ in range(naf):
                    af2=affinities[n2_]
                    vals=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af1)&(df['affinity2']==af2)][cGFP]
                    #print(TF1, af1, TF2, af2, vals.values)
                    avGFP=np.nanmean(vals.values)
                    mat_fc[3*n1+n1_+1,3*n2+n2_]=avGFP
                    if matnames:
                        mat_names[3*n1+n1_+1,3*n2+n2_]=names_short[n2]+'('+affinities_short[n2_]+')\n'+names_short[n1]+'('+affinities_short[n1_]+')'
    for n1 in range(nTFs):
        TF1=TFnames[n1]
        for n1_ in range(naf):
            af1=affinities[n1_]
            vals=df[(df['activator1']==TF1)&(df['activator2']=='-')&(df['affinity1']==af1)][cGFP]
            mat_fc[0,3*n1+n1_]=np.nanmean(vals.values)
            if matnames:
                mat_names[0,3*n1+n1_]=names_short[n1]+' '+affinities_short[n1_]
            #print(TF1,af1,vals)
    if matnames:
        return [mat_fc,mat_names]
    else:
        return mat_fc

def get_parameters_TF_v1(pars,indicesbinding=None,indicesP=None,indicesaf=None,TFidx=None,afidx=None):
    """pars is the array of parameters to be optimized.
    indicesbinding is a list of 2 arrays: one with the indices for kbXa, kbXi, kbXn, and the other with the indices for kuXa, kuXi, kuXn
    indicesP is a list of arrays for each of the possible TFs. Each array has the indices for ktia, ktan, ktin, ktni.
    indicesaf is a list with the indices for the positions that contain the factors to change affinity.
    TFidx is the index of the TF parameters in indicesP
    afidx is the index of the affinity: 0: WT, 1: 5X, 2: 7X
    """
    
    
    
    #print(indicesaf)
    ifcb,ifcu=indicesaf #indices of scale factor for binding and unbinding. Will be None if that is assumed not to change
    idxsb,idxsu=indicesbinding 
    parsbinding=np.zeros(6)
    if afidx>0 and ifcb[0] is not None:
        if afidx==1:
            factor=pars[ifcb[0]] #if afidx=0, nothing, if afidx=1, first factor, and if afidx=2, second factor. 
        else:
            factor=pars[ifcb[0]]*pars[ifcb[1]] #the factor for 5X is multiplied by another factor >1, so that for sure it will be greater 
    else:
        factor=1
    parsbinding[::2]=pars[idxsb]/factor #factor is >=1 so reduced binding with mutation
    if afidx>0 and ifcu[0] is not None:
        if afidx==1:
            
            factor=pars[ifcu[0]] #if afidx=0, nothing, if afidx=1, first factor, and if afidx=2, second factor
        else:
            factor=pars[ifcu[0]]*pars[ifcu[1]]
    else:
        factor=1
    parsbinding[1::2]=pars[idxsu]*factor #greater unbinding than WT if factor > 1
    
    idxsP=indicesP[TFidx+1] #TFidx 0 is basal, so actual TFs start after that
    parsP=pars[idxsP]
    #print(parsP,parsbinding)
    return [parsP,parsbinding]

def get_m_model(pars,funcss=None,funcgetpars=None,argsgetpars=None,nTFs=6,naf=3,indicesP=None,**kwargs):
    """Return the matrix of foldchanges from the model. 
    Pars are the parameters to be optimized.
    funcss is the function to get the ss from parameters.
    kwargs: indicesP, indicesaf,indicesbinding
    """
    
    array1=np.array([1])
    array0=np.array([0])
    mat=np.zeros((naf*nTFs+1,naf*nTFs))
    pars_Pbasal=pars[indicesP[0]]
    parset0=np.hstack((pars_Pbasal,pars_Pbasal,pars_Pbasal,np.ones(6),np.ones(6))) #need ones for binding and unbinding even if it does nothin
    ss0=funcss(parset0,array0,0)
    #print('ss0',ss0)
    for n1 in range(nTFs):
        for n1_ in range(naf):
            kt1,b1=funcgetpars(pars,indicesP=indicesP,TFidx=n1,afidx=n1_,**kwargs)
            
            for n2 in range(n1,nTFs):
                for n2_ in range(naf):
                    kt2,b2=funcgetpars(pars,indicesP=indicesP,TFidx=n2,afidx=n2,**kwargs)
                    parset=np.hstack((pars_Pbasal,kt1,kt2,b1,b2))
                    ss=funcss(parset,array1,1)
                    mat[naf*n1+n1_+1,naf*n2+n2_]=ss/ss0
                    
                    #print(parset)
                    #if False:#notice that for this to work names_short has to be defined
                    #for x in parset:
                    #    print(x,end=',')
                    #print()
                    #    print('1',names_short[n1],n1_,kt1,b1)
                    #    print('2',names_short[n2],n2_,kt2,b2)
                    #print('\nmat position',3*n1+n1_+1,3*n2+n2_)
                    #print(ss, ss/ss0)
                    #    print('-------------------------')
                    
                    
    
        
    for n1 in range(nTFs):
        for n1_ in range(naf):
            kt1,b1=funcgetpars(pars,indicesP=indicesP,TFidx=n1,afidx=n1_,**kwargs)
            parset=np.hstack((pars_Pbasal,kt1,kt1,b1,b1))
            
            ss=funcss(parset,array0,1)
            #print(ss)
            mat[0,naf*n1+n1_]=ss/ss0
                                 
    return mat