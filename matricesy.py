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
                    if n1!=n2 or (n1==n2 and n1_==n2_):
                        vals=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af1)&(df['affinity2']==af2)][cGFP]
                        #print(TF1, af1, TF2, af2, vals.values)
                        avGFP=np.nanmean(vals.values)
                        #print(TF1,af1,TF2,af2,vals.values)
                        mat_fc[naf*n1+n1_+1,naf*n2+n2_]=avGFP
                        if matnames:
                            mat_names[naf*n1+n1_+1,naf*n2+n2_]=names_short[n2]+affinities_short[n2_]+'\n'+names_short[n1]+affinities_short[n1_]
                    else: #for the cases where the two TFs are the same, but affinities are different, there are two different entries in the table. Take the average, and only when n2_>n1_:
                        if n2_>n1_: 
                            vals1=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af1)&(df['affinity2']==af2)][cGFP]
                            #print(TF1, af1, TF2, af2, vals.values)
                            avGFP1=np.nanmean(vals1.values)
                            vals2=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af2)&(df['affinity2']==af1)][cGFP]
                            #print(TF1, af1, TF2, af2, vals.values)
                            avGFP2=np.nanmean(vals2.values)
                            mat_fc[naf*n1+n1_+1,naf*n2+n2_]=(avGFP1+avGFP2)/2
                            if matnames:
                                mat_names[naf*n1+n1_+1,naf*n2+n2_]=names_short[n2]+affinities_short[n2_]+'\n'+names_short[n1]+affinities_short[n1_]
                                #print(mat_names[naf*n1+n1_+1,naf*n2+n2_],avGFP1,avGFP2,mat_fc[naf*n1+n1_+1,naf*n2+n2_])

    for n1 in range(nTFs):
        TF1=TFnames[n1]
        for n1_ in range(naf):
            af1=affinities[n1_]
            vals=df[(df['activator1']==TF1)&(df['activator2']=='-')&(df['affinity1']==af1)][cGFP]
            mat_fc[0,naf*n1+n1_]=np.nanmean(vals.values)
            if matnames:
                mat_names[0,naf*n1+n1_]=names_short[n1]+' '+affinities_short[n1_]
            #print(TF1,af1,vals)
    if matnames:
        return [mat_fc,mat_names]
    else:
        return mat_fc

def get_parameters_TF_v1(pars,indicesbinding=None,indicesP=None,indicesaf=None,TFidx=None,afidx=None,fixedpars=None):
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
    if fixedpars is None:
        parsbinding[::2]=pars[idxsb]/factor #factor is >=1 so reduced binding with mutation
    else:
        #if 'kb' in fixedpars:
        parsbinding[::2]=fixedpars[idxsb]/factor
        #else:
        #    parsbinding[::2]=pars[idxsb]/factor
    if afidx>0 and ifcu[0] is not None:
        if afidx==1:
            
            factor=pars[ifcu[0]] #if afidx=0, nothing, if afidx=1, first factor, and if afidx=2, second factor
        else:
            factor=pars[ifcu[0]]*pars[ifcu[1]]
    else:
        factor=1
    
    if fixedpars is None:
        parsbinding[1::2]=pars[idxsu]*factor #greater unbinding than WT if factor > 1
    else:
        #if 'ku' in fixedpars:
        parsbinding[1::2]=fixedpars[idxsu]*factor
        #else:
        #    parsbinding[1::2]=pars[idxsu]*factor
    idxsP=indicesP[TFidx+1] #TFidx 0 is basal, so actual TFs start after that
    if fixedpars is None:
        parsP=pars[idxsP]
    else:
        parsP=fixedpars[idxsP]
    #print(parsP,parsbinding)
    return [parsP,parsbinding]

def get_parameters_TF_v2(pars,nbasalcycle=None,ncycleperTF=None,nbindingperTF=None,TFidx=None,indicesaf=None,afidx=None):
    #print(indicesaf)
    ifcb,ifcu=indicesaf #indices of scale factor for binding and unbinding. Will be None if that is assumed not to change
    indicesb=get_binding_indices(TFidx,nbasalcycle=nbasalcycle,ncycleperTF=ncycleperTF,nbindingperTF=nbindingperTF,mutations=np.sum(indicesaf!=None))
    idxsb=indicesb[0::2]
    idxsu=indicesb[1::2]
    #idxsb,idxsu=indicesbinding 
    parsbinding=np.zeros(6)
    if afidx>0 and ifcb[0] is not None:
        if afidx==1:
            factor=pars[ifcb[0]] #if afidx=0, nothing (WT), if afidx=1 (5X), first factor, and if afidx=2, second factor. 
        else:
            factor=pars[ifcb[0]]*pars[ifcb[1]] #the factor for 5X is multiplied by another factor >1, so that for sure it will be greater 
    else:
        factor=1
    #if fixedpars is None:
    parsbinding[::2]=pars[idxsb]/factor #factor is >=1 so reduced binding with mutation
    
    if afidx>0 and ifcu[0] is not None:
        if afidx==1:
            
            factor=pars[ifcu[0]] #if afidx=0, nothing, if afidx=1, first factor, and if afidx=2, second factor
        else:
            factor=pars[ifcu[0]]*pars[ifcu[1]]
    else:
        factor=1
    
    parsbinding[1::2]=pars[idxsu]*factor #greater unbinding than WT if factor > 1
    
    idxsP=get_cycle_indices(TFidx,nbasalcycle=nbasalcycle,ncycleperTF=ncycleperTF,nbindingperTF=nbindingperTF,mutations=np.sum(indicesaf!=None))
    
    #print(parsP,parsbinding)
    return [pars[idxsP],parsbinding]


def get_m_model(pars,fixedpars=None,funcss=None,funcgetpars=None,nTFs=6,affinities=['WT','5X','7X'],indicesP=None,**kwargs):
    """Return the matrix of foldchanges from the model. 
    Pars are the parameters to be optimized.
    funcss is the function to get the ss from parameters.
    kwargs: indicesP, indicesaf,indicesbinding
    fixedpars: array of parameter values not to explore
    """
    
    array1=np.array([1])
    array0=np.array([0])
   
    if fixedpars is None:
        if 'v1' in funcgetpars.__name__:
            pars_Pbasal=pars[indicesP[0]]
        elif 'v2' in funcgetpars.__name__:
            pars_Pbasal=pars[indicesP]
    else: #assume all parameters are fixed except those for the factors of the mutations
        if 'v1' in funcgetpars.__name__:
            pars_Pbasal=fixedpars[indicesP[0]]
        elif 'v2' in funcgetpars.__name__:
            pars_Pbasal=fixedpars[indicesP]
    
    parset0=np.hstack((pars_Pbasal,pars_Pbasal,pars_Pbasal,np.ones(6),np.ones(6))) #need ones for binding and unbinding even if it does nothin
    ss0=funcss(parset0,array0,0)
    #print('ss0',ss0)
    naf=len(affinities)

    if not 'WT' in affinities:
        nafc=np.arange(naf)+1 #the first affinity is already a mutant one. 
    else:
        nafc=np.arange(naf)


    mat=np.zeros((naf*nTFs+1,naf*nTFs))

    for n1 in range(nTFs):
        for n1_ in range(naf):
            if 'v1' in funcgetpars.__name__:
                kt1,b1=funcgetpars(pars,indicesP=indicesP,fixedpars=fixedpars,TFidx=n1,afidx=nafc[n1_],**kwargs)
            elif 'v2' in funcgetpars.__name__:
                kt1,b1=funcgetpars(pars,TFidx=n1,afidx=nafc[n1_],**kwargs)
            else:
                print('Incorrect funcgetpars. Exiting.')
                raise ValueError
            
            for n2 in range(n1,nTFs):
                for n2_ in range(naf):
                    if not (n1==n2 and n2_<n1_):
                        if 'v1' in funcgetpars.__name__:
                            kt2,b2=funcgetpars(pars,indicesP=indicesP,fixedpars=fixedpars,TFidx=n2,afidx=nafc[n2_],**kwargs)
                        elif 'v2' in funcgetpars.__name__:
                            kt2,b2=funcgetpars(pars,TFidx=n2,afidx=nafc[n2_],**kwargs)
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
            if 'v1' in funcgetpars.__name__:
                kt1,b1=funcgetpars(pars,indicesP=indicesP,fixedpars=fixedpars,TFidx=n1,afidx=nafc[n1_],**kwargs)
            elif 'v2' in funcgetpars.__name__:
                kt1,b1=funcgetpars(pars,TFidx=n1,afidx=nafc[n1_],**kwargs)
            parset=np.hstack((pars_Pbasal,kt1,kt1,b1,b1))
            
            ss=funcss(parset,array0,1)
            #print(ss)
            mat[0,naf*n1+n1_]=ss/ss0
                                 
    return mat

def make_comparison_matrix(mat1,mat2,f=10):
    rows=len(mat1)
    cols=len(mat1[0])
    newr=f*rows
    newcols=f*cols
    newm=np.zeros((newr,newcols))
    for r in range(rows):
        for c in range(cols):
            val1=mat1[r,c]
            val2=mat2[r,c]
            newcell=newm[f*r:f*r+f,f*c:f*c+f]
            for r_ in range(f):
                for c_ in range(r_,f):
                    newcell[r_,c_]=val1
                for c_ in range(0,r_):
                    newcell[r_,c_]=val2
    return newm

def get_binding_indices(TFidx,nbasalcycle=None,ncycleperTF=None,nbindingperTF=None,mutations=None):
    """Given an array of parameters, returns the indices of those parameters corresponding to binding and unbinding rates. 
    TFidx: index of the TF for which binding parameter indices have to be retrieved.
    basalcycle: number of parameters corresponding to the pol cycle in the basal state.
    cycleperTF: number of parameters corresponding to the pol cycle when the TF is bound
    bindingperTF: number of parameters corresponding to binding and unbinding rates for each TF
    mutations: number of parameters at the beginning of the parameter set array corresponding to the effects of the mutations.
    """
    
    i0=mutations+nbasalcycle+TFidx*(ncycleperTF+nbindingperTF)+ncycleperTF
    i1=i0+nbindingperTF
    return np.arange(i0,i1)

def get_cycle_indices(TFidx, nbasalcycle=None,ncycleperTF=None,nbindingperTF=None,mutations=None ):
    """"Given an array of parameters, returns the indices of those parameters corresponding to changes over the cycle.
    """
    i0=mutations+nbasalcycle+TFidx*(ncycleperTF+nbindingperTF)
    i1=i0+ncycleperTF
    return np.arange(i0,i1)

def get_initial_parset(bounds,logb=False,seed=1):
    npars=len(bounds)
    pars=np.zeros(npars)
    np.random.seed(seed)
    for i in range(npars):
        if not logb:
            pars[i]=10**np.random.uniform(np.log10(bounds[i][0]),np.log10(bounds[i][1]))
        else:
            pars[i]=10**np.random.uniform(bounds[i][0],bounds[i][1])
    return pars



