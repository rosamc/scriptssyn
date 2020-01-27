import numpy as np
import pandas as pd
import re

parenthesis=re.compile(r'(\(.*\))')

def get_exp_matrix(df,TFnames,affinities,matnames=False,min_=False,max_=False,std=False):
    cGFP=['GFPfa1','GFPfa2','GFPfa3']
    naf=len(affinities)
    nTFs=len(TFnames)
    mat_fc=np.zeros((naf*nTFs+1,naf*nTFs)) #additional row on top is to compare to nothing on TF2
    if min_:
    	mat_fcmin=np.zeros_like(mat_fc)
    if max_:
    	mat_fcmax=np.zeros_like(mat_fc)
    if std:
    	mat_fcstd=np.zeros_like(mat_fc)

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
                        vals1=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af1)&(df['affinity2']==af2)][cGFP]
                        vals1=vals1.values

                        vals2=df[(df['activator1']==TF2)&(df['activator2']==TF1)&(df['affinity1']==af2)&(df['affinity2']==af1)][cGFP]
                        vals2=vals2.values

                        if len(vals1)>0:
                            vals=vals1
                        else:
                            vals=vals2

                        #print(TF1, af1, TF2, af2, vals2.values)

                        avGFP=np.nanmean(vals)
                        #print(TF1,af1,TF2,af2,vals.values)
                        r=naf*n1+n1_+1
                        col=naf*n2+n2_
                        mat_fc[r,col]=avGFP
                        if min_:
                            mat_fcmin[r,col]=np.nanmin(vals)
                        if max_:
                            mat_fcmax[r,col]=np.nanmax(vals)
                        if std:
                            mat_fcstd[r,col]=np.std(vals)
                        if matnames:
                            mat_names[r,col]=names_short[n2]+affinities_short[n2_]+'\n'+names_short[n1]+affinities_short[n1_]
                    else: #for the cases where the two TFs are the same, but affinities are different, there are two different entries in the table. Take the average, and only when n2_>n1_:
                        if n2_>n1_: 
                            vals1=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af1)&(df['affinity2']==af2)][cGFP]
                            #print(TF1, af1, TF2, af2, vals.values)
                            #avGFP1=np.nanmean(vals1.values)
                            vals2=df[(df['activator1']==TF1)&(df['activator2']==TF2)&(df['affinity1']==af2)&(df['affinity2']==af1)][cGFP]
                            #print(TF1, af1, TF2, af2, vals.values)
                            #avGFP2=np.nanmean(vals2.values)
                            vals=np.concatenate([vals1.values,vals2.values])
                            avGFP=np.nanmean(vals)
                            r=naf*n1+n1_+1
                            col=naf*n2+n2_
                            mat_fc[r,col]=avGFP
                            if min_:
                                mat_fcmin[r,col]=np.nanmin(vals)
                            if max_:
                                mat_fcmax[r,col]=np.nanmax(vals)
                            if std:
                                mat_fcstd[r,col]=np.std(vals)
                            if matnames:
                                mat_names[r,col]=names_short[n2]+affinities_short[n2_]+'\n'+names_short[n1]+affinities_short[n1_]
                                #print(mat_names[naf*n1+n1_+1,naf*n2+n2_],avGFP1,avGFP2,mat_fc[naf*n1+n1_+1,naf*n2+n2_])

    for n1 in range(nTFs):
        TF1=TFnames[n1]
        for n1_ in range(naf):
            af1=affinities[n1_]
            vals=df[(df['activator1']==TF1)&(df['activator2']=='-')&(df['affinity1']==af1)][cGFP].values
            avGFP=np.nanmean(vals)
            #print(TF1,af1,TF2,af2,vals.values)
            r=0
            col=naf*n1+n1_
            mat_fc[r,col]=avGFP
            if min_:
                mat_fcmin[r,col]=np.nanmin(vals)
            if max_:
                mat_fcmax[r,col]=np.nanmax(vals)
            if std:
                mat_fcstd[r,col]=np.std(vals)
            if matnames:
                mat_names[r,col]=names_short[n1]+' '+affinities_short[n1_]
            #print(TF1,af1,vals)
    returnlist=[mat_fc]
    if matnames:
        returnlist.append(mat_names)
    if min_:
    	returnlist.append(mat_fcmin)
    if max_:
    	returnlist.append(mat_fcmax)
    if std:
    	returnlist.append(mat_fcstd)
    return returnlist



def get_parameters_TF_v1(pars,indicesbinding=None,indicesP=None,indicesaf=None,TFidx=None,afidx=None,fixedbasal=False,fixedpars=None):
    """pars is the array of parameters to be optimized.
    indicesbinding is a list of 2 arrays: one with the indices for kbXa, kbXi, kbXn, and the other with the indices for kuXa, kuXi, kuXn
    indicesP is a list of arrays for each of the possible TFs. Each array has the indices for ktia, ktan, ktin, ktni.
    indicesaf is a list with the indices for the positions that contain the factors to change affinity.
    TFidx is the index of the TF parameters in indicesP
    afidx is the index of the affinity: 0: WT, 1: 5X, 2: 7X
    fixedpars=None does nothing but is left for backward compatibility.
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
    
    if fixedbasal is False:
        idxsP=indicesP[TFidx+1] #TFidx 0 is basal, so actual TFs start after that
    else:
        idxsP=indicesP[TFidx]
    
    parsP=pars[idxsP]
    
    #print(parsP,parsbinding)
    return [parsP,parsbinding]

def get_parameters_TF_v2(pars,nbasalcycle=None,ncycleperTF=None,bindingperTF=None,TFidx=None,indicesaf=None,afidx=None):
    #print(indicesaf)
    ifcb,ifcu=indicesaf #indices of scale factor for binding and unbinding. Will be None if that is assumed not to change
    indicesb=get_binding_indices(TFidx,nbasalcycle=nbasalcycle,ncycleperTF=ncycleperTF,bindingperTF=bindingperTF,mutations=np.sum(indicesaf!=None))
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


def get_m_model(pars,fixedbasal=False,pars_Pbasal=[],funcss=None,funcgetpars=None,nTFs=6,affinities=['WT','5X','7X'],indicesP=None,fixedpars=None,**kwargs):
    """Return the matrix of foldchanges from the model. 
    Pars are the parameters to be optimized.
    funcss is the function to get the ss from parameters.
    kwargs: indicesP, indicesaf,indicesbinding
    fixedbasal: True if the basal parameter of the Pol cycle are fixed
    """
    if fixedbasal is True and len(pars_Pbasal)==0:
        print('No pars_Pbasal given. Exiting...')
    if fixedbasal not in [True,False]:
        print("fixedbasal should either be True or False. Exiting...")

    array1=np.array([1])
    array0=np.array([0])
   
    if fixedbasal is False:
        if 'v1' in funcgetpars.__name__:
            pars_Pbasal=pars[indicesP[0]]
        elif 'v2' in funcgetpars.__name__:
            pars_Pbasal=pars[indicesP]
    #otherwise 
    
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
                kt1,b1=funcgetpars(pars,indicesP=indicesP,fixedbasal=fixedbasal,TFidx=n1,afidx=nafc[n1_],**kwargs)
            elif 'v2' in funcgetpars.__name__:
                kt1,b1=funcgetpars(pars,TFidx=n1,afidx=nafc[n1_],**kwargs)
            else:
                print('Incorrect funcgetpars. Exiting.')
                raise ValueError
            
            for n2 in range(n1,nTFs):
                for n2_ in range(naf):
                    if not (n1==n2 and n2_<n1_):
                        if 'v1' in funcgetpars.__name__:
                            kt2,b2=funcgetpars(pars,indicesP=indicesP,fixedbasal=fixedbasal,TFidx=n2,afidx=nafc[n2_],**kwargs)
                        elif 'v2' in funcgetpars.__name__:
                            kt2,b2=funcgetpars(pars,TFidx=n2,afidx=nafc[n2_],**kwargs)
                        parset=np.hstack((pars_Pbasal,kt1,kt2,b2,b1))
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
                kt1,b1=funcgetpars(pars,indicesP=indicesP,fixedbasal=fixedbasal,TFidx=n1,afidx=nafc[n1_],**kwargs)
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
def get_pb(bindingperTF):
	pbon=0
	pboff=0
	on,off=bindingperTF
	if on=='onr':
	    pbon+=1
	elif on=='3onr':
	    pbon+=3
	else:
	    print('incorrect number of binding parameters per TF.')
	    raise ValueError 
	if off=='offr':
	    pboff+=1
	elif off=='3offr':
	    pboff+=3
	else:
	    print('incorrect number of unbinding parameters per TF')
	    raise 
	return [pbon,pboff]
def get_binding_indices(TFidx,nbasalcycle=None,ncycleperTF=None,bindingperTF=None,mutations=None):
    """Given an array of parameters, returns the indices of those parameters corresponding to binding and unbinding rates. 
    TFidx: index of the TF for which binding parameter indices have to be retrieved.
    nbasalcycle: number of parameters corresponding to the pol cycle in the basal state.
    ncycleperTF: number of parameters corresponding to the pol cycle when the TF is bound
    bindingperTF: list were first position corresponds to onrate, second position corresponds to offrate. 
                if 'onr': means only one onrate per TF. if '3onr': 3 onrates
                if 'offr': means only one offrate. if '3offr': 3 offrates
    mutations: number of parameters at the beginning of the parameter set array corresponding to the effects of the mutations.
    """
    pbon,pboff=get_pb(binDingperTF)
    nbindingperTF=pbon+pboff
    i0=mutations+nbasalcycle+TFidx*(ncycleperTF+nbindingperTF)+ncycleperTF
    i1=i0+nbindingperTF
    return [np.arange(i0,i1),pbon,pboff]

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


def get_indices(TFnames,suff1,suff2,nPcycle=4,returnparnames=False,fixedbasal=False,affinities=['ZF(WT)','ZF(5X)','ZF(7X)']):
    if not 'ZF(WT)' in affinities:
        print("It doesn't make sense to test an affinity mutation without the WT. A reference is required. Exiting...")
        raise ValueError

    names=[]
    if suff2=='_mutu':
        if len(affinities)==3:
            indices_af=np.array([[None,None],[0,1]])
            names.extend(['fu5X','fu7X'])
        elif len(affinities)==2:
            indices_af=np.array([[None],[0]])
            if 'ZF(5X)' in affinities:
                names.extend(['fu5X'])
            elif 'ZF(7X)' in affinities:
                names.extend(['fu7X'])
            else:
                print("Unrecognized affinity.", affinities)
                raise ValueError
        elif len(affinities)==1:
            print("suff _mutu doesn't make sense with one affinity only. Exiting")
            raise ValueError
    elif suff2=='_mutbu':
        if len(affinities)==3:
            indices_af=np.array([[0,1],[2,3]]) #0, 1 positions in parameter array correspond to factor for change in affinity for 5X, and additional factor for change in affinity for 7X
            names.extend(['fb5X','fb7X','fu5X','fu7X'])
        elif len(affinities)==2:
            indices_af=np.array([[0],[1]])
            if 'ZF(5X)' in affinities:
                names.extend(['fb5X','fu5X'])
            elif 'ZF(7X)' in affinities:
                names.extend(['fb7X','fu7X'])
            else:
                print("Unrecognized affinity.", affinities)
                raise ValueError
        elif len(affinities)==1:
            print("suff _mutbu doesn't make sense with one affinity only. Exiting")
            raise ValueError
    else:
        print('wrong suff2',suff2) 
        raise ValueError

    i0=indices_af[1][-1]+1
    if suff1=='_sameb':
        i1=i0+1
        i2=i1+1
        indices_binding=[np.array([i0,i0,i0]),np.array([i1,i1,i1])] #assume binding and unbinding are the same for all states
        names.extend(['kb','ku'])
    elif suff1=='_difkb':
        i1=i0+1
        i2=i1+3
        indices_binding=[np.array([i0,i0,i0]),np.array([i1,i1+1,i1+2])] #assume unbinding is different
        names.extend(['kb','ku1','ku2','ku3'])
    elif suff1=='_difkbku':
        i1=i0+3
        i2=i1+3
        indices_binding=[np.array([i0,i0+1,i0+2]),np.array([i1,i1+1,i1+2])] #assume binding and unbinding are different
        names.extend(['kb1','kb2','kb3','ku1','ku2','ku3'])
    else:
        print("Wrong suff1.")
        raise ValueError
        
    if nPcycle==3:
        Pk=['kia','kan','kni']
    else:
        Pk=['kia','kan','kin','kni']
    if not fixedbasal:
        indices_basal=np.arange(i2,i2+nPcycle)
        for k in Pk:
            names.append(k+'0')
        
        all_indices=[indices_basal]
        i=indices_basal[-1]+1
    else:
        all_indices=[]
        i=indices_binding[-1][-1]+1
    
    for TF in TFnames:
        indices=[]
        for p in range(nPcycle):
            indices.append(i)
            i+=1
        all_indices.append(np.array(indices))
        for k in Pk:
            names.append(k+TF)
    #iprint('npars:',i)

    npars=i
    if returnparnames:
        returnlist=[indices_af,indices_binding,all_indices,npars,names]
    else:
        returnlist=[indices_af,indices_binding,all_indices,npars]
    #print(npars)
    return returnlist

