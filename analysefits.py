import numpy as np
import matricesy
import matplotlib.pyplot as plt
import os, re
import pandas as pd

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


def get_results(fldr,yexp,njobs=0,costf=None,kwargs=None,basename='out',jid=None):
    parsets_=[]
    costs_=[]
    Ts_=[] 
    stepsizes_=[]
    seeds_=[]
    for i in range(1,njobs+1):
        success=False
        if jid is not None:
            outname=os.path.join(os.path.split(fldr)[0],jid+'_%d.out'%i)
            for line in open(outname,'r').readlines():
                if 'success: True' in line:
                    #print(line)
                    success=True
        else:
            success=True

        if success:

            fname=os.path.join(fldr,'%s_%d.out'%(basename,i))
            if os.path.isfile(fname):
                outf=open(fname)
                lines=outf.readlines()
                if len(lines)>1:
                    l1=lines[0]
                    l2=lines[1]
                    l1=l1.strip().replace('#','')
                    T,stepsize,seed=l1.split(',')
                    parset=np.array(list(map(float,l2.strip().strip(',').split(','))))
                    c=costf(parset,yexp,kwargs)
                    costs_.append(c)
                    parsets_.append(parset)
                    Ts_.append(float(T))
                    stepsizes_.append(float(stepsize))
                    seeds_.append(int(seed))
    stepsizes_=np.array(stepsizes_)
    seeds_=np.array(seeds_)
    Ts_=np.array(Ts_)
    costs_=np.array(costs_)
    parsets_=np.vstack(parsets_)
    out={'Ts':Ts_,'seeds':seeds_,'stepsizes':stepsizes_,'costs':costs_,'parsets':parsets_}
    return out
def plot_fitresult(mat_fc,parset,mat_names,cmap1=plt.cm.BuGn,cmap2=plt.cm.Blues,kwargs=None,axes=None,returnaxes=False):

    
    maskexp=np.ma.masked_array(mat_fc,mask=(mat_fc==0))
    if axes is None:
        fig,axes=plt.subplots(1,2,figsize=(20,6))
    
    for x in parset:
        print(x,end=',')
    print()
    m=matricesy.get_m_model(parset,**kwargs)
    maskmodel=np.ma.masked_array(m,mask=(m==0))
    dif=(maskmodel-maskexp)/maskexp
    difup=np.ma.masked_array(dif,mask=dif<0)
    difdown=np.ma.masked_array(dif,mask=dif>0)
    
    ax=axes[0]
    im1=ax.imshow(difup,aspect='auto',cmap=plt.cm.Blues)
    im2=ax.imshow(difdown,aspect='auto',cmap=plt.cm.Greens_r)
    ax.grid()
    plt.colorbar(im1,ax=ax,label='(fc model-fc exp)/(fc exp)') #,extend='both
    plt.colorbar(im2,ax=ax,label='(fc model-fc exp)/(fc exp)') #,extend='both')
        #for row in range(len(mat_names)):
    #    for col in range(len(mat_names[0])):
    #        text=mat_names[row,col]
    #        #ax.text(col-0.5,row+0.5,text,fontsize=10,color='k')
    ax.set_xticks(range(len(mat_names[0])))
    ax.set_xticklabels([x.replace(' ','') for x in mat_names[0]],rotation=90)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(mat_names)))
    ax.set_yticklabels(['-']+list(mat_names[0]))
    ax=axes[1]
    #ax=axes[1,1]
    f=10
    mboth=make_comparison_matrix(mat_fc,m,f=10)
    maskboth=np.ma.masked_array(mboth,mask=(mboth==0))
    im=ax.imshow(maskboth,aspect='auto',cmap=cmap2)
    cb=plt.colorbar(im,ax=ax,label='fold change with respect to basal')
    xticks=range(len(mat_names[0])*f)[int(f/2)::f]
    ax.set_xticks(xticks)
    ax.set_xticklabels(mat_names[0],rotation=90)
    ax.xaxis.tick_top()
    yticks=range(len(mat_names)*f)[int(f/2)::f]
    ax.set_yticks(yticks)
    ax.set_yticklabels(['-']+list(mat_names[0]))
    fhalf=int(f/2)
    for x in xticks:
        ax.axvline(x,color='grey',linewidth=1)
    for y in yticks:
        ax.axhline(y,color='grey',linewidth=1)
    #for ax in [axes[0,0],axes[0,1],axes[1,0]]:
    if returnaxes:
        return [axes,m,f]
    else:
        plt.tight_layout()
        plt.show()

def plot_comparison_lines(exp_,model_,labels,ax=None,min_=[],max_=[],linelabel='model',plotexp=True,colormodel='k'):
    """max_ and min_ should be two lists/arrays with min and max at a given x."""
    if ax is None:
        fig,ax=plt.subplots(1,1,figsize=(18,4))
        show=True
    else:
        show=False
    if len(min_)!=len(max_):
        print('to plot max and min please specify max and min values for each datapoint')
        raise ValueError
    #if min_ is None:
    if plotexp:
	    ax.plot(range(len(exp_)),exp_,color='r',marker='o',label='exp')
	    if len(min_)>0:
	        min_=-(np.asarray(min_)-np.asarray(exp_))
	        max_=np.asarray(max_)-np.asarray(exp_)
	        ax.errorbar(range(len(exp_)),exp_,yerr=np.vstack((min_,max_)),color='r')
    if linelabel is None:
    	linelabel='model'
    ax.plot(range(len(model_)),model_,color=colormodel,marker='o',label=linelabel)
    ax.set_xticks(np.arange(len(model_)))
    ax.set_xticklabels(labels,rotation='90')
    ax.set_ylabel('fold change')
    
    ax.legend()
    if show:
        plt.show()
    else:
        return ax

def plot_result_matricesandlines(dfexp,parset,kwargs=None,TFnames=[],affinities=[],plotmats=True,plotlines=True,ncutoff=100,returnlines=False):
    
    mat_fc,mat_names,mat_expmin,mat_expmax=matricesy.get_exp_matrix(dfexp,TFnames,affinities,matnames=True,min_=True,max_=True)
    mat_fc_=mat_fc
    #if (plotmats is False) and ((plotlines is True) or (returnlines is True)): 
    m=matricesy.get_m_model(parset,**kwargs)
    if plotmats:
        shape=(2,2)
        fig=plt.figure(figsize=(12,5))         
        ax1=plt.subplot2grid(shape,(0,0),rowspan=2,colspan=1,fig=fig)
        ax2=plt.subplot2grid(shape,(0,1),rowspan=2,colspan=1,fig=fig)
        axes=[ax1,ax2]
        axes,m,f=plot_fitresult(mat_fc,parset,mat_names,cmap1=plt.cm.BuGn,cmap2=plt.cm.Greys,kwargs=kwargs,axes=axes,returnaxes=True)
        

        naf=len(affinities)

        for r in range(1,len(mat_fc_))[::naf]:
            for c_ in range(r-1,len(mat_fc_[0]))[::naf]:
                r0=r
                r1=r0+naf
                c0=c_
                c1=c0+naf
                submat1=mat_fc_[r0:r1,c0:c1]
                submat2=m[r0:r1,c0:c1]
                argmax1=np.argmax(submat1)
                argmax2=np.argmax(submat2)
                #print(r0,r1,c0,c1,argmax1,argmax2)
                axes[1].scatter(f*(c0+argmax1%naf)+8,f*(r0+argmax1//naf)+2, color='r')
                axes[1].scatter(f*(c0+argmax2%naf)+2,f*(r0+argmax2//naf)+8,color='b')
        for i in range(len(m[0])*f)[::f][::naf]:
            axes[1].axvline(x=i,color='pink')
        for i in range(f,len(m)*f)[::f][::naf]:
            axes[1].axhline(y=i,color='pink')
        for i in range(len(m[0]))[::naf]:
            axes[0].axvline(x=i-0.5,color='pink')
        for i in range(1,len(m[0]))[::naf]:
            axes[0].axhline(y=i-0.5,color='pink')
        axes[1].set_ylim(len(m)*f,0)

        plt.tight_layout()
        plt.show()
    #m=matricesy.get_m_model(parset,**kwargs)
    
    if plotlines or returnlines:
        model_=[]
        exp_=[]
        exp_min=[]
        exp_max=[]
        labels=[]
        rrange=np.concatenate((np.array([0]),np.arange(1,len(mat_fc_))[::3]))
        for r in rrange:
            if r>0:
                for c_ in range(r-1,len(mat_fc_[0]))[::3]:
                    r0=r
                    r1=r0+3
                    c0=c_
                    c1=c0+3
                    submat=mat_fc_[r0:r1,c0:c1]
                    submatmin=mat_expmin[r0:r1,c0:c1]
                    submatmax=mat_expmax[r0:r1,c0:c1]
                    submatmod=m[r0:r1,c0:c1]
                    submatn=mat_names[r0:r1,c0:c1]
                    ncells=len(submat.flatten())
                    for i in range(ncells):
                        x_=submat.flatten()[i]
                        if x_>0:
                            exp_.append(x_)
                            exp_min.append(submatmin.flatten()[i])
                            exp_max.append(submatmax.flatten()[i])
                            model_.append(submatmod.flatten()[i])
                            labels.append(submatn.flatten()[i].replace('\n','-'))
            else:
                submat=mat_fc_[r]
                submatmin=mat_expmin[r]
                submatmax=mat_expmax[r]
                submatmod=m[r]
                submatn=mat_names[r]
                ncells=len(submat.flatten())
                for i in range(ncells):
                    x_=submat.flatten()[i]
                    if x_>0:
                        exp_.append(x_)
                        exp_min.append(submatmin.flatten()[i])
                        exp_max.append(submatmax.flatten()[i])
                        model_.append(submatmod.flatten()[i])
                        labels.append(submatn.flatten()[i].replace('\n','-'))
        #fig,ax=plt.subplots(1,1,figsize=(11,5))
        #print(exp_min)
        #print(exp_max)
        #ax=axes[c]
        #plt.tight_layout()
        #plt.show()
    if plotlines:
        if len(model_)<ncutoff:
            figsize=(14,3)
            nrows=3
            ncols=1
            
        else:
            figsize=(14,8)
            nrows=7
            ncols=1
            
        fig=plt.figure(figsize=figsize)   
        
        
        if len(model_)<ncutoff:
            ax=plt.subplot2grid((nrows,ncols),(0,0),rowspan=3,colspan=1,fig=fig)
            ax=plot_comparison_lines(exp_,model_,labels,ax=ax,min_=exp_min,max_=exp_max)
            #ax.set_title(title)
        else:
            ncut=int(len(exp_)/2)
            ax=plt.subplot2grid((nrows,ncols),(0,0),rowspan=3,colspan=1,fig=fig)
            ax=plot_comparison_lines(exp_[0:ncut],model_[0:ncut],labels[0:ncut],ax=ax,min_=exp_min[0:ncut],max_=exp_max[0:ncut])
            #ax.set_title(title)
            
            ax=plt.subplot2grid((nrows,ncols),(4,0),rowspan=3,colspan=1,fig=fig)
            ax=plot_comparison_lines(exp_[ncut:],model_[ncut:],labels[ncut:],ax=ax,min_=exp_min[ncut:],max_=exp_max[ncut:])

            
        plt.tight_layout()
        plt.show()
    if returnlines:
    	return [exp_,exp_min,exp_max,model_,labels]
    else:
        return 


def readf(folder,f,fixedbasal=False,nlines=4,affinities=['ZF(WT)','ZF(5X)','ZF(7X)']):
    """Reads a given output file from a fit search and returns to pandas dataframes.
    -fixedbasal: set to True if parameters for the basal polymerase cycle were fixed. 
    - nlines: integer with the number of output lines for an individual search (4 at the beginning, 5 when pars_Pbasal were fixed)"""

    outf=open(os.path.join(folder,f),'r')
    lines=outf.readlines()
    numeric_cols=['w1','w2','nPcycle','T','stepsize','interval','cost']
    cols1='suff1,suff2,TFsim,w1,w2,nPcycle,jid,T,stepsize,seed,interval,cost'.split(',')
    cols2='fb5X,fb7X,fu5X,fu7X,kba,kbi,kbn,kua,kui,kun'.split(',')
    af5X=False
    af7X=False
    if 'ZF(5X)' in af and 'ZF(7X)' in af:
        af5X=True
        af7X=True
    elif 'ZF(5X)' in af:
        af5X=True
    elif 'ZF(7X)' in af:
        af7X=True
    
    cols3=['kia0','kan0','kin0','kni0','kia','kan','kin','kni']
    dfind = pd.DataFrame(columns=cols1+cols2+cols3+['fullparset','TF'])
    columnsfull=cols1+['fullparset','pars_Pbasal','TF']

    dffull = pd.DataFrame(columns=columnsfull) #TF is used to be able to subset df and only plot results were a given TF appears. So for combinations of 2 or more, data will be duplicated
    #print(dfind.columns)
    #print(dffull.columns)
    iloc=0
    nlines=int(nlines)
    
    for l0 in range(len(lines))[::nlines]:
        if nlines==5:
            l1,l2,l3,l4,l5=lines[l0:l0+nlines]
        else:
            l1,l2,l3,l4=lines[l0:l0+nlines]
        flag=True
        
        if not 'c1' in l1:
            print('wrong line 1', l1)
            flag=False
            nPcycle=None
        else:
            l1=l1.strip()
            _,l1=l1.split(':')
            l1s=[x.split('=')[1] for x in l1.split(',')]
            suff1,suff2,TFsim,w1,w2,nPcycle,jid=l1s
            l1s=np.asarray(l1s)
        
        if not 'c2' in l2:
            print('wrong line 2', l2)
            flag=False
        else:
            l2=l2.strip()
            _,l2=l2.split(':')
            l2s=[x.split('=')[1] for x in l2.split(',')]
            l2s=np.asarray(l2s)
        if nlines==5:
            if not 'c3' in l3:
                print("wrong line 3", l3)
                flag=False
            else:
                l3=l3.strip()
                _,l3=l3.split(':')
                l3=l3.split("=")[1]
                pars_Pbasal_=[float(x) for x in l3.strip('[').strip(']').split(',')]
                pars_Pbasal_=np.array(pars_Pbasal_)
                #print('pars Pbasal is', l3, pars_Pbasal_)
                if pars_Pbasal_[-1]==0: #this was a mistake when I first run this
                    flag=False
                    #print('false flag')
            lcost=l4
            lpars=l5
        else:
            lcost=l3
            lpars=l4
        if not 'cost' in lcost:
            print('wrong line cost', lcost)
            flag=False
        else:
            _,cost=lcost.strip().split(':')
            cost=np.array([float(cost)])
        
        if len(lpars)==0:    
            print('empty lpars')
            flag=False
        else:
            lpars=lpars.strip()
        
        if flag:
            
            parset=[float(x) for x in lpars.split(',') if len(x)>0]
            parset=np.array(parset)
            TFnames=TFsim.split('_')
            nPc=int(nPcycle)
            indices_af,indices_binding,indicesP,npars=matricesy.get_indices(TFnames,suff1,suff2,nPcycle=nPc,fixedbasal=fixedbasal,affinities=affinities)
            ifcb,ifcu=indices_af #indices of scale factor for binding and unbinding. Will be None if that is assumed not to change
            if ifcb[0] is not None:
                if af5X and af7X:
                    f5X=parset[ifcb[0]]
                    f7X=parset[ifcb[1]]
                    factorskb=np.array([f5X,f5X*f7X]) #if afidx=0, nothing, if afidx=1, first factor, and if afidx=2, second factor. 

                elif af5X:
                    f5X=parset[ifcb[0]]
                    factorskb=np.array([f5X,1]) 
                elif af7X:
                    f7X=parset[ifcb[0]]
                    factorskb=np.array([1,f7X]) 
                
            else:
                factorskb=np.array([1,1])

            if ifcu[0] is not None:
                if af5X and af7X:
                    f5X=parset[ifcu[0]]
                    f7X=parset[ifcu[1]]
                    factorsku=np.array([f5X,f5X*f7X])
                elif af5X:
                    f5X=parset[ifcu[0]]
                    factorsku=np.array([f5X,1])
                elif af7X:
                    f7X=parset[ifcu[0]]
                    factorsku=np.array([1,f7X])
            else:
                factorsku=np.array([1,1])
            pars_b=parset[indices_binding[0]]
            pars_u=parset[indices_binding[1]]

            #if 'v1' in funcgetpars.__name__:
            if not fixedbasal:
                pars_Pbasal_=parset[indicesP[0]]
                idxoffset=1 #if Pbasal is not fixed, then the TFs are one index position offset because idx 0 is Pbasal
            else:
                idxoffset=0

            if nPc==3:
                pars_Pbasal=np.array([pars_Pbasal_[0],pars_Pbasal_[1],0,pars_Pbasal_[2]])
                pars_Pbasal_forfull=pars_Pbasal_
            else:
                pars_Pbasal=pars_Pbasal_
                pars_Pbasal_forfull=pars_Pbasal
            for TFidx in range(len(TFnames)):
                TF=np.array([TFnames[TFidx]])
                idxsPTF=indicesP[TFidx+idxoffset] #
                parsPTF_=parset[idxsPTF]
                if nPc==3:
                    parsPTF=np.array([parsPTF_[0],parsPTF_[1],0,parsPTF_[2]])
                else:
                    parsPTF=parsPTF_

                dfind.loc[iloc]=np.concatenate((l1s,l2s,cost,factorskb,factorsku,pars_b,pars_u,pars_Pbasal,parsPTF,np.array([','.join(map(str,parset))]),TF))
                dffull.loc[iloc]=np.concatenate((l1s,l2s,cost,np.array([','.join(map(str,parset))]),np.array([','.join(map(str,pars_Pbasal_forfull))]),TF))
                iloc+=1
   
    return [dfind,dffull]

def parse_data(folder='',nlines=4,fixedbasal=False,affinities=['ZF(WT)','ZF(5X)','ZF(7X)']):
    alldataind=[]
    alldatafull=[]
    files=os.listdir(folder)
    nfiles=len(files)
   
    for fnum,f in enumerate(files):
        print('parsing %d/%d'%(fnum,nfiles),end='\r')
        dfind,dffull=readf(folder,f,fixedbasal=fixedbasal,nlines=nlines,affinities=affinities)
        alldataind.append(dfind)
        alldatafull.append(dffull)
    return [pd.concat(alldataind), pd.concat(alldatafull)]


def get_corresponding_fullparset(row,idcols=None,dffull=None,checkcost=False,fixedbasal=False,costf=None,GRF3=None,GRF4=None,affinities=[],dfexp=None):
    """Given a row for the df with one column for each parameter, gets the corresponding array of parameters to simulate again."""
    #row=['%g'%(x) if type(x)!=str else x for x in row.values ]
    #print(row)

    parpat=re.compile('\((.*)\)')
    affinities_=[parpat.findall(x)[0] for x in affinities]

    subdf=dffull.copy()
    for cname in idcols:
    #for i in range(len(idcols)):
        #print(row[i])
        #print(subdf[idcols[i]]==row[i])
        subdf=subdf[subdf[cname]==row[cname]] #i+1 if reset_index has been applied to the df
        #print(idcols[i],row[i],len(subdf))
        
    subdf=subdf.drop_duplicates(subset='fullparset')
    if len(subdf) != 1:
        print('Not 1 row found. Found %d instead.'%(len(subdf)))
        return [None,None]
    else:
        #print(subdf['fullparset'].values[0])
        fullparset=np.array([float(x) for x in subdf['fullparset'].values[0].split(',')])
        if fixedbasal:
            pars_Pbasal=np.array([float(x) for x in subdf['pars_Pbasal'].values[0].split(',')])
        else:
            pars_Pbasal=[]
        #print('len of fp is', len(fullparset))
        
    if checkcost: #checks that we can recover the cost in the dataframe
        #print(fullparset)
        nPcycle=int(subdf['nPcycle'])
        if nPcycle==3:
            GRF_=GRF3
        elif nPcycle==4:
            GRF_=GRF4
        suf1=subdf['suff1'].values[0]
        suf2=subdf['suff2'].values[0]
        TFnames=subdf['TFsim'].values[0].split('_')
        w1=int(subdf['w1'].values[0])
        w2=int(subdf['w2'].values[0])
        #print(TFnames,suf1,suf2,w1,w2)

        indices_af,indices_binding,all_indices,npars,parnames=matricesy.get_indices(TFnames,suf1,suf2,nPcycle=nPcycle,returnparnames=True,fixedbasal=fixedbasal)
        kwargs={'funcss':GRF_,'funcgetpars':matricesy.get_parameters_TF_v1,'fixedbasal':fixedbasal,'pars_Pbasal':pars_Pbasal,'nTFs':len(TFnames),'affinities':affinities_,'indicesbinding':indices_binding,'indicesP':all_indices,'indicesaf':indices_af}
        mat_fc,mat_names,mat_expmin,mat_expmax=matricesy.get_exp_matrix(dfexp,TFnames,affinities,matnames=True,min_=True,max_=True)
        expy=mat_fc
        mask=(mat_fc>0)
        dmin_=mat_expmin[mask]
        dmax_=mat_expmax[mask]
        newcost=costf(fullparset,mat_fc[mask],mask,dmin_,dmax_,kwargs,w1=w1,w2=w2)
        if np.abs(float(subdf['cost'].iloc[0])-newcost)>0.00001:
            print('cost does not coincide', newcost, subdf['cost'])
            costc=False
        else:
            #print('cost does coincide')
            costc=True
    else:
        costc=None
        
    return [fullparset,costc]
        