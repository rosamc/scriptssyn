import numpy as np
import matplotlib.pyplot as plt
import re

afpat=re.compile(r'\((.+)\)')

def plot_syn(mat_fc,mat_names,refmats=None,refmats_names=None,TFpairs=None,markerdict=None,colorsdict=None,verbose=False,WTonly=False):
   
   
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    axempty=axes[1][1]
    axempty.set_title("TF+ZF alone")
    axsAD=axes[1][0]
    axsAD.set_title("same AD, different affinities")
    axdd=axes[0][0]
    axdd.set_title("different AD, different affinities")
    axdAD=axes[0][1]
    axdAD.set_title("different AD, same affinity")

    #print(axempty)
    #print(axsAD)
    #print(axdd)
    #print(axdAD)
    actual_pairs=[]
    
    #now normalise synergetic values to their mean
    for rnum,r in enumerate(range(1,len(mat_fc))[::3]):
        for cnum,c_ in enumerate(range(r-1,len(mat_fc[0]))[::3]):
            r0=r
            r1=r0+3
            c0=c_
            c1=c0+3
            submat=mat_fc[r0:r1,c0:c1]
            submat_names=mat_names[r0:r1,c0:c1]
            if np.any(submat_names!=None):
                TF1,TF2=submat_names[0][0].split('\n')
                TF1=TF1.split('(')[0] #col
                TF2=TF2.split('(')[0] #row
                
                if TF1 in refmats.keys() and TF2 in refmats.keys():
                    ref1=refmats[TF1] 
                    ref2=refmats[TF2] 
                    ref1_names=refmats_names[TF1]
                    ref2_names=refmats_names[TF2]
                    #print(submat_names,r0,c0,TF1,TF2)
                    if TF1==TF2:
                        sameAD=True
                    else:
                        sameAD=False
                    
                    av1=np.nanmean(ref1)
                    av2=np.nanmean(ref2)
                    skip=False
                    if av1<av2:
                        reverse=True
                        pair=TF2+"-"+TF1 #A=TF2, B=TF1
                        
                        if not pair in TFpairs:
                            pair_=TF1+"-"+TF2
                            if not pair_ in TFpairs:
                                skip=True
                        else:
                            pair_=pair

                    else:
                        reverse=False
                        pair=TF1+"-"+TF2
                        if not pair in TFpairs:
                            pair_=TF2+"-"+TF1
                            if not pair_ in TFpairs:
                                skip=True
                        else:
                            pair_=TF1+"-"+TF2
                    if not skip:
                        actual_pairs.append(pair)
                        #color=cmap(norm(TFpairs.index(pair)))
                        color=colorsdict[pair_]
                        
                        for i in range(3): #row
                            for j in range(3): #col
                                if submat_names[i][j]:
                                    x1=submat[i][j]/ref1[j][j] #col
                                    x2=submat[i][j]/ref2[i][i] #row
                                    if verbose:
                                        print("----")

                                        print(submat_names[i][j],"combined fc is", submat[i][j])
                                        print("row TF is", TF2,"with affinity",i,ref2[i][i], "col TF is", TF1, "with affinity", j,ref1[j][j] )
                                        print("ratios are", x1, x2)
                                        print("pair order according to average expression is", pair)

                                    if reverse:
                                        if verbose:
                                            print("reverse affinities, i=%d, j=%d,"%(i,j),submat_names[j,i])
                                        affinities=afpat.findall(submat_names[j,i])
                                    else:
                                        affinities=afpat.findall(submat_names[i,j])
                                    if verbose:
                                        print("affinities for marker are", affinities)
                                    marker=markerdict["-".join(affinities)]

                                    if (TF1=='e' or TF2=='e') and not TF1==TF2:
                                        ax=axempty
                                    elif sameAD:
                                        if i!=j:
                                            ax=axsAD
                                        else:
                                            ax=None
                                    else:
                                        if i!=j:
                                            
                                            ax=axdd
                                        else:
                                            ax=axdAD
                                    if ax:  
                                        
                                        #A is the TF with higher expression
                                        if reverse:
                                            SAB=np.log2(x2)
                                            SBA=np.log2(x1)
                                        else:
                                            SAB=np.log2(x1)
                                            SBA=np.log2(x2)
                                            #print(TF1,TF2,"A is %s, B is %s"%(TF1,TF2))
                                            
                                            #print(TF1,TF2,"A is %s, B is %s"%(TF2,TF1))
                                        if WTonly:
                                            if "-".join(affinities)=="WT-WT":
                                                ax.scatter(SAB,SBA,marker=marker,color=color,s=60) #,label=submat_names[i,j]) #,color=colorsAD[TF1])
                                        else:
                                            ax.scatter(SAB,SBA,marker=marker,color=color,s=60) #,label=submat_names[i,j]) #,color=colorsAD[TF1])

    ax=axes[0][0]
    for key in markerdict.keys():
        ax.scatter(-10,-10,marker=markerdict[key],label=key,color='grey')
    ax.legend(loc='upper right',bbox_to_anchor=(-0.1,1),ncol=1)
    for key in colorsdict.keys():
        t0,t1=key.split("-")
        if t0!=t1 and t1 !="e":
            ax=axes[0][1]
        else:
            ax=axes[1][0]
        color=colorsdict[key]
        if key in actual_pairs:
            label=key
        else:
            label=t1+"-"+t0
        ax.scatter(-10,-10,marker='o',color=color,label=label)
    ax=axes[0][1]
    ax.legend(loc='upper right',bbox_to_anchor=(1.5,-0.1),ncol=10)
    ax=axes[1][0]
    ax.legend(loc='upper left',bbox_to_anchor=(-0.5,-0.25),ncol=11)

    
    for ax in [axempty,axdd,axdAD]:
        ax.set_xticks(np.arange(-1,2.5,0.5))
        ax.set_yticks(np.arange(-1,2,0.5))
        ax.set_xlim(-1.25,1.)
        ax.set_ylim(-0.5,1.5)
    ax=axsAD
    ax.set_xticks(np.arange(-0.5,0.75,0.25))
    ax.set_yticks(np.arange(-0.5,0.75,0.25))
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)

    for r,row in enumerate(axes):
        for a,ax in enumerate(row):
            ax.set_xlabel(r'$S_{AB}$')
            ax.set_ylabel(r'$S_{BA}$')
            ax.tick_params(axis='both',labelsize=8)
            
            #ax.legend(loc=lc,bbox_to_anchor=bbox,ncol=2)
            ax.axhline(y=0,linestyle='--',color='grey')
            ax.axvline(x=0,linestyle='--',color='grey')


    return [fig,axes]

def plot_syn_Astronger(mat_fc,mat_names,refmats=None,refmats_names=None,TFpairs=None,markerdict=None,colorsdict=None,verbose=False,WTonly=False):
   
   
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    axempty=axes[1][1]
    axempty.set_title("TF+ZF alone")
    axsAD=axes[1][0]
    axsAD.set_title("same AD, different affinities")
    axdd=axes[0][0]
    axdd.set_title("different AD, different affinities")
    axdAD=axes[0][1]
    axdAD.set_title("different AD, same affinity")

    #print(axempty)
    #print(axsAD)
    #print(axdd)
    #print(axdAD)
    actual_pairs=[]
    
    #now normalise synergetic values to their mean
    for rnum,r in enumerate(range(1,len(mat_fc))[::3]):
        for cnum,c_ in enumerate(range(r-1,len(mat_fc[0]))[::3]):
            r0=r
            r1=r0+3
            c0=c_
            c1=c0+3
            submat=mat_fc[r0:r1,c0:c1]
            submat_names=mat_names[r0:r1,c0:c1]
            if np.any(submat_names!=None):
                TF1,TF2=submat_names[0][0].split('\n')
                TF1=TF1.split('(')[0] #col
                TF2=TF2.split('(')[0] #row
                
                if TF1 in refmats.keys() and TF2 in refmats.keys():
                    ref1=refmats[TF1] 
                    ref2=refmats[TF2] 
                    ref1_names=refmats_names[TF1]
                    ref2_names=refmats_names[TF2]
                    #print(submat_names,r0,c0,TF1,TF2)
                    if TF1==TF2:
                        sameAD=True
                    else:
                        sameAD=False
                    
                    #av1=np.nanmean(ref1)
                    #av2=np.nanmean(ref2)
                    skip=False
                    pair=TF2+"-"+TF1 #A=TF2, B=TF1
                    if not pair in TFpairs:
                        pair_=TF1+"-"+TF2
                        if not pair_ in TFpairs:
                            skip=True
                    else:
                        pair_=pair
                    color=colorsdict[pair_]
                    
                    if not skip:
                        #actual_pairs.append(pair)
                        #color=cmap(norm(TFpairs.index(pair)))
                        #color=colorsdict[pair_]
                        
                        for i in range(3): #row
                            for j in range(3): #col
                                if submat_names[i][j]:
                                    x1=submat[i][j]/ref1[j][j] #col
                                    x2=submat[i][j]/ref2[i][i] #row
                                    affinities=afpat.findall(submat_names[i,j])

                                    if ref2[i][i]<ref1[j][j]: #row TF needs to be B, col TF needs to be A
                                    	reverse=True
                                    	A_TF=submat_names[j,j]
                                    	B_TF=submat_names[i,i]
                                    	SAB=np.log2(x1)
                                    	SBA=np.log2(x2)
                                    	afstr="-".join(affinities)

                                    else:
                                    	reverse=False
                                    	A_TF=submat_names[i,i]
                                    	B_TF=submat_names[j,j]
                                    	SAB=np.log2(x2)
                                    	SBA=np.log2(x1)
                                    	afstr="-".join([affinities[1],affinities[0]])

                                    actual_pairs.append(A_TF+";"+B_TF)


                                    if verbose:
                                        print("----")

                                        print(submat_names[i][j],"combined fc is", submat[i][j])
                                        print("row TF is", TF2,"with affinity",i,ref2[i][i], "col TF is", TF1, "with affinity", j,ref1[j][j] )
                                        print("ratios are", x1, x2)
                                        print("pair order according to individual expression is", A_TF, B_TF)

                                        print("affinities for marker are", afstr)
                                    marker=markerdict[afstr]

                                    if (TF1=='e' or TF2=='e') and not TF1==TF2:
                                        ax=axempty
                                    elif sameAD:
                                        if i!=j:
                                            ax=axsAD
                                        else:
                                            ax=None
                                    else:
                                        if i!=j:
                                            
                                            ax=axdd
                                        else:
                                            ax=axdAD
                                    if ax:  
                                        
                                        if WTonly:
                                            if "-".join(affinities)=="WT-WT":
                                                ax.scatter(SAB,SBA,marker=marker,color=color,s=60) #,label=submat_names[i,j]) #,color=colorsAD[TF1])
                                        else:
                                            ax.scatter(SAB,SBA,marker=marker,color=color,s=60) #,label=submat_names[i,j]) #,color=colorsAD[TF1])

    ax=axes[0][0]
    for key in markerdict.keys():
        ax.scatter(-10,-10,marker=markerdict[key],label=key,color='grey')
    ax.legend(loc='upper right',bbox_to_anchor=(-0.1,1),ncol=1)
    for key in colorsdict.keys():
        t0,t1=key.split("-")
        if t0!=t1 and t1 !="e":
            ax=axes[0][1]
        else:
            ax=axes[1][0]
        color=colorsdict[key]
        if key in actual_pairs:
            label=key
        else:
            label=t1+"-"+t0
        ax.scatter(-10,-10,marker='o',color=color,label=label)
    ax=axes[0][1]
    ax.legend(loc='upper right',bbox_to_anchor=(1.5,-0.1),ncol=10)
    ax=axes[1][0]
    ax.legend(loc='upper left',bbox_to_anchor=(-0.5,-0.25),ncol=11)

    
    for ax in [axempty,axdd,axdAD]:
        ax.set_xticks(np.arange(-1,2.5,0.5))
        ax.set_yticks(np.arange(-1,2,0.5))
        ax.set_xlim(-1.25,1.)
        ax.set_ylim(-0.5,1.5)
    ax=axsAD
    ax.set_xticks(np.arange(-0.5,0.75,0.25))
    ax.set_yticks(np.arange(-0.5,0.75,0.25))
    ax.set_xlim(-0.5,0.5)
    ax.set_ylim(-0.5,0.5)

    for r,row in enumerate(axes):
        for a,ax in enumerate(row):
            ax.set_xlabel(r'$S_{AB}$')
            ax.set_ylabel(r'$S_{BA}$')
            ax.tick_params(axis='both',labelsize=8)
            
            #ax.legend(loc=lc,bbox_to_anchor=bbox,ncol=2)
            ax.axhline(y=0,linestyle='--',color='grey')
            ax.axvline(x=0,linestyle='--',color='grey')


    return [fig,axes]

def plot_mult_add(mat_fc,mat_names,refmats=None,refmats_names=None,TFpairs=None,markerdict=None,colorsdict=None,verbose=False):
    fig,axes=plt.subplots(2,2,figsize=(12,10))
    axempty=axes[1][1]
    axempty.set_title("TF+ZF alone")
    axsAD=axes[1][0]
    axsAD.set_title("same AD, different affinities")
    axdd=axes[0][0]
    axdd.set_title("different AD, different affinities")
    axdAD=axes[0][1]
    axdAD.set_title("different AD, same affinity")

    print(axempty)
    print(axsAD)
    print(axdd)
    print(axdAD)
    actual_pairs=[]
    verbose=False
    #now normalise synergetic values to their mean
    for rnum,r in enumerate(range(1,len(mat_fc))[::3]):
        for cnum,c_ in enumerate(range(r-1,len(mat_fc[0]))[::3]):
            r0=r
            r1=r0+3
            c0=c_
            c1=c0+3
            submat=mat_fc[r0:r1,c0:c1]
            submat_names=mat_names[r0:r1,c0:c1]
            if np.any(submat_names!=None):
                TF1,TF2=submat_names[0][0].split('\n')
                TF1=TF1.split('(')[0] #col
                TF2=TF2.split('(')[0] #row
                if TF1 in refmats.keys() and TF2 in refmats.keys():
                    ref1=refmats[TF1] 
                    ref2=refmats[TF2] 
                    ref1_names=refmats_names[TF1]
                    ref2_names=refmats_names[TF2]
                    #print(submat_names,r0,c0,TF1,TF2)
                    if TF1==TF2:
                        sameAD=True
                    else:
                        sameAD=False
                    av1=np.nanmean(ref1)
                    av2=np.nanmean(ref2)
                    skip=False
                    if av1<av2:
                        reverse=True
                        pair=TF2+"-"+TF1 #A=TF2, B=TF1
                        
                        if not pair in TFpairs:
                            pair_=TF1+"-"+TF2
                            if not pair_ in TFpairs:
                                skip=True
                        else:
                            pair_=pair

                    else:
                        reverse=False
                        pair=TF1+"-"+TF2
                        if not pair in TFpairs:
                            pair_=TF2+"-"+TF1
                            if not pair_ in TFpairs:
                                skip=True
                        else:
                            pair_=TF1+"-"+TF2
                    if not skip:
                    
                        actual_pairs.append(pair)
                        #color=cmap(norm(TFpairs.index(pair)))
                        color=colorsdict[pair_]
                        
                        for i in range(3): #row
                            for j in range(3): #col
                                if submat_names[i][j]:
                                    Qab=submat[i][j]
                                    Qa=ref2[i]
                                    Qb=ref1[j]
                                    
                                    x=Qab/(Qa*Qb)
                                    y=(Qab-1)/((Qa-1)+(Qb-1))
                                    affinities=afpat.findall(submat_names[i,j])
                                    if verbose:
                                        print("----")
                                        print("affinities", affinities)
                                        print(submat_names[i][j],"combined fc is", submat[i][j])
                                        print("label pair is", pair)
                                        print(TF1,TF2,affinities,';\n Qab',Qab,'rowTF',TF2,affinities[1],Qa,'; col TF',TF1,affinities[0],Qb,';\n x',x,'y',y)
                  
                                    if reverse:
                                        if verbose:
                                            print("reverse affinities, i=%d, j=%d,"%(i,j),submat_names[j,i])
                                        affinities=afpat.findall(submat_names[j,i])
                                    if verbose:
                                        print("affinities for marker are", affinities)
                                    
                                    
                                    
                                    marker=markerdict["-".join(affinities)]

                                    if (TF1=='e' or TF2=='e') and not TF1==TF2:
                                        ax=axempty
                                    elif sameAD:
                                        if i!=j:
                                            ax=axsAD
                                        else:
                                            ax=None
                                    else:
                                        if i!=j:
                                            
                                            ax=axdd
                                        else:
                                            ax=axdAD
                                    if ax:  
                                        
                                        #A is the TF with higher expression
                                        
                                        ax.scatter(x,y,marker=marker,color=color,s=60) #,label=submat_names[i,j]) #,color=colorsAD[TF1])

    ax=axes[0][0]
    for key in markerdict.keys():
        ax.scatter(-10,-10,marker=markerdict[key],label=key,color='grey')
    ax.legend(loc='upper right',bbox_to_anchor=(-0.1,1),ncol=1)
    for key in colorsdict.keys():
        t0,t1=key.split("-")
        if t0!=t1 and t1 !="e":
            ax=axes[0][1]
        else:
            ax=axes[1][0]
        color=colorsdict[key]
        if key in actual_pairs:
            label=key
        else:
            label=t1+"-"+t0
        ax.scatter(-10,-10,marker='o',color=color,label=label)
    ax=axes[0][1]
    ax.legend(loc='upper right',bbox_to_anchor=(1.5,-0.1),ncol=10)
    ax=axes[1][0]
    ax.legend(loc='upper left',bbox_to_anchor=(-0.5,-0.25),ncol=11)

    if True:
        for ax in [axempty,axsAD,axdd,axdAD]:
            #ax.set_xticks(np.arange(-3,3,0.5))
            #ax.set_yticks(np.arange(-3,3,0.5))
            ax.set_xlim(0.2,1.5)
            ax.set_ylim(0.25,1.6)
            ax.set_yscale('log')
            ax.set_xscale('log')
        ax=axempty
        ax.set_xlim(0.2,1.5)
        ax.set_ylim(0.2,2)
        ax.set_yscale('log')
        ax.set_xscale('log')

    for r,row in enumerate(axes):
        for a,ax in enumerate(row):
            ax.set_xlabel('deviation from mult')
            ax.set_ylabel('deviation from add')
            ax.tick_params(axis='both',labelsize=8)
            
            #ax.legend(loc=lc,bbox_to_anchor=bbox,ncol=2)
            ax.axhline(y=1,linestyle='--',color='grey')
            ax.axvline(x=1,linestyle='--',color='grey')


    return [fig,axes]