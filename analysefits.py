import numpy as np
import matricesy
import matplotlib.pyplot as plt
import os

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
    dif=maskmodel/maskexp
    
    ax=axes[0]
    im=ax.imshow(dif,aspect='auto',cmap=cmap1)
    ax.grid()
    plt.colorbar(im,ax=ax,label='fc model/fc exp') #,extend='both')
    #for row in range(len(mat_names)):
    #    for col in range(len(mat_names[0])):
    #        text=mat_names[row,col]
    #        #ax.text(col-0.5,row+0.5,text,fontsize=10,color='k')
    ax.set_xticks(range(len(mat_names[0])))
    ax.set_xticklabels(mat_names[0])
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
    ax.set_xticklabels(mat_names[0])
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

def plot_comparison_lines(exp_,model_,labels):
    fig,ax=plt.subplots(1,1,figsize=(18,4))
    ax.plot(range(len(exp_)),exp_,color='r',marker='o',label='exp')
    ax.plot(range(len(model_)),model_,color='k',marker='o',label='model')
    ax.set_xticks(range(len(model_)))
    ax.set_xticklabels(labels,rotation='90')
    ax.set_ylabel('fold change')
    ax.legend()
    plt.show()