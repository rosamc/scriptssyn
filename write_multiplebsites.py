import numpy as np
import os
from scipy.linalg import null_space

def link_next_binding(n,sites):
    """n: total number of sites
    sites: Dictionary. Key: mesostate (set with node sites bound). Value: node id."""
    allnextsites=dict()
    edges=[]
    
    allids=[]
    for mstate_id in sites.items():
        mstate,id_=mstate_id
        allids.append(id_)
    newid=max(allids)+1 #index of subsequent mesostate
    
    for mstate_id in sites.items(): #for each site, connect to the following ones
        mstate,id_=mstate_id
        if len(mstate)==1 and 0 in mstate:
            free=range(1,n+1)
            bound=set([])

        else:
            allsites=set(range(1,n+1))
            bound=set(mstate)
            free=allsites.difference(bound)
        #nextsites=[]
        for site in free:
            news=bound.copy()
            news.add(site) 
            news=frozenset(news) #new mesostate 
            #nextsites.append(news)
            if news in allnextsites.keys():
                nid=allnextsites[news]
            else:
                nid=newid
                newid+=1
                allnextsites[news]=nid
                
            string="%d_"%site+"U".join(map(str,sorted(mstate))) #newsite_oldsite1,oldsite2,oldsite3. For binding label
            mstate_=[x for x in mstate if x!=0]
            stringr="%d_"%site+"U".join(map(str,sorted(mstate_+[site]))) #newsite_,oldsite1,oldsite2,oldsite3,newsite. For unbinding label
            laba="a%s-x"%string
            labb="b%s"%(stringr)
            
                
            edges.append([id_,laba,nid])
            edges.append([nid,labb,id_])
            
    return [allnextsites,edges]
    
def get_bindinggraph_edges(n):
    prev={frozenset([0]):1} #node with nothing bound gets index 1
    edges=[]
    for i in range(n):
        prev,labels=link_next_binding(n,prev)
        edges.extend(labels)
    return edges

def merge_binding_withcycle(bgraph_edges,nnodes_cycle,rev_cycle=[]):
    alledges=[]
    states=[]
    nnodesb=max([max([e[0] for e in bgraph_edges]),max([e[2] for e in bgraph_edges])]) #number of nodes in binding graph
    for b,edge in enumerate(bgraph_edges):
        i1,label,i2=edge
        
        firsti1=i1
        state=label.strip("-x").split("_")[-1]
        for c,cstate in enumerate(range(1,nnodes_cycle+1)):
            #newbindinglabel
            newlabel=label+"_"+str(cstate)
            if "x" in newlabel:
                newlabel=newlabel.replace("-x","")+"-x" #more -x to the end
            i1_=i1+c*nnodesb
            i2_=i2+c*nnodesb
            alledges.append([i1_,newlabel,i2_])
            if not state in states: #only need to do once per state
                if cstate==nnodes_cycle:
                    alledges.append([i1_,"k_%d_%s"%(cstate,state),firsti1])
                    if cstate in rev_cycle:
                        alledges.append([firsti1,"kr_%d_%s"%(cstate,state),i1_])
                else:
                    i2_=i1_+nnodesb
                    alledges.append([i1_,"k_%d_%s"%(cstate,state),i2_])
                    if cstate in rev_cycle:
                        alledges.append([i2_,"kr_%d_%s"%(cstate,state),i1_])
        states.append(state)
    return alledges

def get_laplacian_toprint(edges, nnodes,TFnames=[],force=False):
    mat=np.zeros((nnodes,nnodes),dtype=object)
    if len(TFnames)==0:
        if not force:
            raise ValueError("Please specify TFnames, or run with force=True.")
    for edge in edges:
        i1,lab,i2=edge
        col=i1-1
        row=i2-1
        for TF in TFnames:
            lab=lab.replace("-"+TF,"*"+TF)
        mat[row,col]=lab
    return mat

def write_laplacian_py(L,alledges,folder="./",fname=None):
    outf=open(os.path.join(folder,"%s.py"%fname),"w")
    outf.write("import numpy as np\n\n")
    outf.write("def get_L_%s(pars,A,B):\n"%fname)
    outf.write("    ")
    for en,edge in enumerate(alledges):
        lab=edge[1]
        if en<len(alledges)-1:
            end=","
        else:
            end="=pars"
        outf.write(lab.strip("-A").strip("-B")+end)
    outf.write("\n\n")
       
    for r,row in enumerate(L):
        if r<len(L)-1:
            end="],"
        else:
            end="]])"
        if r==0:
            beg="    L=np.array(["
        else:
            beg="    "
        string="["+",".join(map(str,row))+end
        outf.write(beg+string)
        outf.write("\n")
    outf.write("    return L")
    outf.write("\n")
    outf.close()

def compute_ss_fromL(L):
    L_d=np.diag(np.sum(L,axis=0))
    L=L-L_d
    rhos=null_space(L) #column vector
    rhos=np.transpose(rhos)[0] #1 row
    ss=np.abs(rhos)/np.sum(np.abs(rhos)) #can be - when very close to 0
    return ss


