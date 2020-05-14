import numpy as np
from scipy.linalg import null_space
import math
import networkx as nx
import matplotlib.pyplot as plt

def compute_synergy(pars,f,ftype='crit',Aval=1.0, Bval=1.0,returnm=False, anystronger=False, fcind=None, fcpair=None):
    mstars=[]
    for i in range(4):
        if i==0:
            A=0.0
            B=0.0
        elif i==1:
            A=2.0*Aval
            B=0.0
        elif i==2:
            A=0.0
            B=2.0*Bval
        else:
            A=Aval
            B=Bval
        if ftype=='crit': #function that uses critical points
            m=f(pars,np.array([B]),A)
        elif ftype=='pythonL':
            m=f(pars,A,B)
        elif ftype=='eigen':
            varar=np.array([A,B])
            m=f(pars,varar)
        else:
            raise ValueError("ftype not understood.")

        mstars.append(m)
    
    if anystronger:
        cont=True
    else: #return None in case A is weaker than B
        if mstars[1]<mstars[2]:
            #print("changing order")
            return [None,None] #[SBA,SAB]
        else:
            cont=True
    if cont:
        if fcind is not None:
            if mstars[1]/mstars[0]>fcind or mstars[2]/mstars[0]>fcind:
                return [None, None]
        if fcpair is not None:
            r=mstars[1]/mstars[2]
            if r>1:
                if r>fcpair:
                    return [None,None]
            else:
                if (1/r)>fcpair:
                    return [None,None]
        
        SAB=np.log2(mstars[3]/mstars[1])
        SBA=np.log2(mstars[3]/mstars[2])
        if returnm:
            return [SAB,SBA,mstars]
        else:
            #print("Afirst")
            return [SAB,SBA]

def compute_ss_fromL(L):
    L_d=np.diag(np.sum(L,axis=0))
    L=L-L_d
    rhos=null_space(L) #column vector
    rhos=np.transpose(rhos)[0] #1 row
    ss=np.abs(rhos)/np.sum(np.abs(rhos)) #can be - when very close to 0
    return ss

def draw_G_fromalledges(alledges,bsites=2,nnodes_cycle=3,separate=True,figsize1=(10,10),figsize2=(10,20),coords_t=[(1.5,4),(4,1.5),(0,0)],drawlabels=True):
    #figsize_transitions=(10,10) #this has to be tuned depending on the size of the graph
    G=nx.DiGraph()
    for edge in alledges:
        G.add_edge(edge[0],edge[2],lab=edge[1])
    coords=dict()
    nnodes_b=2**bsites
    #coords_t=
    yoffset=1.5 #1.1
    xoffset=1.5 #1.1
    #coords_b=[(2,2),(1,1),(3,1),(2,0)]

    #calculate the coordinates of the binding graph
    coords_b=[]
    nodes_per_level=[]
    for i in range(0,bsites+1):
        nodes_per_level.append(int(math.factorial(bsites)/(math.factorial(bsites-i)*math.factorial(i))))
    maxn=max(nodes_per_level)
    middle=maxn/2
    for i in range(bsites+1):
        y=(bsites-i)
        if nodes_per_level[i]==1:
            coords_b.append((middle,y))
        else:
            offset=(maxn-nodes_per_level[i])/2
            for x in np.linspace(0,nodes_per_level[i],nodes_per_level[i]):
                #print(i,":",nodes_per_level[i],x,y)
                coords_b.append((x+offset,y))
    n=1
    for j in range(nnodes_cycle):
        t=coords_t[j]
        dx,dy=t
        for i in range(nnodes_b):
            coords[n]=(coords_b[i][0]+dx*xoffset,coords_b[i][1]+dy*yoffset)
            n+=1

    colors_persite=['','darkolivegreen','lightgreen','navy','lightblue','red','orange','magenta','violet','saddlebrown','maroon']

    if bsites>10:
        for i in range(bsites-len(colors_persite)):
            colors_persite.append('k')

    bcolors=[]
    tcolors=[]
    bedges=[]
    tedges=[]
    labsdict={e:G[e[0]][e[1]]['lab'] for e in G.edges}
    labsdict_t={k:v for k,v in labsdict.items() if v[0]=='k'}
    labsdict_b={k:v for k,v in labsdict.items() if v[0]!='k'}
    for edge in G.edges:
        n0,n1=edge
        bgroup0=n0%nnodes_b
        bgroup1=n1%nnodes_b
        #print(n0,n1,bgroup0,bgroup1)
        if bgroup0!=bgroup1:
            bcolors.append('b')
            bedges.append(edge)
        else:
            tcolors.append('r')
            tedges.append(edge)
        #print()
        
    #first plot a graph labelling the transitions
    fig,ax=plt.subplots(1,1,figsize=figsize1)
    nx.draw_networkx_nodes(G,pos=coords,node_color='lightgray',ax=ax)
    nx.draw_networkx_labels(G,pos=coords,ax=ax)
    nx.draw_networkx_edges(G,pos=coords,edgelist=tedges,edge_color=tcolors,ax=ax,connectionstyle="arc3,rad=-0.1")
    if drawlabels:
        nx.draw_networkx_edge_labels(G,pos=coords,edge_labels=labsdict_t,alpha=1,rotate=True,label_pos=0.2,bbox={'alpha':0},font_color='r',ax=ax)

    if separate:
        nx.draw_networkx_edges(G,pos=coords,edgelist=bedges,edge_color=bcolors,connectionstyle="arc3,rad=0.1",ax=ax)
        if drawlabels:
            nx.draw_networkx_edge_labels(G,pos=coords,edge_labels=labsdict_t,alpha=1,rotate=True,label_pos=0.2,bbox={'alpha':0},font_color='r',ax=ax)
    
        plt.show()
    else:
        for i in range(2): 
            if i==0:
                #linestyle='solid'
                width=2 #the linestyle doesn't work, so I am making binding edges wider than unbinding edges
                edgelist=[k for k,v in labsdict_b.items() if v[0]=='a']
            else:
                width=1
                edgelist=[k for k,v in labsdict_b.items() if v[0]=='b']
        
            ecolors=[]
            for e in edgelist:
                site=G[e[0]][e[1]]['lab'].split("_")[0][1:] #label is of form "asite_"
                color=colors_persite[int(site)]
                ecolors.append(color)
            
            nx.draw_networkx_edges(G,pos=coords,edgelist=edgelist,edge_color=ecolors,ax=ax,width=width,connectionstyle="arc3,rad=0.1")
            if drawlabels:
                nx.draw_networkx_edge_labels(G,pos=coords,edge_labels={k:v for k,v in labsdict_b.items() if e in edgelist},alpha=1,rotate=True,label_pos=0.7,bbox={'alpha':0},font_color='k',ax=ax)
        plt.show()

    if separate:  
        #now plot the binding transitions

        fig,axes=plt.subplots(nnodes_cycle,1,figsize=figsize2)
        for a, ax in enumerate(axes):
            nodes=[node for node in G.nodes if ((node-1)//(nnodes_b))==a]
            nx.draw_networkx_nodes(G,nodelist=nodes,pos=coords,node_color='lightgray',ax=ax)
            nx.draw_networkx_labels(G,labels={n:n for n in G.nodes if n in nodes},pos=coords,ax=ax)
            for i in range(2): 
                if i==0:
                    #linestyle='solid'
                    width=2 #the linestyle doesn't work, so I am making binding edges wider than unbinding edges
                    edgelist=[k for k,v in labsdict_b.items() if ((k[0] in nodes) and (k[1] in nodes) and v[0]=='a')]
                    labelcolor='k'
                else:
                    width=1
                    edgelist=[k for k,v in labsdict_b.items() if ((k[0] in nodes) and (k[1] in nodes) and v[0]=='b')]
                    labelcolor='gray'
                ecolors=[]
                for e in edgelist:
                    color=colors_persite[int(G[e[0]][e[1]]['lab'][1])]
                    ecolors.append(color)
                nx.draw_networkx_edges(G,pos=coords,edgelist=edgelist,edge_color=ecolors,ax=ax,width=width,connectionstyle="arc3,rad=0.1")
                if drawlabels:
                    nx.draw_networkx_edge_labels(G,pos=coords,edge_labels={k:v for k,v in labsdict_b.items() if k in edgelist},alpha=1,rotate=True,label_pos=0.7,bbox={'alpha':0},font_color=labelcolor,ax=ax)
        for ax in axes:
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
        plt.show()
    return 


def mask_spatial_subsample(seed=1, subset=None,grids=[],svar="quadrant",svarvals=[1,2,3],gvar=["SAB","SBA"]):
    all_idxs=np.arange(len(subset))
    maskall=np.zeros(len(subset),dtype=bool)
    for v,value in enumerate(svarvals):
        grid=grids[v]
        x,y=grid
        grid=np.zeros((len(y),len(x)))
        idxs=[] #original indexes selected for this value of svar (e.g. for this quadrant)
        
        mask_=(subset[svar]==value)
        s_=subset[mask_]
        idxs_=all_idxs[mask_] #all original indexes corresponding to this quadrant
        range_=np.arange(len(s_))
        np.random.seed(seed)
        np.random.shuffle(range_)
        print("svarval", value)
        print(len(s_),np.sum(grid))
        for i in range_:
            v1=s_.iloc[i][gvar[0]]
            v2=s_.iloc[i][gvar[1]]
            r=np.where(v2>=y)[0][-1]
            c=np.where(v1>=x)[0][-1]
            #print(SAB,SBA,c,r)
            if grid[r,c]<1:
                grid[r,c]=1
                idxs.append(idxs_[i]) 
        maskall[idxs]=True
        print("selected", np.sum(grid))
        
    return maskall

