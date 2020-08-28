import numpy as np
import networkx as nx

def get_fluxes(edges,A,B,parset,Parray,netJ=False):
    """Get the (square) matrix of fluxes between pairs of edges. If Parray is steady state probability distribution, it is the ss fluxes.
    Otherwise, it is flux at a given timepoint.
    If net is True, it returns net flux, otherwise all fluxes"""
   
    Pdiag=np.diag(Parray)
    ratemat=np.zeros((9,9)) #Important TO ADD CONCENTRATIONS OF A AND B HERE!!
    for i in range(len(parset)):
        v0=edges[i][0]
        v1=edges[i][2]
        ratelab=edges[i][1]
        parval=parset[i]
        if 'kb' in ratelab:
            if 'A' in ratelab:
                #print('A binding edge',ratelab)
                parval=parval*A
            elif 'B' in ratelab:
                #print('B binding edge', ratelab)
                parval=parval*B
        ratemat[v0,v1]=parval
    J=Pdiag@ratemat

    if netJ is True:
        Jnet=np.zeros_like(J)
        for i in range(len(edges)):
            rate=edges[i]
            v0,v1=[rate[0],rate[2]]
            f1=J[v0,v1]
            f2=J[v1,v0]
            
            if f1>f2:
                v0_=v0
                v1_=v1
                netf=f1-f2
                #print('f1',f1,'f2',f2)
            else:
                v0_=v1
                v1_=v0
                netf=f2-f1
            Jnet[v0_,v1_]=netf
        return Jnet
    else:
        return J

def get_mat_fluxes_abonly(df,flux_names,npars=24,nnodes=9,netJ=False,rhofunc=None,graph_edges=None,min_=1e-15):
    """Given a dataframe with npars columns of parameter values, computes the fluxes for each of them, for both A and B together. 
    It removes columns with 0 flux (handy when two nodes are not connected).
    If netJ is True, computes net fluxes."""
    all_Js=[]
    A=1.0
    B=1.0
    
    for idx in range(len(df)):
        
        row=df.iloc[idx]
        parset=row[:npars].values.copy()
        
            
        #print('parset', parset)
        rhos=rhofunc(parset,np.array([B]),A)
        P=rhos/np.sum(rhos)

        J=get_fluxes(graph_edges,A,B,parset,P,netJ=netJ)
        
        all_Js.append(J.flatten())
              
    all_Js=np.array(all_Js)
    #keep only those columns with nonzero fluxes
    mask=[]
    for col in range(nnodes*nnodes):
        if np.any(all_Js[:,col]>min_):
            mask.append(col)
    #print(mask)
    #mat=np.log10(all_Js)[:,mask]
    return [all_Js[:,mask],flux_names[mask]]
def get_order_cols_fluxes(labls_,flux_order_labs_AB):
    """Given the fluxes ordered according to labls_, it will return an array with the indices that would sort them according to flux_order_labls_AB.
    Accounts for some missing labels (i.e. nonexisting flux between that pair)."""
    order_cols=[]
    i=0
    while i<len(flux_order_labs_AB):
        lab=flux_order_labs_AB[i]
        idx=np.where(labls_==lab)[0]
        if len(idx)==1:
            idx=idx[0]
            order_cols.append(idx)
        elif len(idx)>1:
            print('more than one!',idx)

        i+=1
    order_cols=np.array(order_cols)
    return order_cols
def fluxname_to_parname(pair,edges=None):
    """Given a flux name, e.g. 1-0.AB, it gives the corresponding parameter label."""
    n0,n1=list(map(int,pair.split(".")[0].split("-")))
    n0=n0+1
    n1=n1+1
    for e in edges:
        if e[0]==n0 and e[2]==n1:
            return e[1]
def get_graph_fromJmatrix(J,edges,ratios=True, subtract=True,vmin=1e-15):
    """Graph from the J matrix, with edge weights determined by J. 
    Originally J is the matrix of fluxes, but can be used with any other matrix and ratios=False to construct the graph from that matrix."""
    G=nx.DiGraph()
    #P=rhos_3statepol_allrev(A,B,parset)
    #print('ss P', P)
    #fluxes=get_fluxes(edges,A,B,parset,P)
    for i in range(len(edges)):
        rate=edges[i]
        v0,v1=[rate[0],rate[2]]
        f1=J[v0,v1]
        f2=J[v1,v0]
        if f1>vmin or f2>vmin: #added 10.03
            if ratios is False:
                v0_=v0
                v1_=v1
                weight=f1
                G.add_edge(v0_,v1_,weight=weight) #weight is the flux
            else:
                if f1>f2:
                    v0_=v0
                    v1_=v1
                    if subtract:
                        weight=f1-f2
                    else:
                        if f2>vmin:
                            weight=f1/f2
                        else:
                            weight=f1
                    #print('f1',f1,'f2',f2)
                else:
                    v0_=v1
                    v1_=v0
                    if subtract:
                        weight=f2-f1
                    else:
                        if f1>vmin:
                            weight=f2/f1
                        else:
                            weight=f2
                    #print('f2',f2,'f1',f1)
                #print(rate,f1,f2,weight,np.log10(weight))
                #if np.abs(weight)>10**(-8): #nonzero weight
            
                G.add_edge(v0_,v1_,weight=weight) #weight is the net flux
        
    return G

def get_graph_fromJarray(J,edgelabls,min_=0):
    """Returns a graph with weights those of the J array"""
    G=nx.DiGraph()
    for pnum,pair in enumerate(edgelabls):
        flux=J[pnum]
        if flux>min_:
            pair=pair.split('.')[0]
            v0,v1=pair.split('-')
            G.add_edge(int(v0), int(v1),weight=flux)
        #else:
        #    print('discarting',flux)
    return G

def get_dominant_path_G(G,node0):
    G2=nx.DiGraph()
    n=node0
    done=False
    while done is False:
        oute=list(G.out_edges(n))
        if len(oute)==0:
            done=True
        else:
            weights=[G[e[0]][e[1]]['weight'] for e in oute]

            n2=oute[np.argmax(weights)][1]

            if n2==node0 or n2 in G2.nodes:
                done=True
            G2.add_edge(n,n2,weight=np.max(weights))
            n=n2
    return G2
def get_dominant_paths_mat_v4(smat,labls_,node0,min_=1e-14):
    #for a matrix of instances, returns a matrix where each row is a parameter set, and each column is an edge, with value 1 if in the dominant path
    dominant_graphs=np.zeros((len(smat),len(labls_)))
    ext=labls_[0].split('.')[1]
    for idx in range(len(smat)):
        J=smat[idx]
        G=get_graph_fromJarray(J,labls_,min_=min_) #Get corresponding graph, with fluxes as edge weights
        #print(G.edges())
        #Dominant flux
        G2=get_dominant_path_G(G,node0)

        edgelist=G2.edges()
        edgelist_str=['%s-%s.%s'%(edge[0],edge[1],ext) for edge in edgelist]

        indices=[np.where(labls_==e)[0][0] for e in edgelist_str]
        dominant_graphs[idx][indices]=1
            
    return dominant_graphs

def get_G_coords_9nodes():
    dx=8
    dy=2
    dx_=-8
    dy_=-2
    coords={0:[0,0],1:[1,2],2:[-2,4],3:[0+dx,0+dy],4:[1+dx,2+dy],5:[-2+dx,4+dy],6:[0+dx_,0+dy_],7:[1+dx_,2+dy_],8:[-2+dx_,4+dy_]}
    return coords

def merge_graphs(Glist):
    all_edges=[]
    [all_edges.extend(G.edges()) for G in Glist] #list with edges for all graphs to merge
    all_weights=[]
    [all_weights.extend([G[e[0]][e[1]]['weight'] for e in G.edges()]) for G in Glist] #list with weights for each edge of all graphs to merge, in the same order as edge name list above
    all_weights=np.array(all_weights)
    unique_edges=list(set(all_edges))
    count=[all_edges.count(e) for e in unique_edges]
    #print(all_edges)
    #print(all_weights)
    #print(unique_edges)
    #print(count)
    U=nx.DiGraph()
    unique_edges_str=np.array(list(map(str,all_edges)))
    #print(unique_edges_str)
    for e in unique_edges:
        idxs=np.where(unique_edges_str==str(e))[0]
        weights=all_weights[idxs]
        average=np.sum(weights)/len(Glist) #average over all graphs. 
        #print("************careful when averaging weights. Confirm that it is correct and remove this warning")
        #print(e)
        #print(weights)
        #print(idxs)
        #print(average)
        U.add_edge(e[0],e[1],weight=average)
    return U

def merge_mirrored_unique_paths(mat_dp,result, idxs_rev):
    newpaths=[] #accounting for mirror
    newcounts=[] #accountng for mirror
    
    mirror=np.zeros(len(mat_dp),dtype=bool)
    pathmat=result[0]
    oldidxs=result[1]
    
    newinverses=oldidxs.copy()
    
    symmetric=np.zeros(len(pathmat)) #each position will either have a 0 or the index corresponding to the path with which it is symmetric

    pathcount=0
    for p1 in range(len(pathmat)):
        if symmetric[p1]<1: #if it is not the symmetric of a path already encountered
            path1=pathmat[p1]
            symcounts=result[2][p1]
            oldpidx=np.where(oldidxs==p1)[0] #original rows that correspond to this path. below this is changed to the new path index
            #print("path", p1,"originally appeared",len(oldpidx))
            for p_idx in range(p1+1,len(pathmat)):
                path2=pathmat[p_idx]
                pmirror=path2[idxs_rev]
                #print("path is", path2)
                #print("symmetric path is",pmirror)
                if np.sum(np.abs(path1-pmirror))<1:
                    #symmetric[p_idx]=p1
                    #print("symmetric found")
                    #for path_ in [path1,path2]:
                    #    for p1_ in range(len(path_)):
                    #        if path_[p1_]>0:
                    #            print(labls_to_edges(labls_[p1_]))
                    #    print(" ")
                    symcounts+=result[2][p_idx]
                    oldpidxs2=np.where(oldidxs==p_idx)
                    newinverses[oldpidxs2]=pathcount
                    mirror[oldpidxs2]=True
                    #print("idx mirror is", p_idx)
                    #print("adding counts, original: ",result[2][p1], result[2][p_idx])
                    symmetric[p_idx]=1
            newcounts.append(symcounts)
            newpaths.append(pathmat[p1])
            newinverses[oldpidx]=pathcount
            #print("final count is", symcounts, len(np.where(newinverses==pathcount)[0]))
            pathcount+=1
    return [np.array(newpaths),newinverses,np.array(newcounts),mirror]

def plot_graphs(Gs,axes,idxref=3,coords=None,color_lists=None,color_args=None,dominant=[],colord="r"):
    """idxref: graph whose nodes will be plotted in all cases, for reference purposes"""
    edges0=[[1,0],[4,3],[7,6]]
    edges1=[[1, 0], [0, 2], [2, 1], [4, 3], [3, 5], [5, 4], [7, 6], [6, 8], [8, 7]]
    edges1_=[edge for edge in edges1 if not edge in edges0]
    edges2=[[1, 2], [4, 5], [7, 8]]
    edges3=[[0, 3], [3, 0], [1, 4], [4, 1], [2, 5], [5, 2], [0, 6], [6, 0], [1, 7], [7, 1], [2, 8], [8, 2]]
    connectstyles=['arc3,rad=-0.8','arc3,rad=-0.25','arc3,rad=0.1','arc3,rad=-0.075']
    for j in range(len(Gs)):
        ax=axes[j]
        if len(Gs)>1:
            G=Gs[idxref]
        else:
            G=Gs[0]
        #first draw all nodes so that locations are comparable
        nx.draw_networkx_nodes(G,nodelist=[0,1,2],pos=coords,ax=ax,node_size=100,node_color='gray',node_shape="_")
        nx.draw_networkx_nodes(G,nodelist=[3,4,5],pos=coords,ax=ax,node_size=100,node_color='gray',node_shape="H")
        nx.draw_networkx_nodes(G,nodelist=[6,7,8],pos=coords,ax=ax,node_size=80,node_color='gray',node_shape="s")

        G=Gs[j]
        
        #edgelist=G.edges() #edges_list[j]
        
        if color_args is None:
            nx.draw_networkx_edges(G,pos=coords,ax=ax,edgelist=edgelist,width=5)

        else:
            Gedges=list(G.edges())
            for el, edges in enumerate([edges0,edges1_,edges2,edges3]):
                edge_color=[]
                edges_=[]
                for e in edges:
                    if tuple(e) in Gedges:
                        edge_color.append(G[e[0]][e[1]]["weight"])
                        edges_.append(e)

                nx.draw_networkx_edges(G,pos=coords,ax=ax,edgelist=edges_,edge_color=edge_color,connectionstyle=connectstyles[el],**color_args,)
                
            
            #if j==3:
                #cax=fig.add_axes([0.3,0.,0.5,0.03])
                #norm=mpl.colors.Normalize(vmin=color_args["edge_vmin"],vmax=color_args["edge_vmax"])
                #cb1  = mpl.colorbar.ColorbarBase(ax,cmap=color_args["edge_cmap"],norm=norm,orientation='horizontal',label='average net flux')
                #print("args are", color_args["edge_vmin"],color_args["edge_vmax"])
                #print(list(zip(edgelist,color_lists[j])))
        ax.set_xticks([])
        ax.set_yticks([])
        if len(dominant)>0:
            G=dominant[j]
            Gedges=list(G.edges())
            for el, edges in enumerate([edges0,edges1_,edges2,edges3]):
                edgespresent=[e for e in edges if tuple(e) in Gedges]
                nx.draw_networkx_edges(G,pos=coords,ax=ax,edgelist=edgespresent,edge_color=colord,connectionstyle=connectstyles[el],width=1)
    return 



def get_G_colors_from_weights(Gs_chunk):
    """This only works if the graph all have the same edges. Better to use the function from merging graphs"""
    nGchunk=len(Gs_chunk)
    J_av_list=[]
    if nGchunk>1:
        for i_ in range(len(Gs_chunk[0])): #for each of the graphs in each parameter set
            mat_aux=np.zeros((nGchunk,len(Gs_chunk[0][i_].edges())))
            for n_ in range(nGchunk):
                #print('edges',Gs_chunk[n_][i_].edges())
                mat_aux[n_]=[Gs_chunk[n_][i_][e[0]][e[1]]['weight'] for e in Gs_chunk[n_][i_].edges()]
                #if i_==0:
                #print('basal fluxes for synergy',np.mean(chunk_si))
                #print(mat_aux)
            avaux=np.mean(mat_aux,axis=0)
            J_av_list.append(avaux)

    else:
        ng=len(Gs_chunk)
        J_av_list=[[np.log10(Gs[i_][e[0]][e[1]]['weight']) for e in G[i_].edges()] for i_ in range(ng)]

    return J_av_list







