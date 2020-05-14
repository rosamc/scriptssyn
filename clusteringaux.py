import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

def get_cuts_frommat(mati,dtype='numeric'):
    """Identify when there is a change in group cluster on a matrix sorted after a clustering algorithm 
    (only makes sense for matrices with discrete sets of values. Can also be the array with the cluster indices, if all elements of a given cluster are together )."""
    
    
    previous=mati[0].copy()
    nc=0
    cut=[0]
    
    if dtype != 'numeric': #means it is a sequence of characters
        for j in range(1,len(mati)):
            current=mati[j].copy()
            if np.any(previous != current):
                previous=current
                nc+=1
                cut.append(j)
        cut.append(len(mati))
    
    else:

        #find the edges two contiguous clusters
        for j in range(1,len(mati)):
            current=mati[j].copy()
            if np.sum(np.abs(current-previous))>0:
                previous=current
                nc+=1
                cut.append(j)
        cut.append(len(mati))

        #print('nc',nc)
        #print('cut',cut)
        #dif=np.diff(cut)
        #print('dif',dif)
        #difs_common=np.where(dif>=threshold)[0] #
        #difs_uncommon=np.where(dif<threshold)[0] #
    return cut

def get_order_from_labls(labls_,array):
    """Sort clusters according to the average value from array. Return indices that sorts each item that was in the original matrix used for clustering, as well as the boundaries. """
    
    idxs_=np.argsort(labls_) #list with all cluster idxs, sorted by increasing cluster number
    clusters=labls_[idxs_] #list where all elements in a given cluster are together.The cluster order is increasing by cluster number
    array_sorted=array[idxs_]
    cuts=get_cuts_frommat(clusters) #gives indices separating each cluster from the next
    #to test:
    #print(get_order_from_labls)
    #print(cuts, len(cuts)-1)

    means=[np.mean(array_sorted[cuts[c]:cuts[c+1]]) for c in range(len(cuts)-1)]
    #print(means)
    argsort_clusters=np.argsort(means) #gives the sorting order for the clusteres ordered by their cluster number
    
    #sort clusters by the average value of the quantity in array
    original_cluster_order=[]
    for l in labls_:
        if not l in original_cluster_order:
            original_cluster_order.append(l)
    original_cluster_order=np.array(original_cluster_order) 
    argsort1=np.argsort(original_cluster_order) #indices that sort the initial clusters by increasing order, as in the first line
    
    
    idxs=[]
    cuts=[]
    n=0 #counter for the divisions between consecutive clusters
    for a in original_cluster_order[argsort1][argsort_clusters]: #for each cluster id, sorted such that the average of the values in array corresponding to this cluster is increasing when changing from one cluster to the next
        #print('a',a)
        cluster_idxs=[]
        for lidx,lab in enumerate(labls_):
            if lab==a:
                #print('idx',lidx)
                #idxs.append(lidx)
                cluster_idxs.append(lidx)
                n+=1
        #now sort the entities within the cluster according to array value
        argsort=np.argsort(array[cluster_idxs])
        idxs.extend(np.array(cluster_idxs)[argsort])
        cuts.append(n)

    return [idxs,cuts]


def clustermat(mat,library='scipy',axd=None,clustargs=None,linkargs=None,arrayforsort=None):
    if library=='scipy':
        dend = shc.dendrogram(shc.linkage(mat,**linkargs),**clustargs)
        idxs=np.array(dend['ivl'],dtype='int')
        return [idxs,None]
    elif library=='scikit':
        ward = AgglomerativeClustering(**clustargs)
        ward.fit(mat)
        if arrayforsort is not None:
            idxs,cuts=get_order_from_labls(ward.labels_,arrayforsort)
            return [idxs,cuts]
        else:
            idxs=np.argsort(ward.labels_)
            return [idxs,None]
        