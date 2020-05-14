import numpy as np
import os, re


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

def write_pybind_module(fname,parnames,nnodes,TFnames,L,nbsites,indicesC=[None],coeffsC=[None],type="double"):
    if indicesC[0] is None:
        raise ValueError("Please specify the node indices that contribute to the steady state value.")
    elif coeffsC[0] is None:
        raise ValueError("Please specify the indices of the parameter values that multiply each steady state value. Set to -1 if the steady state is to be multiplied by 1.")
    else:
        if len(indicesC)!=len(coeffsC):
            raise ValueError("indicesC and coeffsC should be lists of the same length.")
    
    for row in L:
        for col in row:
            n=0
            for TF in TFnames:
                if TF in col:
                    n+=1
            if n>1:
                raise(ValueError("Code is only prepared for either no or one variable TF per label, but not more than 1. "))

    outf=open(fname,'w')
    outf.write("""
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <iostream>
#include "pos_stp_fromGRF.h"

using namespace std;
using namespace Eigen;
namespace py=pybind11;

""")
    outf.write("typedef %s T;\n"%type)
    outf.write("typedef Eigen::Matrix< T, Eigen::Dynamic, 1 > VectorXd;\n")
    outf.write("typedef Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > MatrixXd;\n")
    outf.write("typedef Eigen::SparseMatrix<T> SM;\n")
    LxTFs=""
    for TF in TFnames:
        LxTFs+=", std::vector<Eigen::Triplet<T>>& Lx%s"%(TF)
    outf.write("""
void pre_laplacianpars(py::array_t<double> parsar, SM& L %s ){
//laplacian matrix with only parameter values, no variables. Also no values for those labels that depend upon variables.

    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;
"""%(LxTFs))
    outf.write("    int n=%d;\n\n"%nnodes)

    #write parameters
    for p, par in enumerate(parnames):
        outf.write("    T %s=pars[%d];\n"%(par,p))
    #outf.write("    L<<")
    
    #write L
    
    outf.write("    std::vector<Eigen::Triplet<T>> clist;\n")
    
    TFliststring="|".join(TFnames)
    pat=re.compile("\\*[%s]"%TFliststring)

    for r,row in enumerate(L):
        for c in range(len(row)):
            data=row[c];
            if data !='0':
                n=0
                for TF in TFnames:
                    if TF in data:
                        n+=1
                if n==0:
                    #only write parameters that are constant 
                    outf.write("    clist.push_back(Eigen::Triplet<T>(%d,%d,%s));\n"%(r,c,data))

    outf.write("""
    Eigen::Triplet<double> trd;
    for (int j=0;j<clist.size();j++){
        trd=clist[j];
        L.insert(trd.row(),trd.col())=trd.value();
    }
    L.makeCompressed();
""")
        
    #write Lx for each TF
    for TF in TFnames:
        print("doing TF", TF)
        for i,row in enumerate(L):
             for j,col in enumerate(row):
                 if TF in col:
                     outf.write("    Lx%s.push_back(Eigen::Triplet<T>(%d,%d,%s));\n"%(TF,i,j,pat.sub("",col)))

    


    outf.write("    return;\n}\n")
    
    
    


    #interfaceps with respect to each TF
    for TF in TFnames:
        
        outf.write("py::array_t<double> interfaceps_%s(py::array_t<double> parsar, py::array_t<double> othervars, int npoints, bool verbose, bool doublecheck ) {\n"%TF)
        outf.write("    auto othervarsbuf=othervars.request();\n")
        outf.write("    double *othervarsC=(double *) othervarsbuf.ptr;\n")
        outf.write("    const int n=%d;\n"%nnodes)
        outf.write("    SM L(n,n);\n")
        LxTFs=""
        for TF2 in TFnames:
            outf.write("    std::vector<Eigen::Triplet<T>> Lx%s;\n"%TF2)
            LxTFs+=", Lx%s"%TF2
        
        outf.write("    L.reserve(VectorXi::Constant(n,%d));\n"%(nbsites+1))
        outf.write("    int i,j;\n")

        j=0
        for TF_ in TFnames:
            #outf.write("    std::vector<Eigen::Triplet<int>> Lx%s;\n"%TF_)
            if TF_!=TF:
                outf.write("    double %sval=othervarsC[%d];\n"%(TF_,j))
                j+=1

        outf.write("    pre_laplacianpars(parsar,L %s);\n"%LxTFs)
        for t,TF_ in enumerate(TFnames):
        #    outf.write("    Lx_%s(Lx%s);\n"%(TF_,TF_))
            if TF_!=TF:
                outf.write("    insert_L_Lx_atval(L,Lx%s,%sval);\n"%(TF_,TF_))
                
        coeffsCstring=["pars[%d]"%idx if idx>=0 else 1 for idx in coeffsC]
        outf.write("""
    vector<double>result;
    vector<int>indicesC={%s};
    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;
    vector<double>coeffsC={%s};
    result={1,1};
    result=compute_pos_stp_fromGRF(L,Lx%s,indicesC,coeffsC,verbose,npoints,doublecheck);
    py::array_t<double> resultpy = py::array_t<double>(2);
    py::buffer_info bufresultpy = resultpy.request();
    double *ptrresultpy=(double *) bufresultpy.ptr;
    ptrresultpy[0]=result[0];
    ptrresultpy[1]=result[1];

    return  resultpy;
    }\n
"""%(",".join(map(str,indicesC)),",".join(coeffsCstring),TF))

    
    #interfacess
    outf.write("double interfacess(py::array_t<double> parsar, py::array_t<double> varvals, bool doublecheck, string method){\n")
    outf.write("    const int n=%d;\n"%nnodes)
    outf.write("    SM L(n,n);\n")
    outf.write("    L.reserve(VectorXi::Constant(n,%d));"%(nbsites+1))
    outf.write("    auto varsbuf=varvals.request();\n")
    outf.write("    double *vars=(double *) varsbuf.ptr;\n")
    LxTFs=""
    for t,TF in enumerate(TFnames):
        outf.write("    double %sval=vars[%d];\n"%(TF,t))
        outf.write("    std::vector<Eigen::Triplet<T>> Lx%s;\n"%TF)
        LxTFs+=", Lx%s"%TF

    outf.write("    pre_laplacianpars(parsar,L %s);\n"%LxTFs)
    #outf.write("    laplacianatX(varvals,L);")
    for TF in TFnames:
        outf.write("    insert_L_Lx_atval(L, Lx%s, %sval);\n"%(TF,TF))
    outf.write("""
    T cs;
    for (int k=0; k<L.outerSize(); ++k) {
        cs=0;
        for(typename Eigen::SparseMatrix<T>::InnerIterator it (L,k); it; ++it){
            cs+=it.value();
        }
        L.insert(k,k)=-cs;
        }
    """)
    outf.write("    double ssval;\n")
    outf.write("""    vector<int>indicesC={%s};
    auto parsarbuf=parsar.request();
    double *pars=(double *) parsarbuf.ptr;
    vector<double>coeffsC={%s};
    if (method=="svd"){
    MatrixXd Ld=MatrixXd(L);
    ssval=ssfromnullspace(Ld,indicesC,coeffsC,doublecheck);
    }else{
    ssval=ssfromnullspace(L,indicesC,coeffsC,doublecheck);
    }
    
    return  ssval;
    }\n
"""%(",".join(map(str,indicesC)),",".join(coeffsCstring)))
    
    #interfacerhos
    outf.write("py::array_t<double> interfacerhos(py::array_t<double> parsar, py::array_t<double> varvals, bool doublecheck, string method){\n")
    outf.write("    const int n=%d;\n"%nnodes)
    outf.write("    SM L(n,n);\n")
    outf.write("    L.reserve(VectorXi::Constant(n,%d));"%(nbsites+1))
    outf.write("    auto varsbuf=varvals.request();\n")
    outf.write("    double *vars=(double *) varsbuf.ptr;\n")
    LxTFs=""
    for t,TF in enumerate(TFnames):
        outf.write("    double %sval=vars[%d];\n"%(TF,t))
        outf.write("    std::vector<Eigen::Triplet<T>> Lx%s;\n"%TF)
        LxTFs+=", Lx%s"%TF

    outf.write("    pre_laplacianpars(parsar,L%s);\n"%LxTFs)
    #outf.write("    laplacianatX(varvals,L);")
    for TF in TFnames:
        outf.write("    insert_L_Lx_atval(L,Lx%s,%sval);\n"%(TF,TF))

    outf.write("""
    T cs;
    for (int k=0; k<L.outerSize(); ++k) {
         cs=0;
            for(typename Eigen::SparseMatrix<T>::InnerIterator it (L,k); it; ++it){
                cs+=it.value();
            }
            L.insert(k,k)=-cs;
            }
    """)
    
    outf.write("""    VectorXd N;
    N.resize(n,1);
    int i;
    if (method=="svd"){
    MatrixXd Ld=MatrixXd(L);
    nullspace(Ld,N,doublecheck);
    }else{
    nullspace(L,N,doublecheck);
    }
    py::array_t<double> resultpy = py::array_t<double>(n);
    py::buffer_info bufresultpy = resultpy.request();
    double *ptrresultpy=(double *) bufresultpy.ptr;
    for (i=0;i<n;i++){
        ptrresultpy[i]=N[i];
    }

    return  resultpy;
    }\n
""")
    
    outf.write("PYBIND11_MODULE(%s,m){\n"%os.path.split(fname)[-1].replace(".cpp",""))
    for TF in TFnames:
        outf.write("""
    m.def("interfaceps_%s", &interfaceps_%s, "A function which returns pos stp.",
    py::arg("parsar"), py::arg("othervars"),py::arg("npoints")=1000, py::arg("verbose")=false,py::arg("doublecheck")=false);\n
    """%(TF,TF))

    outf.write("""m.def("interfacess", &interfacess, "A function which returns ss. Method should be qr (done on sparse matrix) or svd (done on dense matrix).",
            py::arg("parsar"), py::arg("varvals"),py::arg("doublecheck")=false, py::arg("method")="qr");

    m.def("interfacerhos", &interfacerhos, "A function which returns normalised nullspace (sums to 1 already). Method should be qr (done on sparse matrix) or svd (done on dense matrix).",
            py::arg("parsar"), py::arg("varvals"),py::arg("doublecheck")=false, py::arg("method")="qr");
    """)
    outf.write("}\n\n")










