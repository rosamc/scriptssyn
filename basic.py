import numpy as np

def compute_synergy(pars,f,ftype='crit',Aval=1.0, Bval=1.0,returnm=False, anystronger=False):
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
        else:
            varar=np.array([A,B])
            m=f(pars,varar)
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
        SAB=np.log2(mstars[3]/mstars[1])
        SBA=np.log2(mstars[3]/mstars[2])
        if returnm:
            return [SAB,SBA,m]
        else:
            #print("Afirst")
            return [SAB,SBA]

