import numpy as np

def return_parsar_samebindingall(pars,constraint='any',f=np.mean,kofffixed=1):
    """f should be sindep (independent synergystic effect, np.mean, np.max)"""
    
    if constraint=='any':
        if not f in ['sindep',np.mean, np.max, 'omegakin']:
            raise ValueError("Wrong f")
        if f!='omegakin':
            if kofffixed>0:
                k_1_0,k_2_0,k_3_0,a1,a2,a3,b1,b2,b3,kon=pars
                koff=kofffixed
            else:
                k_1_0,k_2_0,k_3_0,a1,a2,a3,b1,b2,b3,kon,koff=pars #11 pars
        
            k_1_1=a1
            k_2_1=a2 #10**a2
            k_3_1=a3 #10**a3
            k_1_2=b1#10**b1
            k_2_2=b2
            k_3_2=b3#10**b3

            if f=='sindep':

                k_1_1U2=max(a1+b1-k_1_0,0)
                k_2_1U2=max(a2+b2-k_2_0,0)
                k_3_1U2=max(a3+b3-k_3_0,0)
            
            else:
                k_1_1U2=f([k_1_1,k_1_2])
                k_2_1U2=f([k_2_1,k_2_2])
                k_3_1U2=f([k_3_1,k_3_2])
        else:
            if kofffixed>0:
                k_1_0,k_2_0,k_3_0,a1,a2,a3,b1,b2,b3,k_1_1U2,k_2_1U2,k_3_1U2,kon=pars
                koff=kofffixed
            else:
                k_1_0,k_2_0,k_3_0,a1,a2,a3,b1,b2,b3,k_1_1U2,k_2_1U2,k_3_1U2,kon,koff=pars #11 pars
            k_1_1=a1
            k_2_1=a2 #10**a2
            k_3_1=a3 #10**a3
            k_1_2=b1#10**b1
            k_2_2=b2
            k_3_2=b3#10**b3

    elif constraint=='one':
        if kofffixed>0:
            k_1_0,k_2_0,k_3_0,a1,b2,kon=pars
            koff=kofffixed
        else:
            k_1_0,k_2_0,k_3_0,a1,b2,kon,koff=pars
    
        k_1_1=a1
        k_2_1=k_2_0 #10**a2
        k_3_1=k_3_0 #10**a3
        k_1_2=k_1_0#10**b1
        k_2_2=b2
        k_3_2=k_3_0#10**b3
        
        k_1_1U2=k_1_1
        k_2_1U2=k_2_2
        k_3_1U2=k_3_0
    
    #koff=1

    
    parvalues=np.array([kon,k_1_0,kon,k_2_0,kon,k_3_0,koff,k_1_1,koff,k_2_1,koff,k_3_1,kon,kon,kon,koff,k_1_2,koff,k_2_2,koff,k_3_2,kon,kon,kon,koff,k_1_1U2,koff,k_2_1U2,koff,k_3_1U2,kon,kon,kon,koff,koff,koff])
    
    return parvalues


def return_parsar_samebindingall_old(pars,constraint='any',f=np.mean):
    if constraint=='any':
        k_1_0,k_2_0,k_3_0,kon,a1,a2,a3,b1,b2,b3=pars
        #koff=1
    
        k_1_1=a1
        k_2_1=a2 #10**a2
        k_3_1=a3 #10**a3
        k_1_2=b1#10**b1
        k_2_2=b2
        k_3_2=b3#10**b3
        
        k_1_1U2=f([k_1_1,k_1_2])
        k_2_1U2=f([k_2_1,k_2_2])
        k_3_1U2=f([k_3_1,k_3_2])
    elif constraint=='one':
        k_1_0,k_2_0,k_3_0,kon,a1,b2=pars
        #koff=kofffixed
    
        k_1_1=a1
        k_2_1=k_2_0 #10**a2
        k_3_1=k_3_0 #10**a3
        k_1_2=k_1_0#10**b1
        k_2_2=b2
        k_3_2=k_3_0#10**b3
        
        k_1_1U2=k_1_1
        k_2_1U2=k_2_2
        k_3_1U2=k_3_0
    
    koff=1

    
    parvalues=np.array([kon,k_1_0,kon,k_2_0,kon,k_3_0,koff,k_1_1,koff,k_2_1,koff,k_3_1,kon,kon,kon,koff,k_1_2,koff,k_2_2,koff,k_3_2,kon,kon,kon,koff,k_1_1U2,koff,k_2_1U2,koff,k_3_1U2,kon,kon,kon,koff,koff,koff])
    
    return parvalues