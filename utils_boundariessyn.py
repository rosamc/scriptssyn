import basic
import re
import numpy as np
import pandas as pd
import alphashape
import matplotlib.pyplot as plt

def return_fullparset(parset,case):
    if case=="any":
        parset2=parset.copy()
    elif case=="difAD_difbnp":
        parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[14:16],parset[14:16],parset[14:16]))
    elif case=="difADsbnp" or case=="difAD_samebnp":
        parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14]))
    elif case=="difAD_samebnp_step12":#ni,ia
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step13": #ni,an
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step42": #in,ia #4,2
        parset2=np.concatenate((parset[0:4],parset[0:2],parset[4:5],parset[3:4],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step43": #in,an #4,3
        parset2=np.concatenate((parset[0:4],parset[0:2],parset[4:5],parset[3:4],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step23":#ia,an
        parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))    
    elif case=="difAD_samebnp_step11":
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[5:6],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step22":
        parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="sameAD_difbp":
        parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:20]))
    elif case=="sameAD_difbp_kuonly":
        kb=parset[8]
        ku1,ku2,ku3,ku4,ku5,ku6=parset[9:]
        bindingar=np.array([kb,ku1,kb,ku2,kb,ku3,kb,ku4,kb,ku5,kb,ku6])
        parset2=np.concatenate((parset[0:8],parset[4:8],bindingar))
    elif case=="sameAD_difbnp":
        parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:10],parset[8:10],parset[8:10],parset[10:12],parset[10:12],parset[10:12]))
    elif case=="sameADk1k1_difbnp":#[ktia0,ktan0,ktin0,ktni0,ktniTF,kbA,kuA,kbB,kuB] #4+1+4=9 #in the paper, ktni is called k1, and ktin is called k4
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[4:5],parset[5:7],parset[5:7],parset[5:7],parset[7:9],parset[7:9],parset[7:9]))
    elif case=="sameADk4k4_difbnp":#[ktia0,ktan0,ktin0,ktni0,ktinTF,kbA,kuA,kbB,kuB]#4+1+4=9
        parset2=np.concatenate((parset[0:4],parset[0:2],parset[4:5],parset[3:4],parset[0:2],parset[4:5],parset[3:4],parset[5:7],parset[5:7],parset[5:7],parset[7:9],parset[7:9],parset[7:9]))
    elif case=="sameADk1k1_difbp":#[ktia0,ktan0,ktin0,ktni0,ktniTF,bindings...]#4+1+12=17
        parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[4:5],parset[5:]))
    elif case=="sameADk4k4_difbp":#[ktia0,ktan0,ktin0,ktni0,ktinTF,bindings...]#4+1+12=17
        parset2=np.concatenate((parset[0:4],parset[0:2],parset[4:5],parset[3:4],parset[0:2],parset[4:5],parset[3:4],parset[5:]))
    elif case=="empty":
         parset2=np.concatenate((parset[0:8],parset[0:4],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10]))
    else:
        print("unrecognised case, ", case)
        raise ValueError
    return parset2

def get_constraints_npars(case,fcd=0.01,fcu=100):
    if case=="any":
        npars=24
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
        
    elif case=="difAD_difbnp":
        npars=16
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="difADsbnp" or case=="difAD_samebnp":
        npars=14
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}, 8:{'target':0,'fcd':1,'fcu':fcu},9:{'target':1,'fcd':1,'fcu':fcu},10:{'target':2,'fcd':fcd,'fcu':1},11:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:12],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14],parset[12:14]))
    elif case=="difAD_samebnp_step12":#ni,ia
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':0,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[5:6],parset[1:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step13": #ni,an
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step42":#in,ia
        npars=8
        constraints={4:{'target':2,'fcd':fcd,'fcu':1},5:{'target':0,'fcd':1,'fcu':fcu}}
    elif case=="difAD_samebnp_step43": #in,an
        npars=8
        constraints={4:{'target':2,'fcd':fcd,'fcu':1},5:{'target':1,'fcd':1,'fcu':fcu}}
    elif case=="difAD_samebnp_step23":#ia,an
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu}}
        npars=8
        #parset2=np.concatenate((parset[0:4],parset[4:5],parset[1:4],parset[0:1],parset[5:6],parset[2:4],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step11":
        npars=8
        constraints={4:{'target':3,'fcd':1,'fcu':fcu},5:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:4],parset[0:3],parset[4:5],parset[0:3],parset[5:6],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8],parset[6:8]))
    elif case=="difAD_samebnp_step22":
        npars=8
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':0,'fcd':1,'fcu':fcu}}
    elif case=="sameAD_difbp":
        npars=20
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:20]))
    elif case=="sameAD_difbp_kuonly":
        #kb=parset[8]
        #ku1,ku2,ku3,ku4,ku5,ku6=parset[9:]
        #bindingar=np.array([kb,ku1,kb,ku2,kb,ku3,kb,ku4,kb,ku5,kb,ku6])
        #parset2=np.concatenate((parset[0:8],parset[4:8],bindingar))
        npars=15
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="sameAD_difbnp":
        npars=12
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}}
        #parset2=np.concatenate((parset[0:8],parset[4:8],parset[8:10],parset[8:10],parset[8:10],parset[10:12],parset[10:12],parset[10:12]))
    elif case=="sameADk1k1_difbnp":#[ktia0,ktan0,ktin0,ktni0,ktniTF,kbA,kuA,kbB,kuB] #4+1+4=9 #in the paper, ktni is called k1, and ktin is called k4
        npars=9
        constraints={4:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="sameADk4k4_difbnp":#[ktia0,ktan0,ktin0,ktni0,ktinTF,kbA,kuA,kbB,kuB]#4+1+4=9
        npars=9
        constraints={4:{'target':2, 'fcd':fcd, 'fcu':1}}
    elif case=="sameADk1k1_difbp":#[ktia0,ktan0,ktin0,ktni0,ktniTF,bindings...]#4+1+12=17
        npars=17
        constraints={4:{'target':3,'fcd':1,'fcu':fcu}}
    elif case=="sameADk4k4_difbp":#[ktia0,ktan0,ktin0,ktni0,ktinTF,bindings...]#4+1+12=17
        npars=17
        constraints={4:{'target':2, 'fcd':fcd, 'fcu':1}}
    elif case=="empty":
        npars=10
        constraints={4:{'target':0,'fcd':1,'fcu':fcu},5:{'target':1,'fcd':1,'fcu':fcu},6:{'target':2,'fcd':fcd,'fcu':1},7:{'target':3,'fcd':1,'fcu':fcu}} 
        #parset2=np.concatenate((parset[0:8],parset[0:4],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10],parset[8:10]))

    else:
        print("unrecognised case, ", case)
        raise ValueError
    return [constraints,npars]


def check_constraints(allouts,fcu,fcd,gridspacing=0.025,nchoice=200,fGRF=None):
    """Checks if parameter sets from boundary search fulfill the constraints on:
    - fold change with respect to the basal parameters
    - fold change between steady state mRNA (fc1, fc2)
    - doublechecks the synergy value as well."""
    pat=re.compile("fc1=([0-9\.]*)_fc2=([0-9\.]*)")
    for case in allouts.keys():
        dic=allouts[case]
        for key in dic.keys():
            print(case, key)
            df=dic[key][1] 
            fc1,fc2=pat.findall(key)[0]
            fc1=float(fc1)
            fc2=float(fc2)
            npoints=len(df)
            allpoints=np.arange(0,npoints)
            for i in np.random.choice(allpoints,size=nchoice,replace=False): #just pick 200 points
                row=df.iloc[int(i)]
                fullpars=return_fullparset(row["parameters"],case)
                conditions=[]
                for idx in [4,5,7,8,9,11]:
                    if idx<8:
                        basal=idx-4
                    else:
                        basal=idx-8
                    conditions.append(fullpars[idx]/fullpars[basal]<=fcu)
                    if conditions[-1]==False:
                        if fullpars[idx]/fullpars[basal]>1.05*fcu:
                            print(i, "wrong", idx, fullpars[idx], fullpars[basal], fullpars[idx]/fullpars[basal], fcu)
                for idx in [6,10]:
                    if idx<8:
                        basal=idx-4
                    else:
                        basal=idx-8
                    conditions.append(fullpars[idx]/fullpars[basal]>=fcd)
                    if conditions[-1]==False:
                        if fullpars[idx]/fullpars[basal]<0.905*fcd:
                            print(i, "wrong", idx, fullpars[idx], fullpars[basal], fullpars[idx]/fullpars[basal], fcd)

                synab,synba,m=basic.compute_synergy(fullpars,fGRF,returnm=True,anystronger=True)
                if m[1]>m[2]:
                    if not np.abs(np.abs(synab)-np.abs(row["col"]))<1.01*gridspacing: #this could be readjusted for the current grid spaces
                        print(i, "wrong sab")
                    if not np.abs(np.abs(synba)-np.abs(row["row"]))<1.01*gridspacing:
                        print(i, "wrong sba")
                    if m[1]/m[0]>fc1:
                        print(i, "wrong fc1", m[1]/m[0])
                    if m[1]/m[2]>fc2:
                        print(i, "wrong fc2", m[1]/m[2])

                else:
                    if not np.abs(np.abs(synba)-np.abs(row["col"]))<1.01*gridspacing:
                        print(i, "wrong sba")
                    if not np.abs(np.abs(synab)-np.abs(row["row"]))<1.01*gridspacing:
                        print(i, "wrong sab")
                    if m[2]/m[0]>fc1:
                        print(i, "wrong fc1", m[2]/m[0])
                    if m[2]/m[1]>fc2:
                        print(i, "wrong fc2", m[2]/m[1])


    return 

def find_boundaryline(allouts):
    allashapes=dict()
    for cnum,case in enumerate(allouts.keys()):
        allashapes[case]=dict()
        
        for fc in allouts[case].keys():
            print(case, fc)
            #allashapes[case][base]=dict()
            #for fc in allpoints[case][base].keys():
            if case=="difAD_samebnp_step12" or case=="sameADk1k1_difbnp" or case=="sameADk4k4_difbnp":
                alphalist=[50,10]
                plistdf=allouts[case][fc][1] 
            else:
                alphalist=[1,0.5,0.1]
                plistdf=allouts[case][fc][0][1]
            plist=[plistdf["col"].values,plistdf["row"].values]
            plist2=[[plist[0][i],plist[1][i]] for i in range(len(plist[0]))]
            #ashape=alphashape.alphashape(plist2,alpha=1)
            plt.scatter(plist[0],plist[1],s=5)
            plt.xlabel("SAB")
            plt.ylabel("SBA")
            #[plt.scatter(p[0],p[1]) for p in plist2]

            found=False


            for alpha in alphalist:
                print(alpha)
                if not found:
                    ashape=alphashape.alphashape(plist2,alpha=alpha)
                    try:
                        plt.plot(*ashape.exterior.xy,color='r')
                        plt.title("%s,%s"%(case,fc))
                        plt.show()
                        found=True
                        #if fc=="fc1=5_fc2=2":
                        #    fc="fc1=5_fc2=2.5"#I made a mistake when saving
                        allashapes[case][fc]=ashape
                    except:
                        continue
            if not found:
                print("no ashape, *********",case,fc)
                allashapes[case][fc]=None
                plt.show()
    return allashapes