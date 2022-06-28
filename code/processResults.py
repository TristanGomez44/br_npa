import os
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
plt.switch_backend('agg')
import matplotlib
import matplotlib.cm as cm
import scipy
import scipy.stats

from args import ArgReader
import load_data

def attMetrics(exp_id,metric="del"):

    suff = metric

    paths = sorted(glob.glob("../results/{}/attMetr{}_*.npy".format(exp_id,suff)))

    resDic = {}
    resDic_pop = {}

    if metric in ["Del","Add"]:
        for path in paths:

            model_id = os.path.basename(path).split("attMetr{}_".format(suff))[1].split(".npy")[0]
            
            pairs = np.load(path,allow_pickle=True)

            allAuC = []

            for i in range(len(pairs)):

                pairs_i = np.array(pairs[i])

                if metric == "Add":
                    pairs_i[:,0] = 1-pairs_i[:,0]/pairs_i[:,0].max()
                else:
                    pairs_i[:,0] = (pairs_i[:,0]-pairs_i[:,0].min())/(pairs_i[:,0].max()-pairs_i[:,0].min())
                    pairs_i[:,0] = 1-pairs_i[:,0]

                auc = np.trapz(pairs_i[:,1],pairs_i[:,0])
                allAuC.append(auc)
        
            resDic_pop[model_id] = np.array(allAuC)
            resDic[model_id] = resDic_pop[model_id].mean()

    csv = "\n".join(["{},{}".format(key,resDic[key]) for key in resDic])
    with open("../results/{}/attMetrics_{}.csv".format(exp_id,suff),"w") as file:
        print(csv,file=file)

    csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key].astype("str"))) for key in resDic_pop])
    with open("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),"w") as file:
        print(csv,file=file)

def getIndsToUse(paths,metric):
    
    modelToIgn = []

    model_targ_ind = 0

    while model_targ_ind < len(paths) and not os.path.exists(paths[model_targ_ind].replace("Add","Targ").replace("Del","Targ")):
        model_targ_ind += 1

    if model_targ_ind == len(paths):
        use_all_inds = True
    else:
        use_all_inds = False 
        targs = np.load(paths[model_targ_ind],allow_pickle=True)
        
        indsToUseBool = np.array([True for _ in range(len(targs))])
        indsToUseDic = {}

    for path in paths:
        
        model_id = os.path.basename(path).split("attMetr{}_".format(metric))[1].split(".npy")[0]
        
        model_id_nosuff = model_id.replace("-max","").replace("-onlyfirst","").replace("-fewsteps","")

        predPath = path.replace(metric,"Preds").replace(model_id,model_id_nosuff)

        if not os.path.exists(predPath):
            predPath = path.replace(metric,"PredsAdd").replace(model_id,model_id_nosuff)

        if os.path.exists(predPath) and not use_all_inds:
            preds = np.load(predPath,allow_pickle=True)

            if preds.shape != targs.shape:
                inds = []
                for i in range(len(preds)):
                    if i % 2 == 0:
                        inds.append(i)

                preds = preds[inds]
            
            indsToUseDic[model_id] = np.argwhere(preds==targs)
            indsToUseBool = indsToUseBool*(preds==targs)
  
        else:
            modelToIgn.append(model_id)
            print("no predpath",predPath)

    if use_all_inds:
        indsToUse = None
    else:
        indsToUse =  np.argwhere(indsToUseBool)
        
    return indsToUse,modelToIgn 

def ttest_attMetr(exp_id,metric="del"):

    suff = metric

    print("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff))
    arr = np.genfromtxt("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),dtype=str,delimiter=",")

    if len(arr.shape) == 1:
        arr = [arr]

    arr = best_to_worst(arr,ascending=metric in ["Add","Spars","Acc"])

    model_ids = arr[:,0]

    res_mat = arr[:,1:].astype("float")

    p_val_mat = np.zeros((len(res_mat),len(res_mat)))
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            p_val_mat[i,j] = scipy.stats.ttest_ind(res_mat[i],res_mat[j],equal_var=False)[1]

    p_val_mat = (p_val_mat<0.05)

    res_mat_mean = res_mat.mean(axis=1)

    diff_mat = np.abs(res_mat_mean[np.newaxis]-res_mat_mean[:,np.newaxis])
    
    diff_mat_norm = (diff_mat-diff_mat.min())/(diff_mat.max()-diff_mat.min())

    cmap = plt.get_cmap('plasma')

    fig = plt.figure()

    ax = fig.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.imshow(p_val_mat*0,cmap="Greys")
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            if i <= j:
                rad = 0.3 if p_val_mat[i,j] else 0.1
                circle = plt.Circle((i, j), rad, color=cmap(diff_mat_norm[i,j]))
                ax.add_patch(circle)

    plt.yticks(np.arange(len(res_mat)),model_ids)
    plt.xticks(np.arange(len(res_mat)),["" for _ in range(len(res_mat))])
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(diff_mat.min(),diff_mat.max()),cmap=cmap))
    for i in range(len(res_mat)):
        plt.text(i-0.2,i-0.4,model_ids[i],rotation=45,ha="left")
    plt.tight_layout()
    plt.savefig("../vis/{}/ttest_{}_attmetr.png".format(exp_id,suff))

def best_to_worst(arr,ascending=True):

    if not ascending:
        key = lambda x:-x[1:].astype("float").mean()
    else:
        key = lambda x:x[1:].astype("float").mean()

    arr = np.array(sorted(arr,key=key))

    return arr

def reverseLabDic(id_to_label,exp_id):

    label_to_id = {}

    for id in id_to_label:
        label = id_to_label[id]

        if label == "BR-NPA":
            if exp_id == "CUB10":
                id = "clus_masterClusRed"
            else:
                id = "clus_mast"
        elif id.startswith("noneRed"):
            id = "noneRed"

        label_to_id[label] = id 
    
    return label_to_id

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    attMetrics(args.exp_id,metric="Add")
    attMetrics(args.exp_id,metric="Del")
            
    ttest_attMetr(args.exp_id,metric="Add")
    ttest_attMetr(args.exp_id,metric="Del")

if __name__ == "__main__":
    main()
