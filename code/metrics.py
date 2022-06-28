
import torch
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,resDict,comp_spars=False):

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    #Accuracy for auxiliary heads for ablation study
    for key in resDict.keys():
        if key.find("pred_") != -1:
            suff = key.split("_")[-1]
            metDict["Accuracy_{}".format(suff)] = compAccuracy(resDict[key],target)

    if "attMaps" in resDict.keys() and comp_spars:
        spar = compAttMapSparsity(resDict["attMaps"].clone(),resDict["features"].clone())
        metDict["Sparsity"] = spar
    elif comp_spars:
        norm = torch.sqrt(torch.pow(resDict["features"],2).sum(dim=1,keepdim=True))
        spar = compSparsity(norm)
        metDict["Sparsity"] = spar 

    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

def compSparsity(norm):
    norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    norm = norm/(norm_max+0.00001)
    sparsity = norm.mean(dim=(2,3))
    return sparsity.sum().item()

def compAttMapSparsity(attMaps,features=None):
    if not features is None:
        norm = torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))
        norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
        norm = norm/norm_max

        attMaps = attMaps*norm

    max = attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    attMaps = attMaps/(max+0.00001)

    if attMaps.size(1) > 1:
        attMaps = attMaps.mean(dim=1,keepdim=True)

    sparsity = attMaps.mean(dim=(2,3))

    return sparsity.sum().item()