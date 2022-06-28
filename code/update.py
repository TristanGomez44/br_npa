from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os

def updateBestModel(metricVal,bestMetricVal,exp_id,model_id,bestEpoch,epoch,net,isBetter,worseEpochNb):

    if isBetter(metricVal,bestMetricVal):
        if os.path.exists("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch)):
            os.remove("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch))

        torch.save(net.state_dict(), "../models/{}/model{}_best_epoch{}".format(exp_id,model_id, epoch))
        bestEpoch = epoch
        bestMetricVal = metricVal
        worseEpochNb = 0
    else:
        worseEpochNb += 1

    return bestEpoch,bestMetricVal,worseEpochNb

def catIntermediateVariables(visualDict,intermVarDict,nbVideos):

    intermVarDict["fullAttMap"] = catMap(visualDict,intermVarDict["fullAttMap"],key="attMaps")
    intermVarDict["fullNormSeq"] = catMap(visualDict,intermVarDict["fullNormSeq"],key="norm")

    return intermVarDict
def saveIntermediateVariables(intermVarDict,exp_id,model_id,epoch,mode="val"):

    intermVarDict["fullAttMap"] = saveMap(intermVarDict["fullAttMap"],exp_id,model_id,epoch,mode,key="attMaps")
    intermVarDict["fullNormSeq"] = saveMap(intermVarDict["fullNormSeq"],exp_id,model_id,epoch,mode,key="norm")

    return intermVarDict

def catMap(visualDict,fullMap,key="attMaps"):
    if key in visualDict.keys():

        #In case attention weights are not comprised between 0 and 1
        tens_min = visualDict[key].min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
        tens_max = visualDict[key].max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        map = (visualDict[key]-tens_min)/(tens_max-tens_min)

        if fullMap is None:
            fullMap = (map.cpu()*255).byte()
        else:
            fullMap = torch.cat((fullMap,(map.cpu()*255).byte()),dim=0)

    return fullMap
def saveMap(fullMap,exp_id,model_id,epoch,mode,key="attMaps"):
    if not fullMap is None:
        np.save("../results/{}/{}_{}_epoch{}_{}.npy".format(exp_id,key,model_id,epoch,mode),fullMap.numpy())
        fullMap = None
    return fullMap
