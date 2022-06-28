import math

import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Identity
plt.switch_backend('agg')

from models import resnet
import args
EPS = 0.000001

def buildFeatModel(featModelName, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("resnet") != -1:
        featModel = getattr(resnet, featModelName)(**kwargs)
    else:
        raise ValueError("Unknown model type : ", featModelName)

    return featModel

class GradCamMod(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net = net
        self.layer4 = net.backbone.featMod.layer4
        self.features = net.backbone.featMod

    def forward(self,x):
        feat = self.net.backbone.featMod(x)["x"]

        x = torch.nn.functional.adaptive_avg_pool2d(feat,(1,1))
        x = x.view(x.size(0),-1)
        x = self.net.classificationHead.linLay(x)

        return x

# This class is just the class nn.DataParallel that allow running computation on multiple gpus
# but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super().__init__(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self, backbone, classificationHead):
        super().__init__()
        self.backbone = backbone
        self.classificationHead = classificationHead

    def forward(self, origImgBatch):


        visResDict = self.backbone(origImgBatch)

        resDict = self.classificationHead(visResDict)

        if visResDict != resDict:
            resDict = merge(visResDict,resDict)

        return resDict

def merge(dictA,dictB,suffix=""):
    for key in dictA.keys():
        if key in dictB:
            dictB[key+"_"+suffix] = dictA[key]
        else:
            dictB[key] = dictA[key]
    return dictB

################################# Visual Model ##########################

class Backbone(nn.Module):

    def __init__(self, featModelName,**kwargs):
        super().__init__()

        self.featMod = buildFeatModel(featModelName,**kwargs)

    def forward(self, x):
        raise NotImplementedError

class CNN2D(Backbone):

    def __init__(self, featModelName,**kwargs):
        super().__init__(featModelName,**kwargs)

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        featModRetDict = self.featMod(x)

        features = featModRetDict["x"]

        retDict = {}

        if not "attMaps" in featModRetDict.keys():
            retDict["attMaps"] = torch.sqrt(torch.pow(features,2).sum(dim=1))
            retDict["features"] = features
        else:
            retDict["attMaps"] = featModRetDict["attMaps"]
            retDict["features"] = featModRetDict["features"]

        retDict["x"] = features.mean(dim=-1).mean(dim=-1)

        return retDict

def buildImageAttention(inFeat,outChan=1):
    attention = []
    attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

def representativeVectors(x,nbVec,no_refine=False,randVec=False):

    xOrigShape = x.size()

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    if randVec:
        raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
    else:
        raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for _ in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)

        simNorm = sim/sim.sum(dim=1,keepdim=True)

        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)

        if not no_refine:
            repreVecList.append(reprVec)
        else:
            repreVecList.append(raw_reprVec[:,0])

        if randVec:
            raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
        else:
            raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    return repreVecList,simList

class CNN_attention(Backbone):

    def __init__(self, featModelName,
                 inFeat=512,nb_parts=3,\
                 br_npa=False,no_refine=False,rand_vec=False,vect_ind_to_use="all",\
                 **kwargs):

        super().__init__(featModelName,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not br_npa:
            self.attention = buildImageAttention(inFeat,nb_parts+1)
        else:
            self.attention = None

        self.nb_parts = nb_parts
        self.br_npa = br_npa
        if not br_npa:
            self.attention_activation = torch.relu
        else:
            self.attention_activation = None
            self.no_refine = no_refine
            self.rand_vec = rand_vec

        self.softmSched_interpCoeff = 0

        self.vect_ind_to_use = vect_ind_to_use

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        output = self.featMod(x)

        features = output["x"]

        retDict = {}

        if not self.br_npa:
            spatialWeights = self.attention_activation(self.attention(features))
            features_weig = (spatialWeights[:,:self.nb_parts].unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),features.size(1)*(spatialWeights.size(1)-1),features.size(2),features.size(3))
            features_agr = self.avgpool(features_weig)
            features_agr = features_agr.view(features.size(0), -1)
        else:
            vecList,simList = representativeVectors(features,self.nb_parts,\
                                                    self.no_refine,self.rand_vec)


            features_agr = torch.cat(vecList,dim=-1)

            spatialWeights = torch.cat(simList,dim=1)

        retDict["x"] = features_agr
        retDict["attMaps"] = spatialWeights
        retDict["features"] = features

        return retDict

################################ Temporal Model ########################""

class ClassificationHead(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super().__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearClassificationHead(ClassificationHead):

    def __init__(self, nbFeat, nbClass, dropout,\
                        bias=True,aux_on_masked=False,num_parts=None):

        super().__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)

        self.linLay = nn.Linear(self.nbFeat, self.nbClass,bias=bias)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_on_masked = aux_on_masked
        if self.aux_on_masked:
            self.lin01 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin12 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin0 = nn.Linear(nbFeat//3,nbClass)
            self.lin1 = nn.Linear(nbFeat//3,nbClass)
            self.lin2 = nn.Linear(nbFeat//3,nbClass)

    def forward(self, visResDict):

        x = visResDict["x"]

        if len(x.size()) == 4:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)

        x = self.dropout(x)

        pred = self.linLay(x)

        retDict = {"pred": pred}

        if self.aux_on_masked:
            retDict["pred_01"] = self.lin01(x[:,:int(self.nbFeat*2/3)].detach())
            retDict["pred_12"] = self.lin12(x[:,int(self.nbFeat*1/3):].detach())
            retDict["pred_0"] = self.lin0(x[:,:int(self.nbFeat*1/3)].detach())
            retDict["pred_1"] = self.lin1(x[:,int(self.nbFeat*1/3):int(self.nbFeat*2/3)].detach())
            retDict["pred_2"] = self.lin2(x[:,int(self.nbFeat*2/3):].detach())

        return retDict

def getResnetFeat(backbone_name, backbone_inplanes):
    if backbone_name in ["resnet50","resnet101","resnet152"]:
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name in ["resnet18","resnet34"]:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args):
    ############### Visual Model #######################

    nbFeat = getResnetFeat(args.backbone, args.resnet_chan)

    if args.attention:
        CNNconst = CNN_attention
        kwargs = {"inFeat":nbFeat,"nb_parts":args.nb_att_maps,\
                    "br_npa":args.br_npa,
                    "no_refine":args.br_npa_norefine,\
                    "rand_vec":args.br_npa_randvec}

        nbFeat *= args.nb_att_maps

    else:
        CNNconst = CNN2D
        kwargs = {}

    backbone = CNNconst(args.backbone,chan=args.resnet_chan, stride=args.resnet_stride,\
                                strideLay2=args.stride_lay2,strideLay3=args.stride_lay3,\
                                strideLay4=args.stride_lay4,\
                                endRelu=args.end_relu,\
                                **kwargs)

    ############### Second Model #######################
    if args.classification_head == "linear":
        classificationHead = LinearClassificationHead(nbFeat, args.class_nb, args.dropout,\
                                            bias=args.lin_lay_bias,aux_on_masked=args.aux_on_masked,num_parts=args.nb_att_maps)
    else:
        raise ValueError("Unknown classification head type : ", args.classification_head)

    ############### Whole Model ##########################

    net = Model(backbone, classificationHead)

    if args.cuda:
        net.cuda()
    if args.multi_gpu:
        net = DataParallelModel(net)

    return net

def addArgs(argreader):
    argreader.parser.add_argument('--backbone', type=str, metavar='MOD',
                                  help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float, metavar='D',
                                  help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--classification_head', type=str, metavar='MOD',
                                  help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')

    argreader.parser.add_argument('--stride_lay2', type=int, metavar='NB',
                                  help='Stride for layer 2.')
    argreader.parser.add_argument('--stride_lay3', type=int, metavar='NB',
                                  help='Stride for layer 3.')
    argreader.parser.add_argument('--stride_lay4', type=int, metavar='NB',
                                  help='Stride for layer 4.')

    argreader.parser.add_argument('--nb_att_maps', type=int, metavar='INT',
                                  help="The number of attention maps model.")
    argreader.parser.add_argument('--attention', type=args.str2bool, metavar='BOOL',
                                  help="To use attention")

    argreader.parser.add_argument('--br_npa', type=args.str2bool, metavar='BOOL',
                                  help="To use BR-NPA instead of B-CNN when --attention is True")

    argreader.parser.add_argument('--br_npa_norefine', type=args.str2bool, metavar='BOOL',
                                  help="To not refine feature vectors by using similar vectors.")
    argreader.parser.add_argument('--br_npa_randvec', type=args.str2bool, metavar='BOOL',
                                  help="To select random vectors as initial estimation instead of vectors with high norms.")

    argreader.parser.add_argument('--lin_lay_bias', type=args.str2bool, metavar='BOOL',
                                  help="To add a bias to the final layer.")

    argreader.parser.add_argument('--aux_on_masked', type=args.str2bool, metavar='BOOL',
                                  help="To train dense layers on masked version of the feature matrix.")

    argreader.parser.add_argument('--master_net', type=args.str2bool, help='To distill a master network into the trained network.')
    argreader.parser.add_argument('--m_model_id', type=str, help='The model id of the master network')
    argreader.parser.add_argument('--kl_interp', type=float, help='If set to 0, will use regular target, if set to 1, will only use master net target')
    argreader.parser.add_argument('--kl_temp', type=float, help='KL temperature.')

    argreader.parser.add_argument('--end_relu', type=args.str2bool, help='To add a relu at the end of the first block of each layer.')

    return argreader
