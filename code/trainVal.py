import os,sys,glob
import time
import configparser
from shutil import copyfile

import numpy as np
import torch
from torch.nn import functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import optuna
import sqlite3

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
import update

OPTIM_LIST = ["Adam", "AMSGrad", "SGD"]

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def epochSeqTr(model, optim, log_interval, loader, epoch, args, **kwargs):

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0
 
    acc_size = 0
    acc_nb = 0

    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if batch_idx % log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]

        #To accumulate gradients
        if acc_size + data.size(0) > args.batch_size:
            if args.batch_size-acc_size < 2*torch.cuda.device_count():
                data = data[:2*torch.cuda.device_count()]
                target = target[:2*torch.cuda.device_count()]
            else:
                data = data[:args.batch_size-acc_size]
                target = target[:args.batch_size-acc_size]
            acc_size = args.batch_size
        else:
            acc_size += data.size(0)
        acc_nb += 1

        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        resDict = model(data)
        output = resDict["pred"]

        if args.master_net:
            with torch.no_grad():
                mastDict = kwargs["master_net"](data)
                resDict["master_net_pred"] = mastDict["pred"]
                resDict["master_net_attMaps"] = mastDict["attMaps"]
                resDict["master_net_features"] = mastDict["features"]

        loss = kwargs["lossFunc"](output, target, resDict, data).mean()
        loss.backward()

        #Optimizer step
        if acc_size == args.batch_size:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= acc_nb
            optim.step()
            optim.zero_grad()
            acc_size = 0
            acc_nb = 0

        loss = loss.detach().data.item()

        optim.step()

        # Metrics
        with torch.no_grad():
            metDictSample = metrics.binaryToMetrics(output, target,resDict)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch > 3 and args.debug:
            break

    # If the training set is empty (which we might want to just evaluate the model), then allOut and allGT will still be None
    if validBatch > 0:
        if not args.optuna:
            torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
            writeSummaries(metrDict, totalImgNb, epoch, "train", args.model_id, args.exp_id)


class Loss(torch.nn.Module):

    def __init__(self,args,reduction="mean"):
        super(Loss, self).__init__()
        self.args = args
        self.reduction = reduction

    def forward(self,output,target,resDict,data):
        return computeLoss(self.args,output, target, resDict, data,reduction=self.reduction).unsqueeze(0)

def computeLoss(args, output, target, resDict, data,reduction="mean"):

    if not args.master_net:
        loss = args.nll_weight * F.cross_entropy(output, target,reduction=reduction)

    else:
        kl = F.kl_div(F.log_softmax(output/args.kl_temp, dim=1),F.softmax(resDict["master_net_pred"]/args.kl_temp, dim=1),reduction="batchmean")
        ce = F.cross_entropy(output, target)
        loss = args.nll_weight*(kl*args.kl_interp*args.kl_temp*args.kl_temp+ce*(1-args.kl_interp))

    #Loss for auxiliary heads for ablation study
    for key in resDict.keys():
        if key.find("pred_") != -1:
            loss += args.nll_weight * F.cross_entropy(resDict[key], target)

    loss = loss

    return loss

def epochImgEval(model, log_interval, loader, epoch, args, metricEarlyStop, mode="val",**kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network

    '''
    
    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    intermVarDict = {"fullAttMap": None, "fullFeatMapSeq": None, "fullNormSeq":None}

    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        seg=None
        path_list = None

        # Puting tensors on cuda
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        # Computing predictions
        resDict = model(data)

        output = resDict["pred"]

        if args.master_net:
            mastDict = kwargs["master_net"](data)
            resDict["master_net_pred"] = mastDict["pred"]
            resDict["master_net_attMaps"] = mastDict["attMaps"]
            resDict["master_net_features"] = mastDict["features"]

        # Loss
        loss = kwargs["lossFunc"](output, target, resDict, data).mean()

        # Other variables produced by the net
        if mode == "test":
            resDict["norm"] = torch.sqrt(torch.pow(resDict["features"],2).sum(dim=1,keepdim=True))
            intermVarDict = update.catIntermediateVariables(resDict, intermVarDict, validBatch)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target,resDict,comp_spars=(mode=="test"))

        metDictSample["Loss"] = loss

        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch  > 3 and args.debug:
            break

    if mode == "test":
        intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id, epoch, mode)

    writeSummaries(metrDict, totalImgNb, epoch, mode, args.model_id, args.exp_id)

    return metrDict[metricEarlyStop]

def writeSummaries(metrDict, totalImgNb, epoch, mode, model_id, exp_id):
  
    for metric in metrDict.keys():
        metrDict[metric] /= totalImgNb

    header = ",".join([metric.lower().replace(" ", "_") for metric in metrDict.keys()])

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id, model_id, epoch, mode), "a") as text_file:
        print(header, file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def getOptim_and_Scheduler(optimStr, lr,momentum,weightDecay,useScheduler,maxEpoch,lastEpoch,net):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim, optimStr)
        if optimStr == "SGD":
            kwargs = {'lr':lr,'momentum': momentum,"weight_decay":weightDecay}
        elif optimStr == "Adam":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':lr,'amsgrad': True,"weight_decay":weightDecay}

    optim = optimConst(net.parameters(), **kwargs)

    if useScheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.9)
        print("Sched",scheduler.get_last_lr())
        for _ in range(lastEpoch):
            scheduler.step()
        print("Sched",scheduler.get_last_lr())
    else:
        scheduler = None

    return optim, scheduler

def initialize_Net_And_EpochNumber(net, exp_id, model_id):
    # Saving initial parameters
    torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id, model_id))
    startEpoch = 1

    return startEpoch

def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=float, metavar='LR',
                                  help='learning rate')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                                  help='SGD momentum')
    argreader.parser.add_argument('--weight_decay', type=float, metavar='M',
                                  help='Weight decay')
    argreader.parser.add_argument('--use_scheduler', type=args.str2bool, metavar='M',
                                  help='To use a learning rate scheduler')
    argreader.parser.add_argument('--always_sched', type=args.str2bool, metavar='M',
                                  help='To always use a learning rate scheduler when optimizing hyper params')

    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                                  help='the optimizer to use (default: \'SGD\')')

    return argreader


def addValArgs(argreader):

    argreader.parser.add_argument('--metric_early_stop', type=str, metavar='METR',
                                  help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=args.str2bool, metavar='BOOL',
                                  help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=int, metavar='NB',
                                  help='The number of epochs to wait if the validation performance does not improve.')
    argreader.parser.add_argument('--run_test', type=args.str2bool, metavar='NB',
                                  help='Evaluate the model on the test set')

    return argreader


def addLossTermArgs(argreader):
    argreader.parser.add_argument('--nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function.')
    return argreader

def preprocessAndLoadParams(init_path,cuda,net):
    print("Loading from",init_path)
    params = torch.load(init_path, map_location="cpu" if not cuda else None)
    res = net.load_state_dict(params, False)

    # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
    if not res is None:
        missingKeys, unexpectedKeys = res
        if len(missingKeys) > 0:
            print("missing keys")
            for key in missingKeys:
                print(key)
        if len(unexpectedKeys) > 0:
            print("unexpected keys")
            for key in unexpectedKeys:
                print(key)

    return net

def removeConvDense(params):

    keyToRemove = []

    for key in params:
        if key.find("featMod.fc") != -1:
            keyToRemove.append(key)
    for key in keyToRemove:
        params.pop(key,None)

    return params

def addOrRemoveModule(params,net):
    # Checking if the key of the model start with "module."
    startsWithModule = (list(net.state_dict().keys())[0].find("module.") == 0)

    if startsWithModule:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = "module." + key if key.find("module") == -1 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    else:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = key.split('.')
            if keyFormat[0] == 'module':
                keyFormat = '.'.join(keyFormat[1:])
            else:
                keyFormat = '.'.join(keyFormat)
            # keyFormat = key.replace("module.", "") if key.find("module.") == 0 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    return params

def initMasterNet(args):
    config = configparser.ConfigParser()

    config.read("../models/{}/{}.ini".format(args.exp_id,args.m_model_id))
    args_master = Bunch(config["default"])

    args_master.multi_gpu = args.multi_gpu
    args_master.distributed = args.distributed

    argDic = args.__dict__
    mastDic = args_master.__dict__

    for arg in mastDic:
        if arg in argDic:
            if not argDic[arg] is None:
                if not type(argDic[arg]) is bool:
                    if mastDic[arg] != "None":
                        mastDic[arg] = type(argDic[arg])(mastDic[arg])
                    else:
                        mastDic[arg] = None
                else:
                    if arg != "multi_gpu" and arg != "distributed":
                        mastDic[arg] = str2bool(mastDic[arg]) if mastDic[arg] != "None" else False
            else:
                mastDic[arg] = None

    for arg in argDic:
        if not arg in mastDic:
            mastDic[arg] = argDic[arg]

    master_net = modelBuilder.netBuilder(args_master)

    best_paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.m_model_id))

    if len(best_paths) > 1:
        raise ValueError("Too many best path for master")
    if len(best_paths) == 0:
        print("Missing best path for master")
    else:
        bestPath = best_paths[0]
        params = torch.load(bestPath, map_location="cpu" if not args.cuda else None)

        params = removeConvDense(params)
        master_net.load_state_dict(params, strict=True)

    master_net.eval()

    return master_net

def run(args,trial=None):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not trial is None:
        args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        args.optim = trial.suggest_categorical("optim", OPTIM_LIST)

        if args.distributed:
            if args.max_batch_size <= 12//torch.cuda.device_count():
                minBS = 1
            else:
                minBS = 12//torch.cuda.device_count()
        else:
            if args.max_batch_size <= 12:
                minBS = 4
            else:
                minBS = 12
        print(minBS,args.distributed)
        args.batch_size = trial.suggest_int("batch_size", minBS, args.max_batch_size, log=True)
        print("Batch size is ",args.batch_size)
        args.dropout = trial.suggest_float("dropout", 0, 0.6,step=0.2)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        if args.optim == "SGD":
            args.momentum = trial.suggest_float("momentum", 0., 0.9,step=0.1)
            if not args.always_sched:
                args.use_scheduler = trial.suggest_categorical("use_scheduler",[True,False])
    
        if args.always_sched:
            args.use_scheduler = True

        if args.opt_data_aug:
            args.brightness = trial.suggest_float("brightness", 0, 0.5, step=0.05)
            args.saturation = trial.suggest_float("saturation", 0, 0.9, step=0.1)
            args.crop_ratio = trial.suggest_float("crop_ratio", 0.8, 1, step=0.05)

        if args.master_net:
            args.kl_temp = trial.suggest_float("kl_temp", 1, 21, step=5)
            args.kl_interp = trial.suggest_float("kl_interp", 0.1, 1, step=0.1)

            if args.transfer_att_maps:
                args.att_weights = trial.suggest_float("att_weights",0.001,0.5,log=True)

        if args.opt_att_maps_nb:
            args.resnet_bil_nb_parts = trial.suggest_int("resnet_bil_nb_parts", 3, 64, log=True)

    args.world_size = 1
    value = train(args,trial)
    return value

def train(args,trial):

    if not trial is None:
        args.trial_id = trial.number

    if not args.only_test:
        trainLoader,_ = load_data.buildTrainLoader(args)
    else:
        trainLoader = None
    valLoader,_ = load_data.buildTestLoader(args,"val")

    # Building the net
    net = modelBuilder.netBuilder(args)

    trainFunc = epochSeqTr
    valFunc = epochImgEval

    kwargsTr = {'log_interval': args.log_interval, 'loader': trainLoader, 'args': args}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    startEpoch = initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id)

    kwargsTr["optim"],scheduler = getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,args.epochs,startEpoch,net)

    epoch = startEpoch
    bestEpoch, worseEpochNb = epoch,0

    if args.maximise_val_metric:
        bestMetricVal = -np.inf
        isBetter = lambda x, y: x > y
    else:
        bestMetricVal = np.inf
        isBetter = lambda x, y: x < y

    if args.master_net:
        kwargsTr["master_net"] = initMasterNet(args)
        kwargsVal["master_net"] = kwargsTr["master_net"]

    lossFunc = Loss(args,reduction="mean")

    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc)

    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc

    if not args.only_test:

        actual_bs = args.batch_size if args.batch_size < args.max_batch_size_single_pass else args.max_batch_size_single_pass
        args.batch_per_epoch = len(trainLoader.dataset)//actual_bs if len(trainLoader.dataset) > actual_bs else 1

        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            trainFunc(**kwargsTr)
            if not scheduler is None:
                scheduler.step()

            with torch.no_grad():
                metricVal = valFunc(**kwargsVal)
            
            bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                        args.model_id, bestEpoch, epoch, net,
                                                                        isBetter, worseEpochNb)
            if trial is not None:
                trial.report(metricVal, epoch)

            epoch += 1

    if trial is None:
        if args.run_test or args.only_test:

            if os.path.exists("../results/{}/test_done.txt".format(args.exp_id)):
                test_done = np.genfromtxt("../results/{}/test_done.txt".format(args.exp_id),delimiter=",",dtype=str)

                if len(test_done.shape) == 1:
                    test_done = test_done[np.newaxis]
            else:
                test_done = None

            alreadyDone = (test_done==np.array([args.model_id,str(bestEpoch)])).any()

            if (test_done is None) or (alreadyDone and args.do_test_again) or (not alreadyDone):

                testFunc = valFunc

                kwargsTest = kwargsVal
                kwargsTest["mode"] = "test"

                testLoader,_ = load_data.buildTestLoader(args, "test")

                kwargsTest['loader'] = testLoader

                net = preprocessAndLoadParams("../models/{}/model{}_best_epoch{}".format(args.exp_id, args.model_id, bestEpoch),args.cuda,net)

                kwargsTest["model"] = net
                kwargsTest["epoch"] = bestEpoch

                with torch.no_grad():
                    testFunc(**kwargsTest)

                with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                    print("{},{}".format(args.model_id,bestEpoch),file=text_file)

    else:

        oldPath = "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, bestEpoch)
        os.rename(oldPath, oldPath.replace("best_epoch","trial{}_best_epoch".format(trial.number)))

        with open("../results/{}/{}_{}_valRet.csv".format(args.exp_id,args.model_id,trial.number),"w") as text:
            print(metricVal,file=text)

        return metricVal

def computeSpars(data_shape,attMaps,args,resDic):
    if args.att_metrics_post_hoc:
        features = None 
    else:
        features = resDic["features"]
        if "attMaps" in resDic:
            attMaps = resDic["attMaps"]
        else:
            attMaps = torch.ones(data_shape[0],1,features.size(2),features.size(3)).to(features.device)

    sparsity = metrics.compAttMapSparsity(attMaps,features)
    sparsity = sparsity/data_shape[0]
    return sparsity 
   
def loadAttMaps(exp_id,model_id):

    paths = glob.glob("../results/{}/attMaps_{}_epoch*.npy".format(exp_id,model_id))

    if len(paths) >1 or len(paths) == 0:
        raise ValueError("Wrong path number for exp {} model {}".format(exp_id,model_id))

    attMaps,norm = np.load(paths[0],mmap_mode="r"),np.load(paths[0].replace("attMaps","norm"),mmap_mode="r")

    return torch.tensor(attMaps),torch.tensor(norm)

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')

    argreader.parser.add_argument('--attention_metrics', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
    
    argreader.parser.add_argument('--att_metrics_few_steps', type=str2bool, help='To do as much step for high res than for low res')

    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    # The folders where the experience file will be written
    if not os.path.exists("../vis/{}".format(args.exp_id)):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not os.path.exists("../results/{}".format(args.exp_id)):
        os.makedirs("../results/{}".format(args.exp_id))
    if not os.path.exists("../models/{}".format(args.exp_id)):
        os.makedirs("../models/{}".format(args.exp_id))

    # Update the config args
    argreader.args = args
    # Write the arguments in a config file so the experiment can be re-run

    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id, args.model_id))
    print("Model :", args.model_id, "Experience :", args.exp_id)

    if args.optuna:
        def objective(trial):
            return run(args,trial=trial)

        study = optuna.create_study(direction="maximize" if args.maximise_val_metric else "minimize",\
                                    storage="sqlite:///../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id), \
                                    study_name=args.model_id,load_if_exists=True)

        con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
        curr = con.cursor()

  
        print("N trials",args.optuna_trial_nb)
        study.optimize(objective,n_trials=args.optuna_trial_nb)

        curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
        query_res = curr.fetchall()

        query_res = list(filter(lambda x:not x[1] is None,query_res))

        trialIds = [id_value[0] for id_value in query_res]
        values = [id_value[1] for id_value in query_res]

        trialIds = trialIds[:args.optuna_trial_nb]
        values = values[:args.optuna_trial_nb]

        bestTrialId = trialIds[np.array(values).argmax()]

        curr.execute('SELECT param_name,param_value from trial_params WHERE trial_id == {}'.format(bestTrialId))
        query_res = curr.fetchall()

        args.only_test = True

        print("bestTrialId-1",bestTrialId-1)
        bestPath = glob.glob("../models/{}/model{}_trial{}_best_epoch*".format(args.exp_id,args.model_id,bestTrialId-1))[0]
        print(bestPath)

        copyfile(bestPath, bestPath.replace("_trial{}".format(bestTrialId-1),""))

        args.distributed=False

        train(args,None)

    elif args.attention_metrics:

        path_suff = args.attention_metrics
        
        args.val_batch_size = 1
        _,testDataset = load_data.buildTestLoader(args, "test")

        bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
        _ = int(os.path.basename(bestPath).split("epoch")[1])

        net = modelBuilder.netBuilder(args)
        net = preprocessAndLoadParams(bestPath,args.cuda,net)
        net.eval()
        
        attMaps_dataset,norm_dataset = loadAttMaps(args.exp_id,args.model_id)

        if not args.attention or (args.attention and args.br_npa):
            attrFunc = lambda i:(attMaps_dataset[i,0:1]*norm_dataset[i]).unsqueeze(0)
        else:
            attrFunc = lambda i:(attMaps_dataset[i].float().mean(dim=0,keepdim=True).byte()*norm_dataset[i]).unsqueeze(0)
    
        torch.set_grad_enabled(False)

        nbImgs = args.att_metrics_img_nb
        print("nbImgs",nbImgs)

        allScoreList = []
        allPreds = []
        allTarg = []

        torch.manual_seed(0)

        if args.debug:
            inds = torch.randint(100,size=(nbImgs,))
        else:
            inds = torch.randint(len(testDataset),size=(nbImgs,))

        blurKernel = torch.ones(121,121)
        blurKernel = blurKernel/blurKernel.numel()
        blurKernel = blurKernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
        blurKernel = blurKernel.cuda() if args.cuda else blurKernel

        for imgInd,i in enumerate(inds):
            if imgInd % 20 == 0 :
                print("Img",i.item(),"(",imgInd,"/",len(inds),")")

            batch = testDataset.__getitem__(i)
            data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)

            data = data.cuda() if args.cuda else data
            targ = targ.cuda() if args.cuda else targ

            allData = data.clone().cpu()
            
            resDic = net(data)
            scores = torch.softmax(resDic["pred"],dim=-1)

            if args.attention_metrics in ["Add","Del"]:
                predClassInd = scores.argmax(dim=-1)
                allPreds.append(predClassInd.item())
                allTarg.append(targ.item())
            else:
                raise ValueError("Unkown metrics",args.attention_metrics)

            if args.attention_metrics=="Add":
                origData = data.clone()
                data = F.conv2d(data,blurKernel,padding=blurKernel.size(-1)//2,groups=blurKernel.size(0))

            attMaps = attrFunc(i)

            attMaps = (attMaps-attMaps.min())/(attMaps.max()-attMaps.min())

            allAttMaps = attMaps.clone().cpu()
            statsList = []

            totalPxlNb = attMaps.size(2)*attMaps.size(3)
            leftPxlNb = totalPxlNb

            if args.att_metrics_few_steps:
                stepNb = 196 
            else:
                stepNb = totalPxlNb

            score_prop_list = []

            ratio = data.size(-1)//attMaps.size(-1)

            stepCount = 0
            while leftPxlNb > 0:

                attMin,attMean,attMax = attMaps.min().item(),attMaps.mean().item(),attMaps.max().item()
                statsList.append((attMin,attMean,attMax))

                _,ind_max = (attMaps)[0,0].view(-1).topk(k=totalPxlNb//stepNb)
                ind_max = ind_max[:leftPxlNb]

                x_max,y_max = ind_max % attMaps.shape[3],ind_max // attMaps.shape[3]
                
                ratio = data.size(-1)//attMaps.size(-1)

                for i in range(len(x_max)):
                    
                    if args.attention_metrics=="Add":
                        data[0,:,y_max[i]*ratio:y_max[i]*ratio+ratio,x_max[i]*ratio:x_max[i]*ratio+ratio] = origData[0,:,y_max[i]*ratio:y_max[i]*ratio+ratio,x_max[i]*ratio:x_max[i]*ratio+ratio]
                    elif args.attention_metrics=="Del":
                        data[0,:,y_max[i]*ratio:y_max[i]*ratio+ratio,x_max[i]*ratio:x_max[i]*ratio+ratio] = 0
                    else:
                        raise ValueError("Unkown attention metric",args.attention_metrics)

                    attMaps[0,:,y_max[i],x_max[i]] = -1                       

                leftPxlNb -= totalPxlNb//stepNb
                if stepCount % 30 == 0:
                    allAttMaps = torch.cat((allAttMaps,torch.clamp(attMaps,0,attMaps.max().item()).cpu()),dim=0)
                    allData = torch.cat((allData,data.cpu()),dim=0)
                stepCount += 1

                resDic = net(data)
                score = torch.softmax(resDic["pred"],dim=-1)[:,predClassInd[0]]

                score_prop_list.append((leftPxlNb,score.item()))

            allScoreList.append(score_prop_list)

        np.save("../results/{}/attMetr{}_{}.npy".format(args.exp_id,path_suff,args.model_id),np.array(allScoreList,dtype=object))

    else:
        train(args,None)

if __name__ == "__main__":
    main()
