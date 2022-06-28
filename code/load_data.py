import numpy as np
import torch
import args
from random import Random
import fineGrainedDataset
from torch.utils.data.sampler import Sampler

def buildTrainLoader(args,shuffle=True):

    if args.big_images:
        imgSize = 448
    else:
        imgSize = 224

    train_dataset = fineGrainedDataset.FineGrainedDataset(args.dataset_train, "train",(imgSize,imgSize),\
                                            sqResizing=args.sq_resizing,\
                                            cropRatio=args.crop_ratio,brightness=args.brightness,\
                                            saturation=args.saturation)

    totalLength = len(train_dataset)

    if args.prop_set_int_fmt:
        train_prop = args.train_prop / 100
    else:
        train_prop = args.train_prop

    np.random.seed(1)
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    train_dataset, _ = torch.utils.data.random_split(train_dataset, [int(totalLength * train_prop),
                                                                     totalLength - int(totalLength * train_prop)])

    bsz = args.batch_size
    bsz = bsz if bsz < args.max_batch_size_single_pass else args.max_batch_size_single_pass

    kwargs = {"shuffle": shuffle}

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz,
                                              pin_memory=True, num_workers=args.num_workers,drop_last=args.drop_last, **kwargs)

    return trainLoader, train_dataset


def buildTestLoader(args, mode):
    datasetName = getattr(args, "dataset_{}".format(mode))
    imgSize = 448 if args.big_images else 224
    test_dataset = fineGrainedDataset.FineGrainedDataset(datasetName, mode,(imgSize,imgSize),\
                                                        sqResizing=args.sq_resizing,\
                                                        cropRatio=args.crop_ratio,brightness=args.brightness,saturation=args.saturation)

    if mode == "val" and args.dataset_train == args.dataset_val:
        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)

        if args.prop_set_int_fmt:
            train_prop = args.train_prop / 100
        else:
            train_prop = args.train_prop

        totalLength = len(test_dataset)
        _, test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength * train_prop),
                                                                       totalLength - int(totalLength * train_prop)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    sampler=None

    if args.val_batch_size == -1:
        args.val_batch_size = int(args.max_batch_size*3.5)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,
                                             num_workers=args.num_workers,sampler=sampler,pin_memory=True)

    return testLoader,test_dataset

def addArgs(argreader):
    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                                  help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for validation')
    argreader.parser.add_argument('--max_batch_size_single_pass', type=int, metavar='BS',
                                  help='The maximum sub batch size when using very big images.')

    argreader.parser.add_argument('--train_prop', type=float, metavar='END',
                                  help='The proportion of the train dataset to use for training when working in non video mode. The rest will be used for validation.')

    argreader.parser.add_argument('--prop_set_int_fmt', type=args.str2bool, metavar='BOOL',
                                  help='Set to True to set the sets (train, validation and test) proportions\
                            using int between 0 and 100 instead of float between 0 and 1.')

    argreader.parser.add_argument('--dataset_train', type=str, metavar='DATASET',
                                  help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='DATASET',
                                  help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='DATASET',
                                  help='The dataset for testing. Can be "big" or "small"')

    argreader.parser.add_argument('--class_nb', type=int, metavar='S',
                                  help='The number of class of to model')

    argreader.parser.add_argument('--big_images', type=args.str2bool, metavar='S',
                                  help='To resize the images to 448 pixels instead of 224')

    argreader.parser.add_argument('--drop_last', type=args.str2bool, metavar='S',
                                    help='To drop the last batch in case it does not fit the batch size.')

    argreader.parser.add_argument('--normalize_data', type=args.str2bool, metavar='S',
                                  help='To normalize the data using imagenet means and std before puting it between 0 and 1.')

    argreader.parser.add_argument('--sq_resizing', type=args.str2bool, metavar='S',
                                  help='To resize each input image to a square.')

    argreader.parser.add_argument('--crop_ratio', type=float, metavar='S',
                                  help='The crop ratio (usually 0.875) for data augmentation.')
    argreader.parser.add_argument('--brightness', type=float, metavar='S',
                                  help='The brightness intensity for data augmentation.')
    argreader.parser.add_argument('--saturation', type=float, metavar='S',
                                  help='The saturation intensity for data augmentation.')

    return argreader
