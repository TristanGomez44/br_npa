""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Modified: Dec 18,2021 - Tristan Gomez
"""
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class FineGrainedDataset(Dataset):
    def __init__(self, root, phase,resize,sqResizing,\
                        cropRatio,brightness,saturation):

        self.image_path = {}
        self.imageSeg_path = None
        self.image_label = {}
        self.root = "../data/"+root
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200

        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]

        classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        instances = []
        id = 0
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        self.image_path[id] = path
                        self.image_label[id] = class_index
                        self.image_id.append(id)

                        id += 1

        # transform
        self.transform = get_transform(self.resize, self.phase,\
                                        sqResizing=sqResizing,cropRatio=cropRatio,brightness=brightness,\
                                        saturation=saturation)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]
        image = Image.open(self.image_path[image_id]).convert('RGB')  # (C, H, W)

        # image
        image = self.transform(image)

        return image, self.image_label[image_id]

    def __len__(self):
        return len(self.image_id)

def is_valid_file(x):
    return has_file_allowed_extension(x, IMG_EXTENSIONS)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def get_transform(resize, phase='train',sqResizing=True,cropRatio=0.875,brightness=0.126,saturation=0.5):

    if sqResizing:
        kwargs={"size":(int(resize[0] / cropRatio), int(resize[1] / cropRatio))}
    else:
        kwargs={"size":int(resize[0] / cropRatio)}

    if phase == 'train':
        transf = [transforms.Resize(**kwargs),
                    transforms.RandomCrop(resize),
                    transforms.RandomHorizontalFlip(0.5)]
        transf.extend([transforms.ColorJitter(brightness=brightness, saturation=saturation)])
    else:
        transf = [transforms.Resize(**kwargs),transforms.CenterCrop(resize)]

    transf.extend([transforms.ToTensor()])
    transf.extend([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transf = transforms.Compose(transf)

    return transf
