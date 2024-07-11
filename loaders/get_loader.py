from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from loaders.CUB200 import CUB_200
from loaders.ImageNet import ImageNet
from loaders.AwA import AwADataset
from loaders.matplob import Matplot, MakeImage
import numpy as np
from PIL import Image
import torch
import os
from torchvision.models import ResNet18_Weights
# from torchvision.datasets import ImageNet

class FeatureDataset(Dataset):
    def __init__(self, features, targets, group_array=None):
        self.features = torch.tensor(features)
        self.targets = torch.tensor(targets)
        self.group_array = group_array

    def __getitem__(self, idx):
        if self.group_array is not None:
            return self.features[idx], self.targets[idx], self.group_array[idx]
        return self.features[idx], self.targets[idx]

    def __len__(self):
        return len(self.features)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_train_transformations(args, norm_value=None):
    if args.dataset == 'awa':
        aug_list = [
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.Resize((224, 224), Image.BILINEAR),  # ImageNet standard
            transforms.ToTensor()
        ]
    else:
        aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]


    return transforms.Compose(aug_list)



def get_val_transformations(args, norm_value=None):
    if args.dataset == 'awa':
        aug_list = [
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor()
        ]
    else:
        aug_list = [
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(norm_value[0], norm_value[1])
                ]


    return transforms.Compose(aug_list)


def get_transformations_synthetic():
    aug_list = [
                transforms.Resize((224, 224), Image.BILINEAR),
                transforms.ToTensor(),
                ]
    return transforms.Compose(aug_list)


def get_transform(args):
    if args.dataset == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        return {"train": transform, "val": transform}
    elif args.dataset == "cifar10":
        transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(),
                                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        return {"train": transform, "val": transform}
    elif args.dataset == "cifar100":
        transform = transforms.Compose([transforms.Resize([args.img_size, args.img_size]), transforms.ToTensor(),
                                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        return {"train": transform, "val": transform}
    elif args.dataset == "CUB200":
        transform_train = get_train_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform_val = get_val_transformations(args, [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        return {"train": transform_train, "val": transform_val}

    elif args.dataset == "ImageNet" or args.dataset == "imagenet":
        if args.base_model == 'resnet18':
            transform_train = ResNet18_Weights.IMAGENET1K_V1.transforms()
            transform_val = ResNet18_Weights.IMAGENET1K_V1.transforms()
            return {"train": transform_train, "val": transform_val}
        else:
            print('we do not have backbone except resnet18 now')
            exit()
    elif args.dataset == "matplot":
        transform_train = get_transformations_synthetic()
        transform_val = get_transformations_synthetic()
        return {"train": transform_train, "val": transform_val}

    elif args.dataset == "imagenet-sep":
        return {"train": None, "val": None}

    elif args.dataset == "awa":
        transform_train = get_train_transformations(args)
        transform_val = get_val_transformations(args)
        return {"train": transform_train, "val": transform_val}

    raise ValueError(f'unknown {args.dataset}')


def select_dataset(args, transform):
    if args.dataset == "MNIST":
        dataset_train = datasets.MNIST('/shared/data/', train=True, download=False, transform=transform["train"])
        dataset_val = datasets.MNIST('/shared/data/', train=False, transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "cifar10":
        dataset_train = datasets.CIFAR10('/shared/data/', train=True, download=False, transform=transform["train"])
        dataset_val = datasets.CIFAR10('/shared/data/', train=False, transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "cifar100":
        dataset_train = datasets.CIFAR100('/shared/data/', train=True, download=False, transform=transform["train"])
        dataset_val = datasets.CIFAR100('/shared/data/', train=False, transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "CUB200":
        dataset_train = CUB_200(args, train=True, transform=transform["train"])
        dataset_val = CUB_200(args, train=False, transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "ImageNet" or args.dataset == "imagenet":
        dataset_train = ImageNet(args, "train", transform=transform["train"])
        dataset_val = ImageNet(args, "val", transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "matplot":
        data_ = MakeImage().get_img()
        dataset_train = Matplot(data_, "train", transform=transform["train"])
        dataset_val = Matplot(data_, "val", transform=transform["val"])
        return dataset_train, dataset_val
    elif args.dataset == "imagenet-sep":
        train_features, train_labels = None, None
        test_features, test_labels = None, None
        dataset_train = FeatureDataset(train_features, train_labels)
        dataset_val = FeatureDataset(test_features, test_labels)
        return dataset_train, dataset_val
    elif args.dataset == "awa":
        dataset_train = AwADataset(args, train=True, transform=transform["train"])
        dataset_val = AwADataset(args, train=False, transform=transform["val"])
        return dataset_train, dataset_val

    raise ValueError(f'unknown {args.dataset}')


def loader_generation(args):
    transform = get_transform(args)
    train_set, val_set = select_dataset(args, transform)
    print('Train samples %d - Val samples %d' % (len(train_set), len(val_set)))

    train_loader1 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=True)
    train_loader2 = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=False, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=False, drop_last=False)
    return train_loader1, train_loader2, val_loader


def load_all_imgs(args):
    def filter(data):
        imgs = []
        labels = []
        for i in range(len(data)):
            root = data[i][0]
            if args.dataset == "matplot":
                ll = data[i][1]
            else:
                ll = int(data[i][1])
            if args.dataset == "CUB200":
                ll -= 1
                root = os.path.join(os.path.join(args.dataset_dir, "CUB_200_2011"), 'images', root)
            if args.dataset == "awa":
                ll -= 1
            imgs.append(root)
            labels.append(ll)
        return imgs, labels

    if args.dataset == "MNIST":
        train_imgs = datasets.MNIST('/shared/data/', train=True, download=False, transform=None).data
        train_labels = datasets.MNIST('/shared/data/', train=True, download=False, transform=None).targets
        val_imgs = datasets.MNIST('/shared/data/', train=False, download=False, transform=None).data
        val_labels = datasets.MNIST('/shared/data/', train=False, download=False, transform=None).targets
        return train_imgs, train_labels, val_imgs, val_labels, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif args.dataset == "cifar10":
        cat = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        train_imgs = datasets.CIFAR10('/shared/data/', train=True, download=False, transform=None).data
        train_labels = datasets.CIFAR10('/shared/data/', train=True, download=False, transform=None).targets
        val_imgs = datasets.CIFAR10('/shared/data/', train=False, download=False, transform=None).data
        val_labels = datasets.CIFAR10('/shared/data/', train=False, download=False, transform=None).targets
        return train_imgs, train_labels, val_imgs, val_labels, cat
    elif args.dataset == "ImageNet" or args.dataset == "imagenet":
        train = ImageNet(args, "train", transform=None).train
        val = ImageNet(args, "train", transform=None).val
        cat = ImageNet(args, "train", transform=None).category
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat

    elif args.dataset == "Custom":
        train = ImageNet(args, "train", transform=None).train
        val = ImageNet(args, "train", transform=None).val
        cat = ImageNet(args, "train", transform=None).category
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat

    elif args.dataset == "CUB200":
        train = CUB_200(args)._train_path_label
        val = CUB_200(args)._test_path_label
        cat = CUB_200(args)._cls_name_dict
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat

    elif args.dataset == "matplot":
        data_ = MakeImage().get_img()
        train = data_[0]
        val = data_[1]
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        cat = {0:0, 1:1, 2:2, 3:3, 4:4}
        return train_imgs, train_labels, val_imgs, val_labels, cat

    elif args.dataset == "awa":
        train = AwADataset(args)._train_path_label
        val = AwADataset(args)._test_path_label
        cat = AwADataset(args)._cls_name_dict
        train_imgs, train_labels = filter(train)
        val_imgs, val_labels = filter(val)
        return train_imgs, train_labels, val_imgs, val_labels, cat


def loader_sparsity(args, transform):
    val_set = ImageNet(args, "val", transform=transform)
    print('Val samples %d' % (len(val_set)))

    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=16,
                           pin_memory=False, drop_last=False)
    return val_loader