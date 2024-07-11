import matplotlib.pyplot as plt
import torch
import torch.nn as nn


import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from configs import parser
from model.partdetection.model_main import MMDiscovery
from loaders.ImageNet import get_name
from loaders.get_loader import load_all_imgs, get_transform
from utils.tools import apply_colormap_on_image, vis, save_maps_for_single
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.size'] = 10


def main():
    os.makedirs(f'/shared/data/vis/Proposed2/{args.name}/', exist_ok=True)
    # load model and weights
    model = MMDiscovery(args)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    ckp = os.path.join(args.output_dir,
                                         f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
                                         f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}_{args.name}.pt")

    print('\nLoad your model from ... \n{}'.format(ckp))
    checkpoint = torch.load(ckp, map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()

    transform = get_transform(args)["val"]
    _, _, data_, labels_val, cat = load_all_imgs(args)

    del _

    label_cnt = 0
    label_i = 0


    for i in tqdm(range(len(data_))):

        root, label = data_[i], labels_val[i]
        if 'American_Crow_0027_25146' not in root:
            continue

        label_cnt += 1
        # if i < 3700:
        #     continue
        #
        # if (label_cnt > 10) and label == label_i:
        #     continue
        #
        # if (label_cnt > 10) and label != label_i:
        #     label_cnt = 1
        #     label_i = label

        name_label = root.split("/")[-1].split(".")[0]
        if "American_Crow" not in name_label:
            continue

        save_pth = f'/shared/data/vis/Proposed2/{args.name}/{args.dataset}/{name_label}'
        os.makedirs(save_pth, exist_ok=True)

        if args.dataset == 'matplot':
            sample_root = "/shared/data/matplob/label/" + name_label
            cpt_sample = get_name(sample_root, mode_folder=False)
            if cpt_sample is None:
                # print("not exist folder " + sample_root)
                continue

        if args.dataset == "MNIST":
            img_orl = Image.open(root).resize([224, 224], resample=Image.BILINEAR)
            img_orl2 = Image.open(root).resize([224, 224], resample=Image.BILINEAR)
        elif args.dataset == "cifar10":
            img_orl = Image.open(root).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
            img_orl2 = Image.open(root).convert('RGB').resize([224, 224], resample=Image.BILINEAR)
        else:
            img_orl = Image.open(root)
            img_orl2 = Image.open(root).convert('RGB').resize([224, 224], resample=Image.BILINEAR)

        if np.array(img_orl).shape[-1] != 3:
            label_cnt -= 1
            continue

        img_orl2.save(f'{save_pth}/origin.png')

        cpt, probs, maps, update, kl = model(transform(img_orl).unsqueeze(0).to(device), train=False)

        b, c, wh = maps.shape
        w = int(wh ** 0.5)
        maps = maps.reshape([b, c, w, -1])
        maps = maps.detach().cpu().numpy()
        np.save('mine_actmap', maps)
        exit()


        # mask_exist = 1 if len(os.listdir(f'{save_pth}/')) > 1 else 0
        # if not mask_exist:
        #     vis(maps, f'{save_pth}/')
        vis(maps, f'{save_pth}/')

        for id in range(args.num_cpt):
            slot_image = cv2.imread(f'{save_pth}/0_slot_{id}.png',
                                    cv2.IMREAD_GRAYSCALE)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
            heatmap_on_image.save(
                f'{save_pth}/0_slot_mask_{id}_{args.name}.png')

        save_maps_for_single(img_orl2, maps,
                             f'{save_pth}/{args.name}_', device,
                             True)


if __name__ == '__main__':
    parser.add_argument('--test', default=1, type=int, help='evaluate on test (1) or train (0)')
    parser.add_argument('--thresh_att', default=0.2, type=float, help='threshold for attention values')
    parser.add_argument('--thresh_overlap', default=0.9, type=float, help='threshold for overlap')

    args = parser.parse_args()
    args.pre_train = False
    if args.test:
        dset = 'test'
    else:
        dset = 'train'
    args.batch_size = 1
    thresh_att = args.thresh_att
    thresh_overlap = args.thresh_overlap
    cpt_num = args.num_cpt
    main()


