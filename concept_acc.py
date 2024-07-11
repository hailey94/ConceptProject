import cv2
import os
import numpy as np
import json

import torch
import torch.nn as nn
from PIL import Image
from configs import parser
from tqdm import tqdm

from loaders.get_loader import get_transformations_synthetic, load_all_imgs
from utils.engine_mine import train, test
from model.partdetection.model_main import MMDiscovery
from loaders.get_loader import loader_generation
from loaders.ImageNet import get_name
from utils.tools import apply_colormap_on_image, vis, save_maps_for_single

from sklearn.metrics import accuracy_score, hamming_loss


def main():
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

    if args.dataset != 'matplot':
        exit()

    transform = get_transformations_synthetic()
    _, _, data_, labels_val, cat = load_all_imgs(args)

    total_label = []
    total_pred = []

    for i in tqdm(range(len(data_))):
        root = data_[i]
        img_orl = Image.open(root).convert('RGB')
        name_label = root.split("/")[-1].split(".")[0]

        _, pred, maps, _, _ = model(transform(img_orl).unsqueeze(0).to(device), train=False, mask=False)

        if len(maps.size()) == 3:
            b, c, wh = maps.shape
            w = int(wh ** 0.5)
            maps = maps.view(b, c, w, -1)

        save_pth = f'/shared/data/vis/Proposed/{args.name}/{args.dataset}/{name_label}'
        os.makedirs(save_pth, exist_ok=True)
        vis(maps, f'{save_pth}/')

        sample_root = "/shared/data/matplob/label/" + name_label
        cpt_sample = get_name(sample_root, mode_folder=False)

        if cpt_sample is None:
            continue

        # label 갯수
        sample_label = np.zeros(5)
        for k in range(len(cpt_sample)):
            sample_index = int(cpt_sample[k].split(".")[0])
            sample_label[sample_index] = 1

        total_label.append(sample_label)

        sample_pred = np.zeros(5)
        for j in range(cpt_num):  # concept number j 번에서
            mask_current = np.array(Image.open(f"{save_pth}/0_slot_" + str(j) + ".png"))
            MAX = np.max(mask_current)
            MIN = np.min(mask_current)

            mask_current = (mask_current - MIN) / (MAX - MIN)
            upper = mask_current > thresh_att
            lower = mask_current <= thresh_att
            mask_current[upper] = 1
            mask_current[lower] = 0

            for s in range(len(cpt_sample)):  # concept s가 activate 된다
                sample_index = int(cpt_sample[s].split(".")[0])

                current_sample = cv2.imread(sample_root + "/" + cpt_sample[s], 0)
                current_sample[current_sample != 255] = 1
                current_sample[current_sample == 255] = 0

                overlap = mask_current + current_sample
                overlap_sum = (overlap == 2).sum()
                union_sum = current_sample.sum()

                if overlap_sum / union_sum > thresh_overlap:
                    sample_pred[sample_index] = 1

        total_pred.append(sample_pred)

    total_pred = np.array(total_pred).squeeze()
    total_label = np.array(total_label).squeeze()
    np.save('concept_prediction.npy', total_pred)
    np.save('concept_labels.npy', total_label)

    a = accuracy_score(total_label, total_pred) # Exact match
    h = hamming_loss(total_label, total_pred) # hamming loss
    print(a, 1.0 - h)



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


