import cv2
import os
import numpy as np
import json

import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from configs import parser
from tqdm import tqdm


from loaders.get_loader import get_transformations_synthetic, load_all_imgs
from utils.engine_mine import test
from model.partdetection.model_main import MMDiscovery
from loaders.get_loader import loader_generation
from loaders.ImageNet import get_name
from utils.tools import apply_colormap_on_image, vis, save_maps_for_single

import concepts_xai
from concepts_xai.evaluation.metrics.oracle import oracle_impurity_score
from concepts_xai.evaluation.metrics.niching import niche_impurity_score

plt.rcParams['font.size'] = 10


def make_statistic(cpt_nums):
    record = []
    for i in range(cpt_nums):
        record.append({0: 0, 1: 0, 2: 0, 3: 0, 4: 0})
    return record



def find_neuron_concept_relation(cpt_num, data_, model, transform, device):
    '''
    statistic = [{0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}, ... , {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}]
    statistic[i]   => attnetion i 에서 어느 concept 이 몇 번 activate 된건지 나타내는 dictionary

    statistic_sample = ground truth concept 의 total 등장 횟수
    thresh_att = attention map threshold
    thresh_overlap = threshold betweent gt  and gt & concept area
    '''
    statistic = make_statistic(cpt_num)

    statistic_sample = [0, 0, 0, 0, 0]

    for i in tqdm(range(len(data_))):
        root = data_[i]
        img_orl = Image.open(root).convert('RGB')
        name_label = root.split("/")[-1].split(".")[0]

        _, pred, maps, cpt, kl = model(transform(img_orl).unsqueeze(0).to(device), train=False, mask=True)
        # _, pred, maps, cpt = model(transform(img_orl).unsqueeze(0).to(device))

        if len(maps.size()) == 3:
            b, c, wh = maps.shape
            w = int(wh**0.5)
            maps = maps.view(b, c, w, -1)

        save_pth = f'/shared/data/vis/Proposed/{args.name}/{args.dataset}/{name_label}'
        os.makedirs(save_pth, exist_ok=True)
        vis(maps, f'{save_pth}/')

        sample_root = "/shared/data/matplob/label/" + name_label
        cpt_sample = get_name(sample_root, mode_folder=False)

        if cpt_sample is None:
            # print("not exist folder " + sample_root)
            continue

        # label 갯수
        for k in range(len(cpt_sample)):
            sample_indexs = int(cpt_sample[k].split(".")[0])
            statistic_sample[sample_indexs] += 1

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
                    statistic[j][sample_index] += 1

    s = [[] for i in range(len(statistic_sample))] # 각 concept 의 15 개 slot 에 대한 activation 정도

    for l in range(len(statistic)):
        s[0].append(statistic[l][0])
        s[1].append(statistic[l][1])
        s[2].append(statistic[l][2])
        s[3].append(statistic[l][3])
        s[4].append(statistic[l][4])

    coverage = np.zeros(len(statistic_sample))
    purity = np.zeros(len(statistic_sample))

    # A[i] = i-th shape 은 A[i] 번째 slot 에서 가장 많이 activation 됨
    A = [int(np.argmax(np.array(s[i]))) for i in range(len(statistic_sample))]
    for s_j, num_sj in enumerate(statistic_sample):
        c = A[s_j] # shape s_j는 c 번째 slot 에서 가장 많이 activation 됨
        num_cov = statistic[c][s_j] # c 번째 slot 에서 s_j가 몇 번 activate 되었는지
        coverage[s_j] = num_cov / num_sj # 실제 s_j의 등장 수 대비 s_j가 몇 번 activate 되었는지
        purity[s_j] = statistic[c][s_j] / sum(statistic[c].values()) # c 번째 slot 전체 activation 중 s_j의 비율

    return A, statistic_sample, statistic, coverage, purity, s

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

    # _, _, val_loader = loader_generation(args)
    # acc = test(args, model, val_loader, device)
    # print('ACC. :{:.3f}'.format(100*acc))

    if args.dataset != 'matplot':
        exit()
    del  val_loader

    transform = get_transformations_synthetic()
    _, _, data_, labels_val, cat = load_all_imgs(args)

    A, statistic_sample, statistic, coverage, purity, s = find_neuron_concept_relation(args.num_cpt, data_, model, transform, device)
    # A[i] = i-th shape 은 A[i] 번째 slot 에서 가장 많이 activation 됨
    #
    ########### draw coverage figure
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_width = 0.18

    # concept이  5개이므로 0, 1, 2, 3, 4 위치를 기준으로 삼음
    index = np.arange(15)
    c = [f'Attn{i}' for i in range(args.num_cpt)]

    # count 를 coverage 로 치환
    s[0] = list(np.array(s[0])/statistic_sample[0])
    s[1] = list(np.array(s[1])/statistic_sample[1])
    s[2] = list(np.array(s[2])/statistic_sample[2])
    s[3] = list(np.array(s[3])/statistic_sample[3])
    s[4] = list(np.array(s[4])/statistic_sample[4])

    # 각 연도별로 3개 샵의 bar를 순서대로 나타내는 과정, 각 그래프는 0.25의 간격을 두고 그려짐
    b1 = plt.bar(index, s[0], bar_width, alpha=0.8, color='red', label='S.1')
    b2 = plt.bar(index + bar_width, s[1], bar_width, alpha=0.4, color='black', label='S.2')
    b3 = plt.bar(index + 2 * bar_width, s[2], bar_width, alpha=0.8, color='blue', label='S.3')
    b4 = plt.bar(index + 3 * bar_width, s[3], bar_width, alpha=0.8, color='purple', label='S.4')
    b5 = plt.bar(index + 4 * bar_width, s[4], bar_width, alpha=0.8, color='green', label='S.5')

    # 축 위치를 정 가운데로 조정하고 x축의 텍스트를 concpet 정보와 매칭
    plt.xticks(np.arange(2*bar_width, 15 + 2*bar_width, 1), c)
    plt.ylim([0, 1])
    # x축, y축 이름 및 범례 설정
    plt.xlabel('Concepts')
    plt.ylabel('Coverage')
    plt.legend(loc="upper left", bbox_to_anchor=(1, 0.95))
    plt.tight_layout()
    plt.savefig(f'./eval/{dset}_beta={thresh_att}_gamma={thresh_overlap}_{args.name}.png')


    cpts = []
    labels = []
    maps_avg = []

    cpts_all = []
    maps_all = []

    for i in tqdm(range(len(data_))):
        root = data_[i]
        img_orl = Image.open(root).convert('RGB')
        name_label = root.split("/")[-1].split(".")[0]

        cpt, probs, maps, update, kl = model(transform(img_orl).unsqueeze(0).to(device), train=False, mask=True)
        # cpt, probs, maps, update = model(transform(img_orl).unsqueeze(0).to(device))

        if len(maps.size()) == 4:
            maps = maps.view([1, args.num_cpt, -1])

        if len(cpt.size()) == 3:
            cpt = torch.sum(cpt, dim=[1]).detach().cpu().numpy()  # B C
        else:
            cpt = cpt.detach().cpu().numpy()

        # maps = maps.detach().cpu().numpy()  # B C
        maps = torch.sum(maps, dim=-1).detach().cpu().numpy()  # B C

        cpt_pred = np.zeros([1, 5])
        maps_pred = np.zeros([1, 5])
        for i in range(5):
            cpt_pred[:, i] = cpt[:, A[i]]
            maps_pred[:, i] = maps[:, A[i]]

        sample_root = "/shared/data/matplob/label/" + name_label
        cpt_sample = get_name(sample_root, mode_folder=False)

        if cpt_sample is None:
            continue

        concept_label = np.zeros(5)
        label_index = [int(i.split('.')[0]) for i in cpt_sample]
        concept_label[label_index] = 1
        labels.append(concept_label)
        cpts.append(cpt_pred)
        maps_avg.append(maps_pred)


    cpts = np.array(cpts).squeeze()
    maps_avg = np.array(maps_avg).squeeze()

    labels = np.array(labels)

    ois = oracle_impurity_score(
        c_soft=cpts,  # Size (n_samples, concept_dim, n_concepts) or (n_samples, n_concepts)
        c_true=labels,  # Size (n_samples, n_concepts)
    )

    nis = niche_impurity_score(
        c_soft=cpts,  # Size (n_samples, concept_dim, n_concepts) or (n_samples, n_concepts)
        c_true=labels,  # Size (n_samples, n_concepts)
    )

    ois_map = oracle_impurity_score(
        c_soft=maps_avg,  # Size (n_samples, concept_dim, n_concepts) or (n_samples, n_concepts)
        c_true=labels,  # Size (n_samples, n_concepts)
    )

    nis_map = niche_impurity_score(
        c_soft=maps_avg,  # Size (n_samples, concept_dim, n_concepts) or (n_samples, n_concepts)
        c_true=labels,  # Size (n_samples, n_concepts)
    )


    print('----Concepts of Interests cpts')
    print('completeness:', np.average(coverage))
    print('purity:', np.average(purity))
    print('founded concept ratio:', len(set(A)) / len(A))
    print('ois:', ois)
    print('nis:', nis)

    print('----Concepts of Interests from MAPS')
    print('ois:', ois_map)
    print('nis:', nis_map)

    coverage = list(coverage)
    coverage = list([float(x) for x in coverage])

    purity = list(purity)
    purity = list([float(x) for x in purity])

    with open(f"./eval/{dset}_beta={thresh_att}_gamma={thresh_overlap}_{args.name}.json", "w") as write_file:
        json.dump({"completeness": coverage, 'purity':  purity,
                   'A': A, 'founded concept ratio:':len(set(A)) / len(A),
                   'ois_cpt': ois, 'nis_cpt': nis, 'ois_map': ois_map, 'nis_map': nis_map,
                   }, write_file)

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


