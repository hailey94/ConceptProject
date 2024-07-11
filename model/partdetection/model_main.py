from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.contrast.slots import ScouterAttention, vis
from model.contrast.position_encode import build_position_encoding
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)
    if args.dataset == 'imagenet':
        return bone
    else:
        if args.dataset == "MNIST":
            bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)

        bone.global_pool = Identical()
        bone.fc = Identical()
        return bone


def bernoulli_kl(p, q, eps=1e-7):
    return (p * ((p + eps).log() - (q + eps).log())) + (1. - p) * ((1. - p + eps).log() - (1. - q + eps).log())


def bin_concrete_sample(a, temperature, eps=1e-8):
    """"
    Sample from the binary concrete distribution
    """

    U = torch.rand_like(a).clamp(eps, 1. - eps) # mean 0, std 1 인 normal distribution 에서 random sampling same size as a, [eps, 1 - eps] range 로 clomp
    L = torch.log(U) - torch.log(1. - U)
    X = torch.sigmoid((L + a) / temperature)

    return X

class DiscoveryMechanism(nn.Module):
    def __init__(self, feat_dim, cdim, prior):
        super(DiscoveryMechanism, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, cdim))
        self.bias = nn.Parameter(torch.Tensor(cdim))
        self.register_buffer('temp', torch.tensor(0.1))
        self.register_buffer('temptest', torch.tensor(.01))

        self.register_buffer('prior', torch.tensor(prior))

        torch.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.0)

    def forward(self, features, train=True, probs_only=False):
        logits = nn.functional.linear(features, self.W.T, self.bias)
        kl = 0.

        if train:
            out = bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
            kl = bernoulli_kl(torch.sigmoid(logits), self.prior).sum(1).mean()

        else:
            out = torch.sigmoid(logits)
            # if probs_only:
            #     out = torch.sigmoid(logits)
            # else:
            #     out = RelaxedBernoulli(self.temptest, logits=logits).sample()

        return out, kl


class MMDiscovery(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(MMDiscovery, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 #+ 1024
        else:
            self.num_features = 512 #+ 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.vis = vis
        self.index = index
        self.dataset = args.dataset

        self.back_bone = load_backbone(args)

        self.fc_landmarks = torch.nn.Conv2d(self.num_features, self.num_concepts, 1, bias=False)
        self.classifier = torch.nn.Linear(self.num_concepts, num_classes)

        self.activation = nn.Tanh()
        self.softmax= torch.nn.Softmax(dim=1)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.disc = DiscoveryMechanism(self.num_features, self.num_concepts, prior=0.0001)


    def forward(self, x, weight=None, train=True, mask=True):
        if self.dataset == 'imagenet':
            intermediate = {}
            def get_activation(name):
                def hook(model, input, output):
                    intermediate[name] = output.detach()
                return hook

            self.back_bone.layer4.register_forward_hook(get_activation('layer4'))
            prediction = self.back_bone(x)
            x = intermediate['layer4']

            if not self.pre_train:
                batch_size = x.shape[0]
                ab = self.fc_landmarks(x)
                b_sq = x.pow(2).sum(1, keepdim=True)
                b_sq = b_sq.expand(-1, self.num_concepts, -1, -1)  # B C W H
                a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2],
                                                                                  x.shape[-1])
                a_sq = a_sq.permute(1, 0, 2, 3)
                maps = b_sq - 2 * ab + a_sq  # B C W H

                maps_ = -maps
                # Softmax so that the attention maps for each pixel add up to 1
                maps_ = self.softmax(maps_)  # # B C W H default dim = 1

                updates = torch.einsum('bdj,bcj->bcd',
                                       x.reshape(batch_size, self.num_features, -1),
                                       maps.reshape(batch_size, self.num_concepts, -1))

                x = self.avgpool(x).squeeze()
                z_samples, kl = self.disc(x, train=train)
                z_samples = z_samples.view([batch_size, self.num_concepts, 1, 1])
                maps = maps_ * z_samples

                attn_cls = torch.sum(maps, dim=[-2, -1])
                cpt = self.activation(attn_cls)
                scores = self.classifier(cpt)

            else:
                print(prediction.shape)
                return prediction, x #prediction, features

        else:
            x = self.back_bone(x)  #B D W H

            batch_size = x.shape[0]
            ab = self.fc_landmarks(x)
            b_sq = x.pow(2).sum(1, keepdim=True)
            b_sq = b_sq.expand(-1, self.num_concepts, -1, -1) # B C W H
            a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2],
                                                                              x.shape[-1])
            a_sq = a_sq.permute(1, 0, 2, 3)
            maps = b_sq - 2 * ab + a_sq # B C W H

            maps_ = -maps
            # Softmax so that the attention maps for each pixel add up to 1
            maps_ = self.softmax(maps_)  # # B C W H default dim = 1

            updates = torch.einsum('bdj,bcj->bcd',
                                   x.reshape(batch_size, self.num_features, -1), maps.reshape(batch_size, self.num_concepts, -1))

            x = self.avgpool(x).squeeze()
            z_samples, kl = self.disc(x, train=train)
            z_samples = z_samples.view([batch_size, self.num_concepts, 1, 1])
            maps = maps_ * z_samples

            attn_cls = torch.sum(maps, dim=[-2, -1])
            cpt = self.activation(attn_cls)
            scores = self.classifier(cpt)

        if mask:
            return cpt, scores, maps.reshape(batch_size, self.num_concepts, -1), updates, kl # maps_
        else:
            return cpt, scores, maps_.reshape(batch_size, self.num_concepts, -1), updates, kl  # maps_



import torch

from torch import Tensor


def gumbel_sigmoid(logits: Tensor, tau: float = 1, hard: bool = False, threshold: float = 0.5) -> Tensor:
    """
    Samples from the Gumbel-Sigmoid distribution and optionally discretizes.
    The discretization converts the values greater than `threshold` to 1 and the rest to 0.
    The code is adapted from the official PyTorch implementation of gumbel_softmax:
    https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized,
            but will be differentiated as if it is the soft sample in autograd
     threshold: threshold for the discretization,
                values greater than this will be set to 1 and the rest to 0

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Sigmoid distribution.
      If ``hard=True``, the returned samples are descretized according to `threshold`, otherwise they will
      be probability distributions.

    """
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class ProjectionOnly(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(ProjectionOnly, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 #+ 1024
        else:
            self.num_features = 512 #+ 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.vis = vis
        self.index = index
        self.dataset = args.dataset

        self.back_bone = load_backbone(args)

        self.fc_landmarks = torch.nn.Conv2d(self.num_features, self.num_concepts, 1, bias=False)
        self.classifier = torch.nn.Linear(self.num_concepts, num_classes)

        self.activation = nn.Tanh()
        self.softmax= torch.nn.Softmax(dim=1)

        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)


    def forward(self, x, weight=None, train=True, mask=True):
        if self.dataset == 'imagenet':
            print('nope')
            exit()
        else:
            x = self.back_bone(x)  #B D W H

            batch_size = x.shape[0]
            ab = self.fc_landmarks(x)
            b_sq = x.pow(2).sum(1, keepdim=True)
            b_sq = b_sq.expand(-1, self.num_concepts, -1, -1) # B C W H
            a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, x.shape[-2],
                                                                              x.shape[-1])
            a_sq = a_sq.permute(1, 0, 2, 3)
            maps = b_sq - 2 * ab + a_sq # B C W H

            maps = -maps
            # Softmax so that the attention maps for each pixel add up to 1
            maps_softmax = self.softmax(maps)  # # B C W H default dim = 1

            updates = torch.einsum('bdj,bcj->bcd',
                                   x.reshape(batch_size, self.num_features, -1), maps_softmax.reshape(batch_size, self.num_concepts, -1))

            cpt = self.avgpool(maps_softmax).squeeze()

            scores = self.classifier(cpt)

        return cpt, scores, cpt.reshape(batch_size, self.num_concepts, -1), updates  # maps_


class DiscoveryOnly_P(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(DiscoveryOnly_P, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 #+ 1024
        else:
            self.num_features = 512 #+ 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.vis = vis
        self.index = index
        self.dataset = args.dataset

        self.back_bone = load_backbone(args)

        self.fc_landmarks = torch.nn.Linear(self.num_features, self.num_concepts)
        self.classifier = torch.nn.Linear(self.num_concepts, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.disc = DiscoveryMechanism(self.num_features, self.num_concepts, prior=0.0001)


    def forward(self, x, weight=None, train=True, mask=True):
        if self.dataset == 'imagenet':
            print()
            exit()
        else:
            x = self.back_bone(x)  #B D W H

            batch_size = x.shape[0]
            x = self.avgpool(x).squeeze()
            cpt, kl = self.disc(x, train=train)

            # x = self.fc_landmarks(x)
            # cpt = gumbel_sigmoid(x, tau=0.5, hard=False)
            scores = self.classifier(cpt)

            updates = torch.einsum('bd,bc->bdc', x, cpt)


        return cpt, scores, x, updates, kl

