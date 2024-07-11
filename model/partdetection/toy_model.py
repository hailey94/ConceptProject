from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from model.contrast.slots import ScouterAttention, vis
from model.contrast.position_encode import build_position_encoding


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


class MainModel(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 + 1024
        else:
            self.num_features = 512 + 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        self.num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis
        self.index = index
        self.dataset = args.dataset
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = nn.BatchNorm2d(11)
        landmark_dropout = 0.3

        self.conv1 = self.back_bone.conv1
        self.bn1 = self.back_bone.bn1
        self.relu = nn.ReLU()
        self.maxpool = self.back_bone.maxpool
        self.layer1 = self.back_bone.layer1
        self.layer2 = self.back_bone.layer2
        self.layer3 = self.back_bone.layer3
        self.layer4 = self.back_bone.layer4


        self.fc_landmarks = torch.nn.Conv2d(self.num_features, self.num_concepts, 1, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(self.num_features, self.num_classes, bias=False)
        self.modulation = torch.nn.Parameter(torch.ones((1, self.num_features, self.num_concepts)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

    def forward(self, x, weight=None, things=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear')
        x = torch.cat((x, l3), dim=1)  # B D W H

        batch_size = x.shape[0]
        w, h = x.shape[-1], x.shape[-2]
        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)
        b_sq = b_sq.expand(-1, self.num_concepts, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, w, h)
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps = b_sq - 2 * ab + a_sq # B C W H
        maps_ = -maps
        maps = nn.functional.softmax(maps_, dim=1)  # self.softmax(maps_) #: spatial

        attn_cls = torch.sum(maps, dim=[-2, -1]) # B C
        feature_tensor = x
        # all_features = ((maps).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-3) # B C W H

        all_features = ((maps).unsqueeze(1) * feature_tensor.unsqueeze(2)).mean(-1).mean(-1) # B C D
        # Classification based on the landmarks
        all_features_modulated = all_features * self.modulation
        # all_features_modulated = all_features
        if not self.infer:
            all_features_modulated = self.dropout_full_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0,
                                                                                                              2,
                                                                                                  1)

        scores = self.fc_class_landmarks(all_features_modulated.permute(0, 2, 1)).permute(0, 2, 1)
        scores = scores.sum(-1)


        return all_features_modulated, scores, maps, attn_cls # maps_

class Bottleneck1x1(nn.Module):
    expansion = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MainModel2(MainModel):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(MainModel2, self).__init__(args, vis=False, index=None, infer=False)

        self.fc_class_landmarks = torch.nn.Linear(self.num_concepts, self.num_classes, bias=False)
        self.smooth_factor = nn.Parameter(torch.FloatTensor(self.num_concepts))
        nn.init.kaiming_normal_(self.fc_landmarks.weight)
        self.fc_landmarks.weight.data.clamp_(min=1e-5)
        nn.init.constant_(self.smooth_factor, 0)

        # an attention for each classification head
        self.attconv = nn.Sequential(
            Bottleneck1x1(self.num_features, 256, stride=1),
            Bottleneck1x1(self.num_features, 256, stride=1),
            nn.Conv2d(self.num_features, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        # the final batchnorm
        self.groupingbn = nn.BatchNorm2d(self.num_concepts)

    def forward(self, x, weight=None, things=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear')
        x = torch.cat((x, l3), dim=1)  # b hidden1+hidden2 w h

        batch_size = x.shape[0]
        w, h = x.shape[-2], x.shape[-1]

        ab = self.fc_landmarks(x)
        b_sq = x.pow(2).sum(1, keepdim=True)

        b_sq = b_sq.expand(-1, self.num_concepts, -1, -1)
        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(1).expand(-1, batch_size, w, h)
        a_sq = a_sq.permute(1, 0, 2, 3)
        maps = b_sq - 2 * ab + a_sq
        maps_ = -maps

        # Softmax so that the attention maps for each pixel add up to 1
        maps = nn.functional.softmax(maps_, dim=1)  # default dim = 1

        # 3. compute residual coding
        x = x.contiguous().view(batch_size, self.num_features, -1) # NCHW -> B * D * HW
        x = x.permute(0, 2, 1) # inputs -> B * HW * D

        # compute weighted feats B * C * D
        assign = maps.contiguous().view(batch_size, self.num_concepts, -1)
        qx = torch.bmm(assign, x) # (B * C * D)

        # repeat the graph_weights (C * D) -> (B * C * D)
        c = self.fc_landmarks.weight.contiguous()\
            .view(1, self.num_concepts, self.num_features)\
            .expand(batch_size, self.num_concepts, self.num_features)

        # sum_ass = torch.sum(assign, dim=2, keepdim=True) # (B * C * 1)
        # sum_ass = sum_ass.expand(-1, -1, self.num_features).clamp(min=1e-5)  # (B * C * D)

        # residual coding
        # out = ((qx / sum_ass) - c)
        z = qx - c

        # 4. prepare outputs
        assign = assign.contiguous().view(batch_size, self.num_concepts, w, h) #  memorize the assignment (B * C * H * W)

        # output features has the size of B * C * D
        z = nn.functional.normalize(z, dim=2)
        z_t = z.permute(0, 2, 1)
        z_t = z_t.contiguous().unsqueeze(3)  # B, D, C, 1

        # generate region attention
        att = self.attconv(z_t)
        att = F.softmax(att, dim=2) # B, 1, C, 1
        # att = F.sigmoid(att)  # B, 1, C, 1

        out = z_t * att # apply the attention on the features
        out_ = out.contiguous().squeeze(3).permute(0,2,1)  # B, D, C

        # average all region features into one vector based on the attention
        out = F.avg_pool1d(out_, self.num_features) * self.num_features
        out = out.contiguous().unsqueeze(3)
        # final bn
        out = self.groupingbn(out)

        # linear classifier
        out = out.contiguous().view(out.size(0), -1)
        pred = self.fc_class_landmarks(out)
        return out_, pred, assign, out



class MyModel(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(MyModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 + 1024
        else:
            self.num_features = 512 + 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        self.num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis
        self.index = index
        self.dataset = args.dataset
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = nn.BatchNorm2d(11)
        landmark_dropout = 0.3

        self.conv1 = self.back_bone.conv1
        self.bn1 = self.back_bone.bn1
        self.relu = nn.ReLU()
        self.maxpool = self.back_bone.maxpool
        self.layer1 = self.back_bone.layer1
        self.layer2 = self.back_bone.layer2
        self.layer3 = self.back_bone.layer3
        self.layer4 = self.back_bone.layer4

        # self.fc_landmarks = torch.nn.Conv2d(self.num_features, self.num_concepts, 1, bias=False)
        self.fc_landmarks = torch.nn.Linear(196, self.num_concepts, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(self.num_concepts, self.num_classes, bias=False)
        # self.modulation = torch.nn.Parameter(torch.ones((1, self.num_features, self.num_concepts)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

        nn.init.kaiming_normal_(self.fc_landmarks.weight)
        self.fc_landmarks.weight.data.clamp_(min=1e-5)
        nn.init.kaiming_normal_(self.fc_class_landmarks.weight)
        self.fc_class_landmarks.weight.data.clamp_(min=1e-5)

    def forward(self, x, weight=None, things=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear')

        x = torch.cat((x, l3), dim=1)  # B D WH
        batch_size, d, w, h = x.shape
        x = x.view(batch_size, d, -1)

        ab = self.fc_landmarks(x) # B D C
        b_sq = x.pow(2).sum(-1, keepdim=True) # B D 1
        b_sq = b_sq.expand(-1, -1, self.num_concepts)

        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(0).expand(batch_size, d, -1)
        maps = b_sq - 2 * ab + a_sq # B D C

        maps_ = -maps
        maps_ = nn.functional.softmax(maps_, dim=-1) # B D C # self.softmax(maps_) #: spatial

        maps = torch.einsum('bdi,bdj->bij', maps_, x) # B C WH # x의 pixel 들이 각각의 c로 mapping
        sum_maps = torch.sum(maps, dim=[-1]) # B C
        sum_maps = nn.functional.normalize(sum_maps, dim=-1)
        cpt = self.activation(sum_maps)
        if not self.infer:
            cpt = self.dropout_full_landmarks(cpt)

        scores = self.fc_class_landmarks(cpt)

        return (cpt - 0.5) * 2, scores, maps, maps_


class MyModelCUB(nn.Module):
    def __init__(self, args, vis=False, index=None, infer=False):
        super(MyModelCUB, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        self.infer = infer
        if "18" not in args.base_model:
            self.num_features = 2048 + 1024
        else:
            self.num_features = 512 + 256
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.num_concepts = args.num_cpt
        self.num_classes = args.num_classes
        self.back_bone = load_backbone(args)
        self.activation = nn.Tanh()
        self.vis = vis
        self.index = index
        self.dataset = args.dataset
        self.softmax: Softmax2d = torch.nn.Softmax2d()
        self.batchnorm = nn.BatchNorm2d(11)
        landmark_dropout = 0.3

        self.conv1 = self.back_bone.conv1
        self.bn1 = self.back_bone.bn1
        self.relu = nn.ReLU()
        self.maxpool = self.back_bone.maxpool
        self.layer1 = self.back_bone.layer1
        self.layer2 = self.back_bone.layer2
        self.layer3 = self.back_bone.layer3
        self.layer4 = self.back_bone.layer4

        # self.fc_landmarks = torch.nn.Conv2d(self.num_features, self.num_concepts, 1, bias=False)
        self.fc_landmarks = torch.nn.Linear(196, self.num_concepts, bias=False)
        self.fc_class_landmarks = torch.nn.Linear(self.num_concepts, self.num_classes, bias=False)
        # self.modulation = torch.nn.Parameter(torch.ones((1, self.num_features, self.num_concepts)))
        self.dropout = torch.nn.Dropout(landmark_dropout)
        self.dropout_full_landmarks = torch.nn.Dropout1d(landmark_dropout)

        nn.init.kaiming_normal_(self.fc_landmarks.weight)
        self.fc_landmarks.weight.data.clamp_(min=1e-5)
        nn.init.kaiming_normal_(self.fc_class_landmarks.weight)
        self.fc_class_landmarks.weight.data.clamp_(min=1e-5)

    def forward(self, x, weight=None, things=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        l3 = self.layer3(x)
        x = self.layer4(l3)
        x = torch.nn.functional.interpolate(x, size=(l3.shape[-2], l3.shape[-1]), mode='bilinear')

        x = torch.cat((x, l3), dim=1)  # B D WH
        batch_size, d, w, h = x.shape
        x = x.view(batch_size, d, -1)

        ab = self.fc_landmarks(x) # B D C
        b_sq = x.pow(2).sum(-1, keepdim=True) # B D 1
        b_sq = b_sq.expand(-1, -1, self.num_concepts)

        a_sq = self.fc_landmarks.weight.pow(2).sum(1).unsqueeze(0).expand(batch_size, d, -1)
        maps = b_sq - 2 * ab + a_sq # B D C

        maps_ = -maps
        maps_ = nn.functional.softmax(maps_, dim=-1) # B D C # self.softmax(maps_) #: spatial

        maps = torch.einsum('bdi,bdj->bij', maps_, x) # B C WH # x의 pixel 들이 각각의 c로 mapping
        sum_maps = torch.sum(maps, dim=[-1]) # B C
        cpt = nn.functional.normalize(sum_maps, dim=-1)
        cpt = self.activation(sum_maps)
        if not self.infer:
            cpt = self.dropout_full_landmarks(cpt)

        scores = self.fc_class_landmarks(cpt)

        return (cpt - 0.5) * 2, scores, maps, maps_

# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)


