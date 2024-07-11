import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as visionF
import numpy as np


def rigid_transform(img: torch.Tensor, angle: int, translate: [int], scale: float, invert: bool=False):
    """
    Affine transforms input image
    Parameters
    ----------
    img: torch.Tensor
        Input image
    angle: int
        Rotation angle between -180 and 180 degrees
    translate: [int]
        Sequence of horizontal/vertical translations
    scale: float
        How to scale the image
    invert: bool
        Whether to invert the transformation

    Returns
    ----------
    img: torch.Tensor
        Transformed image
    """
    shear = 0
    bilinear = visionF.InterpolationMode.BILINEAR
    if not invert:
        img = visionF.affine(img, angle, translate, scale, shear,
                             interpolation=bilinear)
    else:
        translate = [-t for t in translate]
        img = visionF.affine(img, 0, translate, 1, shear)
        img = visionF.affine(img, -angle, [0, 0], 1/scale, shear)
    return img



def equiv_loss_origin(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    _, _, equiv_map, _ = net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps[:, :-1, :, :], (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back[:, 0:-1, :, :], (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv

def equiv_loss(X: torch.Tensor, maps: torch.Tensor, net: torch.nn.Module, device: torch.device, num_parts: int) \
        -> torch.Tensor:
    """
    Calculates the equivariance loss, which we calculate from the cosine similarity between the original attention map
    and the inversely transformed attention map of a transformed image.
    Parameters
    ----------
    X: torch.Tensor
        The input image
    maps: torch.Tensor
        The attention maps
    net: torch.nn.Module
        The model
    device: torch.device
        The device to use
    num_parts: int
        The number of landmarks

    Returns
    -------
    loss_equiv: torch.Tensor
        The equivariance loss
    """
    # Forward pass
    angle = np.random.rand() * 180 - 90
    translate = list(np.int32(np.floor(np.random.rand(2) * 100 - 50)))
    scale = np.random.rand() * 0.6 + 0.8
    transf_img = rigid_transform(X, angle, translate, scale, invert=False)
    _, _, equiv_map, _ = net(transf_img.to(device))

    # Compare to original attention map, and penalise high difference
    translate = [(t * maps.shape[-1] / X.shape[-1]) for t in translate]
    rot_back = rigid_transform(equiv_map, angle, translate, scale, invert=True)
    num_elements_per_map = maps.shape[-2] * maps.shape[-1]
    orig_attmap_vector = torch.reshape(maps, (-1, num_parts, num_elements_per_map))
    transf_attmap_vector = torch.reshape(rot_back, (-1, num_parts, num_elements_per_map))
    cos_sim_equiv = F.cosine_similarity(orig_attmap_vector, transf_attmap_vector, -1)
    loss_equiv = 1 - torch.mean(cos_sim_equiv)
    return loss_equiv

def conc_loss_origin(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, 0:-1, :, :].mean()
    return loss_conc

def conc_loss(centroid_x: torch.Tensor, centroid_y: torch.Tensor, grid_x: torch.Tensor, grid_y: torch.Tensor,
              maps: torch.Tensor) -> torch.Tensor:
    """
    Calculates the concentration loss, which is the weighted sum of the squared distance of the landmark
    Parameters
    ----------
    centroid_x: torch.Tensor
        The x coordinates of the map centroids
    centroid_y: torch.Tensor
        The y coordinates of the map centroids
    grid_x: torch.Tensor
        The x coordinates of the grid
    grid_y: torch.Tensor
        The y coordinates of the grid
    maps: torch.Tensor
        The attention maps

    Returns
    -------
    loss_conc: torch.Tensor
        The concentration loss
    """
    spatial_var_x = ((centroid_x.unsqueeze(-1).unsqueeze(-1) - grid_x) / grid_x.shape[-1]) ** 2
    spatial_var_y = ((centroid_y.unsqueeze(-1).unsqueeze(-1) - grid_y) / grid_y.shape[-2]) ** 2
    spatial_var_weighted = (spatial_var_x + spatial_var_y) * maps
    loss_conc = spatial_var_weighted[:, :, :, :].mean()
    return loss_conc


def orth_loss_origin(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
    similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts + 1).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth

def orth_loss(num_parts: int, landmark_features: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    landmark_features: torch.Tensor, [batch_size, feature_dim, num_landmarks + 1 (background)]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    normed_feature = torch.nn.functional.normalize(landmark_features, dim=1)
    similarity = torch.matmul(normed_feature.permute(0, 2, 1), normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth


def concept_orth_loss(num_parts: int, attn: torch.Tensor, device) -> torch.Tensor:
    """
    Calculates the orthogonality loss, which is the mean of the cosine similarities between every pair of landmarks
    Parameters
    ----------
    num_parts: int
        The number of landmarks
    attn: torch.Tensor, [batch_size, num_cpt, w, h]
        Tensor containing the feature vector for each part
    device: torch.device
        The device to use
    Returns
    -------
    loss_orth: torch.Tensor
        The orthogonality loss
    """
    if len(attn.shape) == 4:
        attn = attn.reshape([attn.shape[0], num_parts, -1])

    normed_feature = torch.nn.functional.normalize(attn, dim=-1)
    similarity = torch.einsum('bid,bjd->bij', normed_feature, normed_feature)
    similarity = torch.sub(similarity, torch.eye(num_parts).to(device))
    loss_orth = torch.mean(torch.square(similarity))
    return loss_orth

def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1.):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    mask_positive = similarity > 0
    mask_negative = similarity <= 0
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0+S1

    exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
    exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)

    loss = torch.mean(exp_loss)

    return loss


def pairwise_similarity_label(label):
    pair_label = F.cosine_similarity(label[:, None, :], label[None, :, :], dim=2)
    return pair_label


def soft_similarity(features, label):
    s_loss = torch.abs(torch.sigmoid(features) - label)
    return torch.mean(s_loss)


def hard_similarity(dot_product, similarity):
    exp_product = torch.exp(dot_product)
    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    mask_positive = similarity > 0
    mask_negative = similarity <= 0
    S1 = torch.sum(mask_positive.float())
    S0 = torch.sum(mask_negative.float())
    S = S0 + S1

    exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
    exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)
    loss = torch.mean(exp_loss)

    return loss


def quantization_loss(cpt):
    q_loss = torch.mean((torch.abs(cpt)-1.0)**2)
    return q_loss



def get_retrieval_loss(args, y, label, num_cls, device):
    b = label.shape[0]
    if args.dataset != "matplot":
        label = label.unsqueeze(-1)
        label = torch.zeros(b, num_cls).to(device).scatter(1, label, 1)
    similarity_loss = pairwise_loss(y, y, label, label, sigmoid_param=10. / 32)
    q_loss = quantization_loss(y)
    return similarity_loss, q_loss

def att_binary(att):
    att = (att - 0.5) * 2
    return torch.mean((torch.abs(att)-1.0)**2)


def att_discriminate(att):
    b, cpt, spatial = att.size()
    att_mean = torch.sum(att, dim=-1)
    dis_loss = 0.0
    for i in range(b):
        current_mean = att_mean[i].mean()
        indices = att_mean[i] > current_mean
        need = att[i][indices]
        dis_loss += torch.tanh(((need[None, :, :] - need[:, None, :]) ** 2).sum(-1)).mean()
    return dis_loss/b


def batch_cpt_discriminate(data, att):
    b1, d1, c = data.shape
    record = []

    for i in range(c):
        current_f = data[:, :, i]
        current_att = att.sum(-1)[:, i]
        indices = current_att > current_att.mean()
        b, d = current_f[indices].shape
        current_f = current_f[indices]
        record.append(torch.mean(current_f, dim=0, keepdim=True))
    record = torch.cat(record, dim=0)
    sim = F.cosine_similarity(record[None, :, :], record[:, None, :], dim=-1)
    return sim.mean()



def att_consistence(update, att):
    b, cpt, spatial = att.size()
    consistence_loss = 0.0
    for i in range(cpt):
        current_up = update[:, i, :]
        current_att = att[:, i, :].sum(-1)
        indices = current_att > current_att.mean()
        b, d = current_up[indices].shape
        need = current_up[indices]
        consistence_loss += F.cosine_similarity(need[None, :, :], need[:, None, :], dim=-1).mean()
    return consistence_loss/cpt



def att_area_loss(att):
    slot_loss = torch.sum(att, (0, 1, 2)) / att.size(0) / att.size(1) / att.size(2)
    return torch.pow(slot_loss, 1)

# def area_loss(att):
#     att_existence = Variable(att > 0).float()
#     dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
#     exp_product = torch.exp(dot_product)
#
#     exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
#     mask_positive = att > 0.5
#     mask_negative = att <= 0.5
#     S1 = torch.sum(mask_positive.float())
#     S0 = torch.sum(mask_negative.float())
#     S = S0+S1
#
#     exp_loss[similarity > 0] = exp_loss[similarity > 0] * (S / S1)
#     exp_loss[similarity <= 0] = exp_loss[similarity <= 0] * (S / S0)
#
#     loss = torch.mean(exp_loss)

    # return loss
