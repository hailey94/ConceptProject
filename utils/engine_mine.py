import torch
import torch.nn.functional as F
from model.contrast.loss import get_retrieval_loss, batch_cpt_discriminate, att_consistence, att_discriminate, att_binary, \
    att_area_loss
from .record import AverageMeter, ProgressMeter, show
from .tools import cal_acc, predict_hash_code, mean_average_precision


def train(args, model, device, loader, optimizer, epoch):
    retri_losses = AverageMeter('Retri_loss Loss', ':.4')
    att_losses = AverageMeter('Att Loss', ':.4')
    q_losses = AverageMeter('Q_loss', ':.4')
    batch_dis_losses = AverageMeter('Dis_loss_batch', ':.4')
    consistence_losses = AverageMeter('Consistence_loss', ':.4')
    kl_losses = AverageMeter('kl', ':.4')

    cls_loss = AverageMeter('Cls', ':.4')
    pred_acces = AverageMeter('Acc', ':.4')

    if not args.pre_train:
        show_items = [retri_losses, q_losses, pred_acces, cls_loss, batch_dis_losses, consistence_losses, att_losses, kl_losses]
    else:
        show_items = [pred_acces, cls_loss]
    progress = ProgressMeter(len(loader),
                             show_items,
                             prefix="Epoch: [{}]".format(epoch))

    model.train()
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        if not args.pre_train:
            cpt, pred, att, update, kl = model(data, train=True)

            retri_loss, quantity_loss = get_retrieval_loss(args, cpt, label, args.num_classes, device)
            if args.dataset != "matplot":
                pred = F.log_softmax(pred, dim=-1)
                loss_pred = F.nll_loss(pred, label)
                acc = cal_acc(pred, label, False)
            else:
                pred = F.sigmoid(pred)
                loss_pred = F.binary_cross_entropy(pred, label.float())
                acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]

            batch_dis_loss = batch_cpt_discriminate(update, att)
            consistence_loss = att_consistence(update, att)
            attn_loss = att_area_loss(att)

            retri_losses.update(retri_loss.item())
            att_losses.update(attn_loss.item())
            q_losses.update(quantity_loss.item())
            batch_dis_losses.update(batch_dis_loss.item())
            consistence_losses.update(consistence_loss.item())
            pred_acces.update(acc)
            cls_loss.update(loss_pred)
            kl_losses.update(kl.item())


            loss_total = args.weak_supervision_bias * retri_loss + args.att_bias * attn_loss + args.quantity_bias * quantity_loss + \
                         loss_pred - args.consistence_bias * consistence_loss + args.distinctiveness_bias * batch_dis_loss + 0.0001 * kl
        else:
            pred, features = model(data)
            if args.dataset != "matplot":
                pred = F.log_softmax(pred, dim=-1)
                loss_pred = F.nll_loss(pred, label)
                acc = cal_acc(pred, label, False)
            else:
                pred = F.sigmoid(pred)
                loss_pred = F.binary_cross_entropy(pred, label.float())
                acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]

            pred_acces.update(acc)
            cls_loss.update(loss_pred)
            loss_total = loss_pred

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            progress.display(batch_idx)


@torch.no_grad()
def test(args, model, test_loader, device):
    model.eval()
    record = 0.0
    for batch_idx, (data, label) in enumerate(test_loader):
        data, label = data.to(device, dtype=torch.float32), label.to(device, dtype=torch.int64)
        _, pred, _, _, _ = model(data, train=False)
        # _, pred, _, _ = model(data)

        if args.dataset != "matplot":
            pred = F.log_softmax(pred, dim=-1)
            acc = cal_acc(pred, label, False)
        else:
            pred = F.sigmoid(pred)
            acc = torch.eq(pred.round(), label).sum().float().item() / pred.shape[0] / pred.shape[1]
        record += acc
    ACC = record/len(test_loader)
    print("ACC:", record/len(test_loader))
    return ACC
