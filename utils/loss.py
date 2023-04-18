# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria 二分类交叉熵损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n: # a, b, gi, gj的shape均是(156), p[0]的shape(4,3,80,80,85)
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions, 筛出mask掩码为true的格子
                # pxy, pwh的shape均是(156,2), pcls的shape是(156,80), _的shape(156,1)

                # Regression 计算CIOU
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i] # anchors[0]的shape(156,2), 这里为什么是(0,4)的原因可能与超参数anchor_t有关
                pbox = torch.cat((pxy, pwh), 1)  # predicted box 下面的squeeze()能够除去数值为1的维度
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target) iou的shape(156) 这里的pbox与tbox都是在小格子上归一化后的数据
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness 计算有物体置信度
                iou = iou.detach().clamp(0).type(tobj.dtype) # detach()用于从当前计算图中分离张量。它返回一个不需要梯度的新张量
                if self.sort_obj_iou: # detach()用于阻断梯度反向传播, 默认False
                    j = iou.argsort() # argsort()用于返回从小到大排序的坐标
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1: # 默认为1.0
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio tobj的shape(4,3,80,80)
                # iou中不为0的数有155个，但是tobj中只有151个，可能有些数据下标重合了

                # Classification 计算分类概率
                if self.nc > 1:  # cls loss (only if multiple classes) torch.full_like(shape, val)会生成一个值全为val，形状为shape的tensor
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets self.cn=0
                    t[range(n), tcls[i]] = self.cp # range(n)和tcls[0]的shape都是(156), self.cp=1仅mask掩码为true的预测框需要计算分类损失
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj) # pi[..., 4]得到的shape为(4,3,80,80,1)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance: # 默认为False
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box'] # 0.05 CIOU
        lobj *= self.hyp['obj'] # 1.0  置信度
        lcls *= self.hyp['cls'] # 0.5  分类概率
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach() # Tensor.detach() 的作用是阻断反向梯度传播

    def build_targets(self, p, targets): # targets的shape=[42, 6]
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], [] # .repeat(2, 3), 会把shape为[1, 2]的tensor复制变为shape是[2, 6]的tensor
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain .repeat()函数会在对应维度复制tensor个数
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices，ai的shape=[3,42]
        # targets的shape=[3,42,7], 由[3,42,6]+[3,42,1]拼接而成
        # targets的shape[0]表示anchor个数，shape[1]表示目标框个数，shape[2]表示数据个数
        # 其中shape[2]为7，7个数据解释为[img_idx, class, x, y, w, h, anchor_idx]
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0], # 原来的格子，保持不变
                [1, 0], # 左格子，round(x-0.5)
                [0, 1], # 上格子，round(y-0.5)
                [-1, 0], # 右格子，round(x+0.5)
                [0, -1],  # j,k,l,m 下格子，round(y+0.5)
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl): # anchors的shape为[3,3,2] p是predictions的缩写，里面有三个tensor
            # p[0]shape(4,3,80,80,85)
            # p[1]shape(4,3,40,40,85)
            # p[2]shape(4,3,20,20,85)
            anchors, shape = self.anchors[i], p[i].shape # anchors是yaml文件中的anchors三个元素除以8,16,32
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors 这里的targets的shape=[3,42,7]
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches 这一步是求宽高比在1/4~4之间的框
                r = t[..., 4:6] / anchors[:, None]  # wh ratio r的shape=[3,42,2], anchors的shape=[3,2]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare 这里的j就是mask掩码
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                sf = j.float().view(-1).sum() # 通过sf得出j中True的个数为53个
                t = t[j]  # filter t的shape(3,42,7)->(53,7), j的shape为(3,42), 以一个tensor作为索引取值左端对齐

                # Offsets
                gxy = t[:, 2:4]  # grid xy gxy的shape=(53,2)
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T # j, k的shape都是(53), 此处是求是否取另外两个框进行计算
                l, m = ((gxi % 1 < g) & (gxi > 1)).T # .T是求tensor的转置 下面j的shape是(5, 53)
                j = torch.stack((torch.ones_like(j), j, k, l, m)) # 函数stack()对序列数据内部的张量进行扩维拼接，默认第0维
                t = t.repeat((5, 1, 1))[j] # ones_like()基本功能是根据给定张量，生成与其形状相同的全1张量或全0张量,此处t的shape(156,7)
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # .repeat()右端对齐,比如(2,3).repeat(3,1,2)->(3,2,6)
            else: # 上面的加法是(1,53,2) + (5,1,2) -> (5,53,2)广播机制右端对齐
                t = targets[0]
                offsets = 0

            # Define           .chunk(4, 1)表示在第1维上分为4份, t的shape为(156,7),因此会分为(156,2),(156,2),(156,2)和(156,1)
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long() # long()型数据是int64, offsets的shape(156,2)
            gi, gj = gij.T  # grid indices gij的shape是(156,2), gi, gj分别为另外两个格子的x和y坐标（左上角的坐标，所以都是整数）

            # Append                 clamp_()函数用于限定tensor数据的上下界
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box 把新的box的x,y,w,h数据拼好tbox里第一个tuple中tensor的shape(156,4)
            anch.append(anchors[a])  # anchors 上面gxy - gij每个值的范围都是(-0.5, 1.5)，上面的操作相当于在对应的小格子上归一化，gwh的数据并也是归一化的
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
