# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
AutoAnchor utils
"""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr('AutoAnchor: ')


def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f'{PREFIX}Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f'{PREFIX}ERROR')
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    # Check anchor fit to data, recompute if necessary
    m = model.module.model[-1] if hasattr(model, 'module') else model.model[-1]  # Detect() 下方的shapes是每张图片的[width,height], shapes.shape(2636,2)
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True) # max(1, keepdims=True)表示选出第1维中最大的元素，且保持维度显示
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale l[:, 3:5]表示取出l所有三四列的数据
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh shape(3635,2)
    # l表示每张图片中label的数据（可能有多个label）每个label的数据格式为[class, x, y, w, h]xywh是归一化数据
    def metric(k):  # compute metric
        r = wh[:, None] / k[None] # shape(3635,1,2) / shape(1,9,2) r的shape(3635,9,2)
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric shape(3635,9) 取出框中宽高比例最极端的值
        best = x.max(1)[0]  # best_x shape(3635) 对每个label，取宽高比例中最接近的那个框
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold 只有 1/4 < r < 4的框是符合条件的
        bpr = (best > 1 / thr).float().mean()  # best possible recall 9个anchors中最好的那一个，然后在所有labels中求符合条件的均值
        return bpr, aat # aat: 在一组anchors(9个框)中平均每个label有符合aat个的anchor满足thr，bpr: 一组anchors中有一个与label满足thr的比例

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model stride值为[8,16,32],shape(3,1,1)
    anchors = m.anchors.clone() * stride  # current anchors shape(3,3,2)
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f'\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). '
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f'{s}Current anchors are a good fit to dataset ✅')
    else:
        LOGGER.info(f'{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...')
        na = m.anchors.numel() // 2  # number of anchors numel()会数出所有的值的个数，因此不需要指定维数
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False) # shape(9,2)
        new_bpr = metric(anchors)[0]
        if new_bpr > 0.3:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors) # 赋值
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f'{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)'
        else:
            s = f'{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)'
        LOGGER.info(s)


def kmean_anchors(dataset='./data/coco128.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """ Creates kmeans-evolved anchors from training dataset

        Arguments:
            dataset: path to data.yaml, or a loaded dataset
            n: number of anchors
            img_size: image size used for training
            thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
            gen: generations to evolve anchors using genetic algorithm
            verbose: print all results

        Return:
            k: kmeans evolved anchors

        Usage:
            from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics k[None]表示在最开始加一维，比如shape(2, 3)变为shape(1, 2, 3)
        r = wh[:, None] / k[None] # wh[:, None]表示在第二维加一维如shape(2, 3)变为shape(2, 1, 3)
        # tensor.min(i)会返回第i维上的最小值构成的tensor与其坐标
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric torch.min(a, b)会分别选择a, b中的较小数据拼成一个新的tensor且shape不变
        # x = wh_iou(wh, torch.tensor(k))  # iou metric 上面的.min(2)表示在第二维上求最小值（维度从0开始）
        return x, x.max(1)[0]  # x, best_x [0]表示不需要坐标，只保留数据

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large prod是连乘操作，会将里面的所有元素相乘
        x, best = metric(k, wh0)
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = f'{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n' \
            f'{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, ' \
            f'past_thr={x[x > thr].mean():.3f}-mean: '
        for x in k:
            s += '%i,%i, ' % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors='ignore') as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels
        dataset = LoadImagesAndLabels(data_dict['train'], augment=True, rect=True)

    # Get label wh
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True) # 设置keepdims保持结果的维数不变 shape(2635,2)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh zip把对应函数列打包成一系列元组

    # Filter
    i = (wh0 < 3.0).any(1).sum() # any(1)会列出(who < 3.0)构成的tensor的第二维度中含有True的列，然后把他们组合成一个新的tensor
    if i:
        LOGGER.info(f'{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size')
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f'{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...') # n=9, len(wh)=3635
        assert n <= len(wh)  # apply overdetermined constraint wh.shape(3635,2)
        s = wh.std(0)  # sigmas for whitening .std()计算的是样本标准偏差除的是n-1，该函数会分别求出x列与y列的样本标准偏差
        k = kmeans(wh / s, n, iter=30)[0] * s  # points wh/s的样本标准差就变成了1
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f'{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init')
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
