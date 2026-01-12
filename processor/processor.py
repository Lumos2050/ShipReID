import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def plot_tsne_with_cam(features, labels, cams, save_path=None):
    # t-SNE 降维到2维
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, init='pca', random_state=42)
    X_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8,8))

    # 生成调色板，颜色数和类别数一致
    unique_labels = np.unique(labels)
    palette = sns.color_palette("hsv", len(unique_labels))
    label_to_color = {lab: palette[i] for i, lab in enumerate(unique_labels)}

    markers = {0: 'o', 1: '^'}  # cam=0圆点，cam=1三角形

    # 遍历所有 cam 和 label 组合分别绘制，方便图例和区分
    for cam_id in np.unique(cams):
        for lab in unique_labels:
            idx = np.where((cams == cam_id) & (labels == lab))
            if len(idx[0]) == 0:
                continue
            plt.scatter(
                X_2d[idx, 0],
                X_2d[idx, 1],
                color=label_to_color[lab],
                marker=markers.get(cam_id, 'o'),
                s=20,
                label=f"Label {lab} Cam {cam_id}"
            )

    plt.title("t-SNE Visualization with CAM markers")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def do_train(cfg,
             model,
             center_criterion,
             train_loader_pair,
             val_loader,
             val_loader_x,
             val_loader_y,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, 
             num_query_x,
             num_query_y, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter_i = AverageMeter()
    loss_meter_t = AverageMeter()
    loss_meter_c = AverageMeter()
    loss_meter_x = AverageMeter()
    loss_meter_y = AverageMeter()
    acc_meter = AverageMeter()
    acc_meter_x = AverageMeter()
    acc_meter_y = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='cross')
    evaluator_x = R1_mAP_eval(num_query_x, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='x')
    evaluator_y = R1_mAP_eval(num_query_y, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='y')

    scaler = amp.GradScaler()#混合精度训练
    # train
    for epoch in range(1, epochs + 1):
        #print('trainloder长度', len(train_loader_pair))
        start_time = time.time()
        loss_meter_i.reset()#动态记录 epoch 内多个 batch 的 loss，加权平均后计算出 epoch 平均损失
        loss_meter_t.reset()
        loss_meter_c.reset()
        loss_meter_x.reset()
        loss_meter_y.reset()
        acc_meter.reset()#动态记录 epoch 内多个 batch 的准确率，加权平均后计算出 epoch 平均准确率
        acc_meter_x.reset()
        acc_meter_y.reset()
        evaluator.reset()
        evaluator_x.reset()
        evaluator_y.reset()
        scheduler.step(epoch)
        model.train()
        train_loader_iter = tqdm(
            train_loader_pair,
            total=len(train_loader_pair),
            desc=f"Epoch {epoch}/{epochs}",
            ncols=150
        )
        for n_iter, (img_x, img_y, vid, camids, _) in enumerate(train_loader_iter):

            #print('img_x的形状', img_x.size())
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            img_x = img_x.to(device)
            img_y = img_y.to(device)
            target = vid.to(device)

            with amp.autocast(enabled=True):
                #score, feat, lorthx_sum, lorthy_sum, lc_sum = model(img1, img2, target, cam_label=target_cam, view_label=target_view )
                score, feat, lorthx_sum, lorthy_sum, lc_sum = model(label=camids, x=img_x, y=img_y, mode='train')
                #print("score,target",score, target)
                id_loss, tri_loss = loss_fn(score, feat, target)
                loss_total = cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss+ cfg.MODEL.C_LOSS_WEIGHT * lc_sum + cfg.MODEL.XT_LOSS_WEIGHT * lorthx_sum + cfg.MODEL.YT_LOSS_WEIGHT * lorthy_sum
                #print("Loss咋回事到底", loss_total, loss, lc_sum, lorthx_sum, lorthy_sum)
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
                acc_x = (score[1].max(1)[1] == target).float().mean()
                acc_y = (score[2].max(1)[1] == target).float().mean()
                acc_meter.update(acc, 1)
                acc_meter_x.update(acc_x, 1)
                acc_meter_y.update(acc_y, 1)
            else:
                acc = (score.max(1)[1] == target).float().mean()
                acc_meter.update(acc, 1)

            loss_meter_i.update(id_loss.item(), img_x.shape[0])#shape0就是batchsize
            loss_meter_t.update(tri_loss.item(), img_x.shape[0])
            loss_meter_c.update(lc_sum.item(),img_x.shape[0])
            loss_meter_x.update(lorthx_sum.item(), img_x.shape[0])
            loss_meter_y.update(lorthy_sum.item(),img_x.shape[0])

            train_loader_iter.set_postfix({
                'li': f"{loss_meter_i.avg:.3f}",
                'lt': f"{loss_meter_t.avg:.3f}",
                'lc': f"{loss_meter_c.avg:.3f}",
                'ac': f"{acc_meter.avg*100:.2f}",
                'acx': f"{acc_meter_x.avg*100:.2f}",
                'acy': f"{acc_meter_y.avg*100:.2f}",
            })
            #print('Im here')
            torch.cuda.synchronize()
            #if (n_iter + 1) % log_period == 0:
                #logger.info("Epoch[{}] Iteration[{}/{}] LossT: {:.3f}, Lossc: {: .3f}, Lossx: {: .3f}, Lossy: {: .3f}\n Accc: {:.3f}, Accx: {:.3f}, Accy: {:.3f}, Base Lr: {:.2e}"
                            #.format(epoch, (n_iter + 1), len(train_loader_pair),
                                    #loss_meter.avg, loss_meter_c.avg, loss_meter_x.avg, loss_meter_y.avg, acc_meter.avg, acc_meter_x.avg, acc_meter_y.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch[{}] Iteration[{}/{}] Lossi: {:.3f}, Losst: {:.3f}, Lossc: {: .3f}, Lossx: {: .3f}, Lossy: {: .3f}\n Accc: {:.3f}, Accx: {:.3f}, Accy: {:.3f},Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader_pair),
                                loss_meter_i.avg, loss_meter_t.avg, loss_meter_c.avg, loss_meter_x.avg, loss_meter_y.avg, acc_meter.avg, acc_meter_x.avg, acc_meter_y.avg, scheduler._get_lr(epoch)[0]))
            #logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    #.format(epoch, time_per_batch, (train_loader_pair.batch_size)*2 / time_per_batch))

        if epoch >190 and epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    #跨模态重识别===================================================================
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(x=img, y=None, label=camids, view_label=target_view, mode='test-xy')
                            evaluator.update((feat, vid, camid))
                    cmc_cross, mAP_cross, _, _, _, _, _ = evaluator.compute()
                    logger.info("xy-Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP_cross: {:.1%}".format(mAP_cross))
                    for r in [1, 5, 10]:
                        logger.info("CMC_cross curve, Rank-{:<3}:{:.1%}".format(r, cmc_cross[r - 1]))
                    torch.cuda.empty_cache()
                    #单模态x重识别===================================================================
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader_x):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(x=img, y=None, label=camids, view_label=target_view, mode="test-x")
                            evaluator_x.update((feat, vid, camid))
                    cmc_x, mAP_x,  _, _, _, _, _ = evaluator_x.compute()
                    logger.info("x-Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP_x: {:.1%}".format(mAP_x))
                    for r in [1, 5, 10]: 
                        logger.info("CMC_x curve, Rank-{:<3}:{:.1%}".format(r, cmc_x[r - 1]))
                    torch.cuda.empty_cache()
                    #单模态y重识别===================================================================
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader_y):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(x=None, y=img, label=camids, view_label=target_view, mode='test-y')
                            evaluator_y.update((feat, vid, camid))
                    cmc_y, mAP_y, _, _, _, _, _ = evaluator_y.compute()
                    logger.info("y-Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP_y: {:.1%}".format(mAP_y))
                    for r in [1, 5, 10]:
                        logger.info("CMC_y curve, Rank-{:<3}:{:.1%}".format(r, cmc_y[r - 1]))
                    torch.cuda.empty_cache()

            else:
                model.eval()
                #跨模态重识别===================================================================
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(x=img, y=None, label=camids, mode='test-xy')#######这有问题！！
                        evaluator.update((feat, vid, camid))
                cmc_cross, mAP_cross, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP_cross: {:.1%}".format(mAP_cross))
                for r in [1, 2, 3, 5, 10]:
                    logger.info("CMC_cross curve, Rank-{:<3}:{:.1%}".format(r, cmc_cross[r - 1]))
                torch.cuda.empty_cache()
                #单模态x重识别===================================================================
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader_x):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(x=img, y=None, label=camids, mode="test-x")
                        evaluator_x.update((feat, vid, camid))
                cmc_x, mAP_x,  _, _, _, _, _ = evaluator_x.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP_x: {:.1%}".format(mAP_x))
                for r in [1, 2, 3, 5, 10]: 
                    logger.info("CMC_x curve, Rank-{:<3}:{:.1%}".format(r, cmc_x[r - 1]))
                torch.cuda.empty_cache()
                #单模态y重识别===================================================================
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader_y):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(x=None, y=img, label=camids, mode='test-y')
                        evaluator_y.update((feat, vid, camid))
                cmc_y, mAP_y, _, _, _, _, _ = evaluator_y.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP_y: {:.1%}".format(mAP_y))
                for r in [1, 2, 3, 5, 10]:
                    logger.info("CMC_y curve, Rank-{:<3}:{:.1%}".format(r, cmc_y[r - 1]))
                torch.cuda.empty_cache()

#此处还没修改，等用到test.py的时候再改吧。现在还需要改一下model里面的测试流程，变成单输入。
def do_inference(cfg,
                 model,
                 val_loader,
                 val_loader_x,
                 val_loader_y,
                 num_query,
                 num_query_x,
                 num_query_y):
    device = "cuda"
    logger = logging.getLogger("shipreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='cross')
    evaluator_x = R1_mAP_eval(num_query_x, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='x')
    evaluator_y = R1_mAP_eval(num_query_y, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='y')

    evaluator.reset()
    evaluator_x.reset()
    evaluator_y.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list_cross = []
    img_path_list_x = []
    img_path_list_y = []
    all_feats = []
    all_labels = []
    all_cams = []
    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(x=img, y=None, label=camids, mode='test-xy')
            evaluator.update((feat, vid, camid))
            img_path_list_cross.extend(imgpath)
            #print(type(vid), vid)
            all_feats.append(feat.cpu().numpy())
            all_labels.append(np.array(vid))
            all_cams.append(np.array(camid))
    cmc_cross, mAP_cross, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP_cross: {:.1%}".format(mAP_cross))
    for r in [1, 2, 3, 5, 10]:
        logger.info("CMC_cross curve, Rank-{:<3}:{:.1%}".format(r, cmc_cross[r - 1]))
    all_feats = np.vstack(all_feats)
    all_labels = np.concatenate(all_labels)
    all_cams = np.concatenate(all_cams)
    save_path = os.path.join(cfg.OUTPUT_DIR, "tsne_epoch.png")
    plot_tsne_with_cam(all_feats, all_labels, all_cams, save_path)

    
    #单模态x重识别===================================================================
    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader_x):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(x=img, y=None, label=camids, mode="test-x")
            evaluator_x.update((feat, vid, camid))
            img_path_list_x.extend(imgpath)
    cmc_x, mAP_x,  _, _, _, _, _ = evaluator_x.compute()
    logger.info("Validation Results")
    logger.info("mAP_x: {:.1%}".format(mAP_x))
    for r in [1, 2, 3, 5, 10]: 
        logger.info("CMC_x curve, Rank-{:<3}:{:.1%}".format(r, cmc_x[r - 1]))
    #单模态y重识别===================================================================
    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader_y):
        with torch.no_grad():
            #print('y', vid, camid)
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(x=None, y=img, label=camids, mode='test-y')
            evaluator_y.update((feat, vid, camid))
            img_path_list_y.extend(imgpath)
    cmc_y, mAP_y, _, _, _, _, _ = evaluator_y.compute()
    logger.info("Validation Results")
    logger.info("mAP_y: {:.1%}".format(mAP_y))
    for r in [1, 2, 3, 5, 10]:
        logger.info("CMC_y curve, Rank-{:<3}:{:.1%}".format(r, cmc_y[r - 1]))

    return cmc_cross[0], cmc_cross[4]
"""
def do_inference(cfg,
                 model,
                 val_loader,
                 val_loader_x,
                 val_loader_y,
                 num_query,
                 num_query_x,
                 num_query_y):
    device = "cuda"
    logger = logging.getLogger("shipreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM, mode='x')

    evaluator.reset()


    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list_cross = []

    for n_iter, (img, vid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(x=img, y=None, label=camids, mode='test-xy')#######这有问题！！
            evaluator.update((feat, vid, camid))
            img_path_list_cross.extend(imgpath)
    cmc_cross, mAP_cross, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results")
    logger.info("mAP_cross: {:.1%}".format(mAP_cross))
    for r in [1, 5, 10]:
        logger.info("CMC_cross curve, Rank-{:<3}:{:.1%}".format(r, cmc_cross[r - 1]))
    return cmc_cross[0], cmc_cross[4]"""