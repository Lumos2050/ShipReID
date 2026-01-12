import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset, ImageDatasetPair
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler, PairedRandomIdentitySampler
from .shipreid import Shipreid, Shipreid1
from .sampler_ddp import RandomIdentitySampler_DDP, PairedRandomIdentitySampler_DDP
import torch.distributed as dist

__factory = {
    'shipreid' : Shipreid,
    'shipreid1' : Shipreid1,
}
"""
Dataset   --> 通过索引[index] 取出样本
   ↓
Sampler   --> 给出 index 顺序，比如 [12, 8, 25, 7, ...]
   ↓
DataLoader   --> 一次从 Sampler 拿 batch_size 个 index，然后调用 Dataset 取出样本
   ↓
train_collate_fn(batch) --> 对一个 batch 的原始数据进行打包、对齐、转换等
"""
#collate_fn 的作用是自定义一个 batch 中样本打包的方式，特别适用于混合类型（Tensor、list、str 等）或者需要多种格式的情形。
#DataLoader 的默认行为是用内置的 default_collate，它会自动将每个样本的字段打包成一个 batch，比如把 N 个图像 [C, H, W] 堆成 [N, C, H, W] 的 Tensor，把 int 拼成一个 Tensor 等
def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)#理解zip是关键！！------------------Here begin------------------
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)#所以有个地方不能判定为0吧？？
    return torch.stack(imgs, dim=0), pids, camids, viewids

def train_pair_collate_fn(batch):
    # batch 是一个 list，里面每个元素是 ((img1, pid1, camid1, trackid1), (img2, pid2, camid2, trackid2))
    imgs1, pids1, camids1, trackids1 = [], [], [], []
    imgs2, pids2, camids2, trackids2 = [], [], [], []
    #flag=0
    for (img1, pid1, camid1, trackid1, _), (img2, pid2, camid2, trackid2, _) in batch:
        #flag = flag + 1
        #print('flag',flag)
        imgs1.append(img1)
        pids1.append(pid1)
        camids1.append(camid1)
        trackids1.append(trackid1)

        imgs2.append(img2)
        pids2.append(pid2)
        camids2.append(camid2)
        trackids2.append(trackid2)

    # 转为tensor
    imgs1 = torch.stack(imgs1, dim=0)
    imgs2 = torch.stack(imgs2, dim=0)
    pids = torch.tensor(pids1, dtype=torch.int64)  # pid1 和 pid2 是一样的，可以选一个
    camids = torch.tensor(camids1, dtype=torch.int64)
    trackids = torch.tensor(trackids1, dtype=torch.int64)

    return imgs1, imgs2, pids, camids, trackids

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomRotation(
                degrees=180,  # 表示从 [-180, 180] 范围内随机选择任意角度
                interpolation=T.InterpolationMode.BILINEAR,  # 旋转插值方式
                expand=False,  # 是否扩大画布防止裁切
                fill=0         # 旋转产生空白区域用0填充(黑色)，可改成tuple实现三通道不同颜色
            ),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    #print(len(dataset.train_x), len(dataset.train_y))
    train_set_pair = ImageDatasetPair(dataset.train_x, dataset.train_y, train_transforms)

    #print('trainloader长度', len(train_set_pair))
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:#分布式训练
            print('DIST_TRAIN START')
            mini_batch_size = int(cfg.SOLVER.IMS_PER_BATCH / 2) // dist.get_world_size()#每个进程中多少个样本
            data_sampler = PairedRandomIdentitySampler_DDP(train_set_pair, int(cfg.SOLVER.IMS_PER_BATCH / 2), cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_pair = torch.utils.data.DataLoader(
                train_set_pair,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_pair_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_pair = DataLoader(
                train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
                sampler=PairedRandomIdentitySampler(train_set_pair, int(cfg.SOLVER.IMS_PER_BATCH / 2), cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_pair_collate_fn
            )
        print('trainpairloader长度', len(train_loader_pair))
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_pair = DataLoader(
            train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=num_workers,
            collate_fn=train_pair_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    val_set_x = ImageDataset(dataset.query_x + dataset.gallery_x, val_transforms)
    val_loader_x = DataLoader(
        val_set_x, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )#img, pid, camid, trackid, img_path.split('/')[-1]

    val_set_y = ImageDataset(dataset.query_y + dataset.gallery_y, val_transforms)
    val_loader_y = DataLoader(
        val_set_y, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )#img, pid, camid, trackid, img_path.split('/')[-1]

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )#img, pid, camid, trackid, img_path.split('/')[-1]


    return train_loader_pair, val_loader, val_loader_x, val_loader_y, len(dataset.query), len(dataset.query_x), len(dataset.query_y), num_classes, cam_num, view_num

"""def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.RandomRotation(
                degrees=180,  # 表示从 [-180, 180] 范围内随机选择任意角度
                interpolation=T.InterpolationMode.BILINEAR,  # 旋转插值方式
                expand=False,  # 是否扩大画布防止裁切
                fill=0         # 旋转产生空白区域用0填充(黑色)，可改成tuple实现三通道不同颜色
            ),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            #RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)

    train_set_pair = ImageDatasetPair(dataset.train_x, dataset.train_y, train_transforms)

    #print('trainloder长度', len(train_set_pair))
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:#分布式训练
            print('DIST_TRAIN START')
            mini_batch_size = int(cfg.SOLVER.IMS_PER_BATCH / 2) // dist.get_world_size()#每个进程中多少个样本
            data_sampler = PairedRandomIdentitySampler_DDP(train_set_pair, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE)
            batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
            train_loader_pair = torch.utils.data.DataLoader(
                train_set_pair,
                num_workers=num_workers,
                batch_sampler=batch_sampler,
                collate_fn=train_pair_collate_fn,
                pin_memory=True,
            )
        else:
            train_loader_pair = DataLoader(
                train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2),
                sampler=PairedRandomIdentitySampler(train_set_pair, int(cfg.SOLVER.IMS_PER_BATCH / 2), cfg.DATALOADER.NUM_INSTANCE),
                num_workers=num_workers, collate_fn=train_pair_collate_fn
            )

    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_pair = DataLoader(
            train_set_pair, batch_size=int(cfg.SOLVER.IMS_PER_BATCH / 2), shuffle=True, num_workers=num_workers,
            collate_fn=train_pair_collate_fn
        )
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))
    val_set_x = ImageDataset(dataset.query_x, val_transforms)
    val_loader_x = DataLoader(
        val_set_x, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )#img, pid, camid, trackid, img_path.split('/')[-1]

    val_set_y = ImageDataset(dataset.query_y, val_transforms)
    val_loader_y = DataLoader(
        val_set_y, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )#img, pid, camid, trackid, img_path.split('/')[-1]

    #val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    #val_loader = DataLoader(
        #val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        #collate_fn=val_collate_fn
    #)#img, pid, camid, trackid, img_path.split('/')[-1]

    val_train_set = ImageDataset(dataset.query, val_transforms)
    val_loader = DataLoader(
        val_train_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    print(int(len(dataset.query)/4), int(len(dataset.query_x)/4), int(len(dataset.query_y)/4))
    return train_loader_pair, val_loader, val_loader_x, val_loader_y, int(len(dataset.query)/4), int(len(dataset.query_x)/4), int(len(dataset.query_y)/4), num_classes, cam_num, view_num"""