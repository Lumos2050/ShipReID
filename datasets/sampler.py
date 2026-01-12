from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}创建一个字典，key 是 pid，value 是所有该 pid 的图像索引列表
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())#把所有键提出来
        
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


class PairedRandomIdentitySampler(Sampler):
    """
    For each batch:
    - Randomly sample N identities.
    - For each identity, sample K RGB images and K SAR images.
    - Ensure RGB and SAR samples are aligned by label.
    - Output: [N*K RGB samples] + [N*K SAR samples]

    Args:
    - data_source (list): list of (img_path, pid, modality)，其中 modality 为 'rgb' 或 'sar'
    - batch_size (int): must be even; total number of samples (RGB + SAR), e.g., 64 means 32 RGB + 32 SAR.
    - num_instances (int): K, number of pairs per identity.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        #print("First item in data_source:", self.data_source[0])
        self.batch_size = batch_size
        self.num_instances = num_instances
        assert batch_size % num_instances == 0, "batch_size // 2 must be divisible by num_instances"
        
        self.num_pids_per_batch = batch_size // num_instances

        # 构建两个 index_dict，分别存储 RGB 和 SAR 图像按 pid 分类的索引
        self.index_dic = defaultdict(list)
        #self.data_source = [item for pair in self.data_source for item in pair]#解除嵌套[ [[x1],[x2]], [[x3],[x4]], ...]->[[x1],[x2],[x3],...]

        for index, (rgb_tuple, sar_tuple) in enumerate(self.data_source):
            #print("what?", rgb_tuple)
            pid = rgb_tuple[1]
            assert pid == sar_tuple[1], f"数据对中的 pid 不匹配: {rgb_tuple[1]} vs {sar_tuple[1]}"
            self.index_dic[pid].append(index)
        
        self.pids = set(self.index_dic.keys())
        #print("Total pids:", len(self.pids))
        # 计算总长度（按 pair 考虑）
        self.length = 0
        for pid in self.pids:
            num_pairs = len(self.index_dic[pid])
            if num_pairs < self.num_instances:
                num_pairs = self.num_instances
            self.length += num_pairs - num_pairs % self.num_instances#不被整除的部分丢弃了，这是主流做法
    def __iter__(self):#def __iter__(self): 是你的 Sampler 提供“采样顺序”给 DataLoader 的接口，你控制的是索引排列顺序；DataLoader 拿到这些 index，按你设定的 batch_size 把它们一批批送给模型。
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)
    def __len__(self):
        return self.length