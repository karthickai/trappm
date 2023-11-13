import os
import torch
from torch_geometric.data import Dataset, DataLoader
import random
from torch.utils.data import SubsetRandomSampler
from utils import set_seed

categories = ["densenet", "efficientnet", "vgg", "swin", "visformer", "resnet", 
              "poolformer", "mobilenet", "mnasnet", "convnext", "vit"]

class PPMDataset(Dataset):
    def __init__(self, root, files=None, transform=None, pre_transform=None, pre_filter=None, unsup=False):

        self.data_dir = root
        self.unsup = unsup
        self.categories = categories
        if files is None:
            self.files = []
            with os.scandir(root) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.endswith('.pt'):
                        for category in self.categories:
                            if category in entry.name:
                                self.files.append(entry.name)
            self.files = random.sample(self.files, len(self.files))
        else:
            self.files = files
        self.total_files = self.files
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        
        
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return self.total_files

    @property
    def num_classes(self) -> int:
        return 1

    def process(self):
        pass

    def len(self):
        return len(self.processed_file_names) -1

    def get(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.files[idx]))
        if not self.unsup:
            selected_indices = [0, 5, 6, 7, 8, 9]
            data.static = data.static[selected_indices]
            mean = data.static.mean()
            std = data.static.std()
            data.static = (data.static - mean) / std 
        return data
    
    def train_test_splits(self, base_seed, ratio=0.7, num_folds=5):
        category_files = {category: [] for category in self.categories}
        
        for file in self.files:
            for category in self.categories:
                if category in file:
                    category_files[category].append(file)
                    break
        folds = []
        for fold_idx in range(num_folds):
            train_files, test_files = [], []
            set_seed(base_seed + fold_idx)
            for files in category_files.values():
                random.shuffle(files)
                split_idx = int(len(files) * ratio)
                train_files.extend(files[:split_idx])
                test_files.extend(files[split_idx:])
            folds.append((train_files, test_files))
        return folds


def get_dataset(root_folder, batch_size=1, base_seed=1337, ratio=0.7, num_folds=5):
    dataset = PPMDataset(root_folder)
    folds = dataset.train_test_splits(ratio=ratio, base_seed=base_seed, num_folds=num_folds)
    k_fold_datasets = []
    for train_files, test_files in folds:
        train_dataset = PPMDataset(root_folder, files=train_files)
        train_sampler = SubsetRandomSampler(range(len(train_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        test_datasets = []

        for category in categories:
            category_files = [file for file in test_files if category in file]
            if category_files:
                category_dataset = DataLoader(PPMDataset(root_folder, files=category_files), batch_size=batch_size, shuffle=False)
                test_datasets.append({'category': category, 'dataset': category_dataset})
        k_fold_datasets.append({'train': train_loader, 'test': test_datasets})
    return k_fold_datasets

class Y:
    Train_Gpu_Memory_Mb = 0
    Train_Gpu_Power_W = 1
    Train_Gpu_Utilsation = 2
    Train_Mem_Utilsation = 3
    Train_Step_Time = 4
    Inference_Gpu_Memory_Mb = 5
    Inference_Gpu_Power_W = 6
    Inference_Gpu_Utilsation = 7
    Inference_Mem_Utilsation = 8
    Inference_Step_Time = 9

labels = {
    'Train_Gpu_Memory_Mb': Y.Train_Gpu_Memory_Mb,
    'Train_Gpu_Power_W': Y.Train_Gpu_Power_W,
    'Train_Step_Time': Y.Train_Step_Time,
}