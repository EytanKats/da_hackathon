import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MMWHS2dDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, name, mode):
        root = '/data_supergrover3/kats/data/mmwhs/hackathon_mmwhs/processed'
        if 'ct_train' in name:
            data_folder = 'train_ct'
        elif 'ct_val' in name:
            data_folder = 'val_ct'
        elif 'mr_train' in name:
            data_folder = 'train_mr'
        elif 'mr_val' in name:
            data_folder = 'val_mr'
        else:
            raise NotImplementedError()
        self.data_split_name = name
        self.image_list = sorted(glob.glob(os.path.join(root, data_folder, '*_data.pth')))

        self.is_train = True if mode == 'train' else False
        self.mean_center = cfg.DATA.MEAN_CENTER
        self.standardize = cfg.DATA.STANDARDIZE

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = img_path.replace('_data.pth', '_label.pth')

        img = torch.load(img_path)
        if self.mean_center:
            img -= torch.mean(img)
        if self.standardize:
            img /= torch.std(img)
        img = img.permute(2, 0, 1)

        if self.data_split_name == 'ct_train':
            label = 0
            label_oneHot = 0
        else:
            label = torch.load(label_path)[:, :, 0].long()
            label_oneHot = F.one_hot(label, 5).permute(2, 0, 1).float()

        return img, label, label_oneHot, idx

    def __len__(self):
        return len(self.image_list)


def build_dataloader(cfg):
    # build source datasets for training
    if cfg.ORACLE:
        train_set_source = MMWHS2dDataset(cfg, name='ct_train', mode='train')
    else:
        train_set_source = MMWHS2dDataset(cfg, name='mr_train', mode='train')
    train_loader_source = DataLoader(train_set_source, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                     num_workers=cfg.DATA.NUM_WORKERS, shuffle=True, drop_last=True)

    # build target datasets for training
    train_set_target = MMWHS2dDataset(cfg, name='ct_train', mode='train')
    train_loader_target = DataLoader(train_set_target, batch_size=cfg.SOLVER.BATCH_SIZE_TRAIN,
                                     num_workers=cfg.DATA.NUM_WORKERS, shuffle=True, drop_last=True)

    # build datasets for validation
    val_sets = []
    for dataset_name in ['mr_val', 'ct_val']:
        dataset = MMWHS2dDataset(cfg, name=dataset_name, mode='val')
        val_sets.append(dataset)

    val_loaders = [DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE_TEST, num_workers=cfg.DATA.NUM_WORKERS,
                              shuffle=False, drop_last=False) for val_set in val_sets]

    return train_loader_source, train_loader_target, val_loaders
