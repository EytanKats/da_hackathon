import os
import glob
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import DataLoader

from proda.data.randaugment import RandAugmentMC

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

        self.augmentations = None
        self.randaug = RandAugmentMC(2, 10)


    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label_path = img_path.replace('_data.pth', '_label.pth')

        img = torch.load(img_path)

        # if self.mean_center:
        #     img -= torch.mean(img)
        # if self.standardize:
        #     img /= torch.std(img)
        # img = img.permute(2, 0, 1)

        img = img - torch.min(img)
        img = img / torch.max(img) * 255

        if self.data_split_name == 'ct_train':
            label = np.zeros(shape=(256, 256))
            label_oneHot = 0
            lpsoft = np.load(os.path.join('/data_supergrover3/kats/experiments/da_hackathon/mmwhs/proda/deep_try/pseudo/', os.path.basename(img_path).replace('.pth', '.npy')))
        else:
            label = torch.load(label_path)[:, :, 0].long()
            label_oneHot = F.one_hot(label, 5).permute(2, 0, 1).float()
            label = label.numpy().astype(np.uint8)

        img = img.numpy().astype(np.uint8)
        img_full = np.copy(img)
        if self.data_split_name == 'mr_train':
            img, label, _, _, weak_params = self.augmentations(img, label)
            img_strong, strong_params = self.randaug(Image.fromarray(img))
            img_strong, _ = self.transform(img_strong, label)
            lpsoft = []
        elif self.data_split_name == 'ct_train':
            img, label, _, lpsoft, weak_params = self.augmentations(img, label, None, lpsoft)
            img_strong, strong_params = self.randaug(Image.fromarray(img))
            img_strong, _ = self.transform(img_strong, label)
        else:
            strong_params = {}
            weak_params = {}
            img_strong = []
            lpsoft = []

        img_full, _ = self.transform(img_full, label)

        img, lbl = self.transform(img, label)
        return img, lbl, label_oneHot, idx, img_path, weak_params, img_strong, strong_params, img_full, lpsoft

    def __len__(self):
        return len(self.image_list)

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        img = np.array(img)
        # img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= np.mean(img)
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        # classes = np.unique(lbl)
        # lbl = np.array(lbl)
        # lbl = lbl.astype(float)
        # # lbl = m.imresize(lbl, self.img_size, "nearest", mode='F')
        # lbl = lbl.astype(int)
        #
        # if not np.all(classes == np.unique(lbl)):
        #     print("WARN: resizing labels yielded fewer classes")  # TODO: compare the original and processed ones
        #
        # if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
        #     print("after det", classes, np.unique(lbl))
        #     raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

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
