import sys
sys.path.append('../')


import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from base_code.defaults import get_cfg_defaults
from base_code.dataset import build_dataloader
from base_code.models.build import build_model
from base_code.optimizer import build_optimizer
from base_code.validate import validate

def train(cfg, output_directory):
    # parameters
    device = cfg.MODEL.DEVICE
    num_epochs = cfg.SOLVER.NUM_EPOCHS

    # prepare datasets
    train_loader_src, train_loader_tgt, val_loader = build_dataloader(cfg)
    train_loader_tgt_iter = iter(train_loader_tgt)

    # build model
    model = build_model(cfg).to(device)
    model_arch = cfg.MODEL.ARCHITECTURE
    if cfg.MODEL.WEIGHT != '':
        model.load_state_dict(torch.load(cfg.MODEL.WEIGHT), strict=False)

    # optimizer and scheduler
    use_amp = cfg.SOLVER.USE_AMP
    optimizer, scheduler, scaler = build_optimizer(cfg, model)

    # create loss function
    criterion = torch.nn.CrossEntropyLoss()

    # logging
    checkpoint_interval = cfg.MODEL.CHECKPOINT_INTERVAL
    log_array = np.zeros([num_epochs, 1 + len(val_loader)])

    # training
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        losses_src = []
        for it, data_src in enumerate(tqdm(train_loader_src), 1):
            images_src, targets_src, targets_oneHot_src = data_src[0].to(device), data_src[1].to(device), data_src[2].to(device)

            # try:
            #     data_tgt = next(train_loader_tgt_iter)
            # except StopIteration:
            #     train_loader_tgt_iter = iter(train_loader_tgt)
            #     data_tgt = next(train_loader_tgt_iter)

            with torch.cuda.amp.autocast(enabled=use_amp):
                if model_arch in ['UNet2d']:
                    pred = model(images_src)
                elif model_arch in ['EfficientUNet2d']:
                    pred = model(images_src)
                else:
                    raise NotImplementedError()

                loss_src = criterion(pred, targets_src)
                losses_src.append(loss_src.item())

                loss = loss_src

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        scheduler.step()

        # Validation
        model.eval()
        val_dsc = []
        for loader in val_loader:
            dsc = validate(loader, model, model_arch, device, use_amp, cfg.DATA.OUT_CHANNELS)
            val_dsc.append(dsc.item())
        torch.save(model.state_dict(), os.path.join(output_directory, 'model.pth'))
        end_time = time.time()
        print('epoch', epoch + 1, 'validation time', '%0.3f' % ((end_time - start_time) / 60.), 'loss_src',
              np.mean(losses_src), 'val DSC', val_dsc)
        log_array[epoch, :] = np.array([np.mean(losses_src), *val_dsc])
        np.save(os.path.join(output_directory, 'log.npy'), log_array)

        if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), os.path.join(output_directory, 'model_ep{:03d}.pth'.format(epoch + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="/data_supergrover3/kats/projects/da_hackathon/base_code/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    output_directory = os.path.join(cfg.BASE_DIRECTORY, cfg.EXPERIMENT_NAME)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    train(cfg, output_directory)
