import os
import glob
import nibabel as nib
from base_code.defaults import get_cfg_defaults
from base_code.models.build import build_model

if __name__ == "__main__":
    model_path = ''
    config_path = ''
    out_path = ''
    device = 'cuda'
    gpu = '0'

    start_slice = 1
    end_slice = 254
    use_amp = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    import torch

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    # build model
    model_arch = cfg.MODEL.ARCHITECTURE
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # prepare data
    img_list = sorted(
        glob.glob('../test_ct_image_labels/image_ct_*.nii.gz'))

    for idx, f in enumerate(img_list):
        img_data = torch.from_numpy(nib.load(f).get_fdata()).float()
        img_data = img_data.flip(dims=[0, 1])
        w, h, d = img_data.shape
        pred_volume = torch.zeros([1, w, h, end_slice - start_slice + 1])

        for i, slice_idx in enumerate(torch.arange(start_slice, end_slice + 1)):
            images = img_data[:, :, slice_idx - 1:slice_idx + 2].permute(2, 0, 1).unsqueeze(0).to(device)

            if cfg.DATA.MEAN_CENTER:
                images -= torch.mean(images)
            if cfg.DATA.STANDARDIZE:
                images /= torch.std(images)

            with torch.cuda.amp.autocast(enabled=use_amp):
                with torch.no_grad():
                    if model_arch in ['UNet2d']:
                        pred = model(images)
                    elif model_arch in ['EfficientUNet2d']:
                        pred = model(images)
                    pred = pred.argmax(dim=1)

                    pred_volume[0, :, :, i] = pred[0].cpu()

        torch.save(pred_volume.to(torch.int8), out_path.format(idx))
