import torch


def dice_coeff(outputs, labels, num_classes, smooth=1e-7, eps=0.):
    dice = torch.FloatTensor(num_classes - 1).fill_(0)
    for label_num in range(1, num_classes):
        iflat = (outputs == label_num).view(-1).float()
        tflat = (labels == label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num - 1] = (2. * intersection + smooth) / (torch.mean(iflat) + torch.mean(tflat) + smooth + eps)

    return dice


def validate(loader, model, model_arch, device, use_amp, num_classes):
    dice_scores = []
    for it, data in enumerate(loader, 1):
        images, targets, targets_oneHot = data[0].to(device), data[1].to(device), data[2].to(device)
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.no_grad():
                if model_arch in ['UNet2d']:
                    pred = model(images)
                elif model_arch in ['EfficientUNet2d']:
                    pred = model(images)
                pred = pred.argmax(dim=1)

                for p, g in zip(pred, targets):
                    dice_scores.append(dice_coeff(p, g, num_classes))

    dice_scores = torch.stack(dice_scores, dim=0)
    return dice_scores.mean()


