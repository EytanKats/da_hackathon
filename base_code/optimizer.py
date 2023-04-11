import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR, StepLR, LambdaLR


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == 'adam':
        optimizer = Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise NotImplementedError()

    if cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = MultiStepLR(optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.LR_DECAY_FACTOR)
    elif cfg.SOLVER.SCHEDULER == 'step':
        lr_scheduler = StepLR(optimizer, cfg.SOLVER.LR_DECAY_STEPS, cfg.SOLVER.LR_DECAY_FACTOR)
    else:
        raise NotImplementedError()

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.SOLVER.USE_AMP)

    return optimizer, lr_scheduler, scaler
