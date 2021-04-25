def get_scheduler(CFG,optimizer):
    if CFG.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
    elif CFG.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler == 'CyclicLR':
        scheduler = CyclicLR(optimizer, base_lr=CFG.base_lr, max_lr=CFG.max_lr, step_size_up=CFG.step_size, step_size_down=CFG.step_size_down,mode=CFG.scheduler_mode)
    elif CFG.scheduler == 'ExponetialLR':
        scheduler = ExponentialLR(optimizer, gamma=CFG.gamma)
    elif CFG.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, step_size=CFG.step_size, gamma=CFG.gamma)
    elif CFG.scheduler == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=CFG.milestones, gamma=CFG.gamma)
    return scheduler
