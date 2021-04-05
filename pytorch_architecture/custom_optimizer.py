def get_optimizer(CFG,model):
    if CFG.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=CFG.lr)
    elif CFG.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay = CFG.weight_decay)
    elif CFG.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=CFG.lr, momentum = CFG.momentum)
    return optimizer