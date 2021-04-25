def get_criterion(CFG):
    if CFG.criterion == 'MSELoss':
        criterion =  nn.MSELoss()
    elif CFG.criterion == 'L1Loss':
        criterion =  nn.L1Loss()
    elif CFG.criterion == 'BCELoss':
        criterion =  nn.BCELoss()
    elif CFG.criterion == 'BCEWithLogitsLoss':
        criterion =  nn.BCEWithLogitsLoss()
    elif CFG.criterion == 'CrossEntropyLoss':
        criterion =  nn.CrossEntropyLoss()

    return criterion
