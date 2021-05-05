import pandas as pd
import numpy as np

def table_train(CFG, train, custommodel, dataset_preprocess_train, dataset_postprocess_train, get_result, MODEL_NAME, LOGGER):
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            # ====================================================
            # Dataset Preprocess
            # ====================================================
            x_train, y_train, x_valid, y_valid = dataset_preprocess_train(CFG,train,fold)

            # ====================================================
            # train & valid
            # ====================================================
            LOGGER.info(f"========== fold: {fold} training ==========")

            model = custommodel.train(CFG,x_train, y_train, x_valid, y_valid)
            predictions = custommodel.valid(CFG,x_valid,model)
            
            # ====================================================
            # save
            # ====================================================
            pickle.dump(model, open(CFG.MAIN_PATH + f'{MODEL_NAME}' + f'_fold_{fold}.sav','wb'))

            _oof_df = dataset_postprocess_train(CFG, predictions, train, fold)

            oof_df = pd.concat([oof_df, _oof_df])
            score = get_result(_oof_df)
            LOGGER.info(f"========== fold: {fold} result:{score}==========")

    # ====================================================
    # CV result
    # ====================================================  
    score = get_result(oof_df)
    LOGGER.info(f"========== CVresult:{score}==========")
    oof_df.to_csv(CFG.MAIN_PATH+f'oof_df.csv', index=False)
    return oof_df
 
def table_inference(CFG, test, custommodel, dataset_preprocess_test, dataset_postprocess_test, MODEL_NAME, LOGGER):
    LOGGER.info(f"========== inference ==========")
    predictions = []
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            LOGGER.info(f"========== fold: {fold} inference ==========")
            # ====================================================
            # Dataset Preprocess
            # ====================================================
            x_test = dataset_preprocess_test(CFG, test)

            # ====================================================
            # model loading
            # ====================================================
            model = pickle.load(open(CFG.MAIN_PATH + f'{MODEL_NAME}' + f'_fold_{fold}.sav','rb'))

            # ====================================================
            # inference
            # ====================================================
            y_preds = custommodel.inference(CFG, x_test, model)
            predictions.append(y_preds)

    # ====================================================
    # Averaging
    # ====================================================
    predictions = np.mean(predictions, axis=0)
   
    # ====================================================
    # Submission
    # ====================================================        
    submission = dataset_postprocess_test(CFG, predictions, test)
    submission.to_csv(CFG.MAIN_PATH+f'submission.csv', index=False)
    return submission
