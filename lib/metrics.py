import numpy as np

def MAE_np(pred, true, mask_value=None):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    MAE = np.mean(np.absolute(pred-true))
    return MAE

def RMSE_np(pred, true, mask_value=0.0):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    RMSE = np.sqrt(np.mean(np.square(pred-true)))
    return RMSE

def MAPE_np(pred, true, mask_value=0.0):
    if mask_value != None:
        mask = np.where(true > (mask_value), True, False)
        true = true[mask]
        pred = pred[mask]
    return np.mean(np.absolute(np.divide((true - pred), (true)))) * 100

def RMSE_MAE_MAPE(y_true, y_pred):
    return (

        RMSE_np(y_pred, y_true),
        MAE_np(y_pred, y_true),
        MAPE_np(y_pred, y_true),
    )
