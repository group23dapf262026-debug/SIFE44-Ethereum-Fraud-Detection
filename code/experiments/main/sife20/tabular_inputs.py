"""Explicit portable numeric transform and stable named column schema for tabular arms."""
import numpy as np
import pandas as pd
from .features import SIFE44_NAMES,PURE36_INDICES,select_inputs
from .utilities import require

SOURCE14_NAMES=('bal_c','V_inv_c','V_return_c','N_inv_c','N_return_c',
    'V_meaninv_c','V_meanreturn_c','bal_t','V_inv_t','V_return_t','N_inv_t','N_return_t','V_meaninv_t','V_meanreturn_t')
LR_PROFILES=('signed_log1p','raw_reference')


def tabular_column_names(arm):
    """Preserve exact source/extra order; reject mismatched prediction schemas."""
    extra=tuple(SIFE44_NAMES[index] for index in PURE36_INDICES) if arm in ('Pure36','X14_Pure36') else tuple(SIFE44_NAMES)
    if arm=='X14':return SOURCE14_NAMES
    return extra if arm=='Pure36' else SOURCE14_NAMES+extra


def tabular_input_frame(source14,sife44,arm,model_name,lr_profile='signed_log1p'):
    """LR signed-log touches only source14; named SIFE/Pure columns remain numerically unchanged."""
    require(lr_profile in LR_PROFILES,'Unknown fixed LR preprocessing profile')
    values=select_inputs(None if arm=='Pure36' else source14,sife44,arm,model_name).astype(np.float64,copy=True)
    if model_name=='lr' and arm!='Pure36' and lr_profile=='signed_log1p':
        source=values[:,:14]
        values[:,:14]=np.sign(source)*np.log1p(np.abs(source))
    require(np.isfinite(values).all(),'Nonfinite tabular input after declared preprocessing')
    columns=tabular_column_names(arm)
    require(len(columns)==values.shape[1] and len(set(columns))==len(columns),'Invalid input schema')
    return pd.DataFrame(values,columns=columns)


def check_estimator_columns(estimator,frame):
    """Require the prediction frame's named columns to match the trained estimator in order."""
    if hasattr(estimator,'feature_names_in_'):
        require(tuple(estimator.feature_names_in_)==tuple(frame.columns),'Prediction feature names/order differ from TRAIN')
