from level0_models.rf_model import RFLevel0
from level0_models.xgb_model import XGBLevel0
from level0_models.lgbm_model import LGBMLevel0
from level0_models.svr_model import SVRLevel0
from level0_models.mlp_model import MLPLevel0


def get_level0_models(random_state=42):
    """
    Return all Level-0 base learners used in the stacking framework.

    Each model follows the same interface:
    - fit()
    - predict_proba()
    - predict()
    """
    models = {
        "RF": RFLevel0(random_state=random_state),
        "XGB": XGBLevel0(random_state=random_state),
        "LGBM": LGBMLevel0(random_state=random_state),
        "SVR": SVRLevel0(),
        "MLP": MLPLevel0(random_state=random_state),
    }

    return models
