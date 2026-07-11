import time
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder

def evaluar_modelo(modelo, X, y, cv, scorers, nombre_modelo, nombre_features):
    inicio= time.time()
    cv_results = cross_validate(modelo, X, y, cv=cv, scoring=scorers, return_train_score=False)
    fin = time.time()

    return {
        'Modelo'    : nombre_modelo,
        'Features'  : nombre_features,
        'RMSE_medio': round(-cv_results['test_RMSE'].mean(), 4),
        'RMSE_std'  : round(cv_results['test_RMSE'].std(), 4),
        'MAE_medio' : round(-cv_results['test_MAE'].mean(), 4),
        'MAE_std'   : round(cv_results['test_MAE'].std(), 4),
        'R2_medio'  : round(cv_results['test_R2'].mean(), 4),
        'R2_std'    : round(cv_results['test_R2'].std(), 4),
        'Tiempo_seg': round(fin - inicio, 2)
    }

def encodear_ordinal(train_df, test_df, features, features_categoricas):
    cat_feats = [f for f in features if f in features_categoricas]
    num_feats = [f for f in features if f not in features_categoricas]

    tr = train_df[features].copy()
    te = test_df[features].copy()

    if cat_feats:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        tr[cat_feats] = enc.fit_transform(tr[cat_feats])
        te[cat_feats] = enc.transform(te[cat_feats])

    return tr, te
