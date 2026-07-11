import time
import pandas as pde
from sklearn.model_selection import cross_validate

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

def encodear_para_rf(train_df, test_df, features, features_categoricas):
    cat_feats = [f for f in features if f in features_categoricas]

    tr_enc = pd.get_dummies(train_df[features], columns=cat_feats, drop_first=True)
    te_enc = pd.get_dummies(test_df[features], columns=cat_feats, drop_first=True)
    te_enc = te_enc.reindex(columns=tr_enc.columns, fill_value=0)

    return tr_enc, te_enc, cat_feats