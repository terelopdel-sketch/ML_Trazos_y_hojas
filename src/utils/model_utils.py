import time
import pandas as pd
import numpy as np
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

    tr = train_df[features].copy()
    te = test_df[features].copy()

    if cat_feats:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        tr[cat_feats] = enc.fit_transform(tr[cat_feats])
        te[cat_feats] = enc.transform(te[cat_feats])

    return tr, te



def split_temporal_por_fecha(df, columna_fecha='Fecha', n_splits=5):
    """
    Genera índices de train/val para validación temporal sobre un panel
    de datos (varios productos x fechas). Corta por fechas únicas ordenadas,
    de modo que todas las filas de una misma fecha caen en el mismo lado.
    Devuelve una lista de tuplas (train_idx, val_idx) para pasar como cv.
    """
    fechas = np.sort(df[columna_fecha].unique())
    n_fechas = len(fechas)
    tam_fold = n_fechas // (n_splits + 1)

    splits = []
    for i in range(1, n_splits + 1):
        corte_train = tam_fold * i
        corte_val   = tam_fold * (i + 1) if i < n_splits else n_fechas

        fechas_train = fechas[:corte_train]
        fechas_val   = fechas[corte_train:corte_val]

        train_idx = df.index[df[columna_fecha].isin(fechas_train)].to_numpy()
        val_idx   = df.index[df[columna_fecha].isin(fechas_val)].to_numpy()

        splits.append((train_idx, val_idx))

    return splits

def evaluar_modelo_log(modelo, X, y, splits, nombre_modelo, nombre_features):
    """
    Igual que evaluar_modelo pero entrena sobre log1p(y) y revierte con expm1
    antes de calcular las métricas, que se miden en la escala original.
    Hace la validación cruzada manualmente sobre la lista de splits.
    """
    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse_list, mae_list, r2_list = [], [], []
    inicio = time.time()

    for tr_idx, val_idx in splits:
        X_tr, X_val = X.loc[tr_idx], X.loc[val_idx]
        y_tr, y_val = y.loc[tr_idx], y.loc[val_idx]

        m = clone(modelo)
        m.fit(X_tr, np.log1p(y_tr))
        y_pred = np.expm1(m.predict(X_val))
        y_pred = np.clip(y_pred, 0, None)  # no permitir predicciones negativas

        rmse_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        mae_list.append(mean_absolute_error(y_val, y_pred))
        r2_list.append(r2_score(y_val, y_pred))

    fin = time.time()
    return {
        'Modelo'    : nombre_modelo,
        'Features'  : nombre_features,
        'RMSE_medio': round(np.mean(rmse_list), 4),
        'RMSE_std'  : round(np.std(rmse_list), 4),
        'MAE_medio' : round(np.mean(mae_list), 4),
        'MAE_std'   : round(np.std(mae_list), 4),
        'R2_medio'  : round(np.mean(r2_list), 4),
        'R2_std'    : round(np.std(r2_list), 4),
        'Tiempo_seg': round(fin - inicio, 2)
    }