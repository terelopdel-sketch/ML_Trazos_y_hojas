import time
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