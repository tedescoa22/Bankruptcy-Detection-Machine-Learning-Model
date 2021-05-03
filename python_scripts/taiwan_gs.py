import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
# from xgboost import XGBClassifier


taiwan = pd.read_csv('taiwan.csv')

poland = pd.read_csv('poland.csv')




X_t = taiwan.drop(columns = ['Bankrupt?'])
y_t = taiwan['Bankrupt?']

X_p = poland.drop(columns = ['Bankrupt'])
y_p = poland['Bankrupt']

X_t_train, X_t_test,y_t_train,y_t_test = train_test_split(X_t, y_t, stratify = y_t ,random_state = 9)

X_p_train, X_p_test,y_p_train,y_p_test = train_test_split(X_p, y_p, stratify = y_p,random_state = 9)



sc = StandardScaler()

X_t_train_sc = sc.fit_transform(X_t_train)
X_t_test_sc = sc.transform(X_t_test)

X_p_train_sc = sc.fit_transform(X_p_train)
X_p_test_sc = sc.transform(X_p_test)


pipe_lr = Pipeline([
    ('lr', LogisticRegression())
])
pipe_lr_params = {
    'lr__C': [0.001,0.01,1]
}
# --------------------------------
pipe_rf = Pipeline([
    ('rf', RandomForestClassifier())
])
pipe_rf_params = {
    'rf__n_estimators': [100, 150, 200],
    'rf__max_depth': [None, 1, 2, 3, 4],
    'rf__min_samples_split': [2, 3, 4],
    'rf__min_samples_leaf': [1, 2, 3, 4]
}

# --------------------------------
pipe_gbc = Pipeline([
    ('gbc', GradientBoostingClassifier())
])
pipe_gbc_params = {
    'gbc__loss': ['deviance', 'exponential'],
    'gbc__learning_rate': [0.1, 0.01, 0.001],
    'gbc__n_estimators': [100, 150, 200],
    'gbc__min_samples_split': [2, 3, 4],
    'gbc__min_samples_leaf': [1, 2, 3, 4],
}
# --------------------------------
pipe_knn = Pipeline([
    ('knn', KNeighborsClassifier())
])
pipe_knn_params = {
    'knn__n_neighbors': [5, 10, 25, 50, 75],
    'knn__weights': ['uniform', 'distance']
}

# --------------------------------
pipe_ext = Pipeline([
    ('ext', ExtraTreesClassifier())
])
pipe_ext_params = {
    'ext__n_estimators': [100, 150, 200],
    'ext__max_depth': [None, 1, 2, 3, 4],
    'ext__min_samples_split': [2, 3, 4],
    'ext__min_samples_leaf': [1, 2, 3, 4]
}


gs_lr = GridSearchCV(pipe_lr, pipe_lr_params, cv=3, verbose=0, n_jobs = -1)
gs_rf= GridSearchCV(pipe_rf, pipe_rf_params, cv=3, verbose=0, n_jobs = -1)
gs_gbc = GridSearchCV(pipe_gbc, pipe_gbc_params, cv =3, verbose=0, n_jobs = -1)
gs_knn = GridSearchCV(pipe_knn, pipe_knn_params, cv =3, verbose=0, n_jobs = -1)
gs_ext = GridSearchCV(pipe_ext, pipe_ext_params, cv =3, verbose=0, n_jobs = -1)



gs_lr.fit(X_t_train_sc, y_t_train)
gs_rf.fit(X_t_train_sc, y_t_train)
gs_gbc.fit(X_t_train_sc, y_t_train)
gs_knn.fit(X_t_train_sc, y_t_train)
gs_ext.fit(X_t_train_sc, y_t_train)





print(gs_lr.best_score_, gs_lr.best_params_)
print(gs_rf.best_score_, gs_rf.best_params_)
print(gs_gbc.best_score_, gs_gbc.best_params_)
print(gs_knn.best_score_, gs_knn.best_params_)
print(gs_ext.best_score_, gs_ext.best_params_)


pred_lr = gs_lr.predict(X_t_test_sc, y_t_test)
pred_rf = gs_rf.predict(X_t_test_sc, y_t_test)
pred_gbc = gs_gbc.predict(X_t_test_sc, y_t_test)
pred_knn = gs_knn.predict(X_t_test_sc, y_t_test)
pred_ext = gs_ext.predict(X_t_test_sc, y_t_test)

df_preds = pd.DataFrame()
for func in func_list:
    preds = func.predict(X_t_train_sc, y_t_train)
    df_preds_p[f'{func}'] = df_preds_p

df_preds['actual_values'] = y_t_test
csv = df_preds.to_csv("df_preds_t.csv", index = False)
