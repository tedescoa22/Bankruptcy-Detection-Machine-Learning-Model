import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
from xgboost import XGBClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier



taiwan = pd.read_csv('data/taiwan.csv')

poland = pd.read_csv('data/poland.csv')

taiwan = taiwan.head(20)
poland = poland.head(20)


X_t = taiwan.drop(columns= ['Bankrupt?'])
y_t = taiwan['Bankrupt?']

X_p = poland.drop(columns= ['Bankrupt'])
y_p = poland['Bankrupt']

X_t_train, X_t_test,y_t_train,y_t_test = train_test_split(X_t, y_t, stratify = y_t ,random_state = 9)

X_p_train, X_p_test,y_p_train,y_p_test = train_test_split(X_p, y_p, stratify = y_p,random_state = 9)



sc = StandardScaler()

X_t_train_sc = sc.fit_transform(X_t_train)
X_t_test_sc = sc.transform(X_t_test)

X_p_train_sc = sc.fit_transform(X_p_train)
X_p_test_sc = sc.transform(X_p_test)


pipe_lr = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('lr', LogisticRegression())
])
pipe_lr_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'lr__C': [0.001,0.01,1]
}
# --------------------------------
pipe_rf = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('rf', RandomForestClassifier())
])
pipe_rf_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'rf__n_estimators': [100, 150, 200, 250],
    'rf__max_depth': [None, 1, 2, 3, 4, 5],
    'rf__min_samples_split': [2, 3, 4, 5],
    'rf__min_samples_leaf': [1, 2, 3, 4, 5]
}
# --------------------------------
pipe_svc = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('svc', SVC())
])
pipe_svc_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'svc__C':np.arange(1, 5, 20),
    'svc__kernel':['linear', 'rbf', 'polynomial','sigmoid'],
    'svc__degree':[1, 2, 3, 4, 5]
}
# --------------------------------
pipe_xgb = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('xgb', XGBClassifier(use_label_encoder=False, objective='binary:logistic', verbosity = 0))
])
pipe_xgb_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'xgb__n_estimators': [10, 50, 100, 200, 500],
    'xgb__learning_rate': [0.1, 0.2, 0.5, 0.7, 0.9]
}
# --------------------------------
pipe_gbc = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('gbc', GradientBoostingClassifier())
])
pipe_gbc_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'gbc__loss': ['deviance', 'exponential'],
    'gbc__learning_rate': [0.1, 0.01, 0.001],
    'gbc__n_estimators': [100, 150, 200, 250],
    'gbc__min_samples_split': [2, 3, 4, 5],
    'gbc__min_samples_leaf': [1, 2, 3, 4, 5],
    'gbc_min_depth': [3, 4, 5, 6]
}
# --------------------------------
pipe_knn = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])
pipe_knn_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'knn__n_neighbors': [5, 10, 25, 50, 75, 100],
    'knn__weights': ['uniform', 'distance']
}
# --------------------------------
pipe_dec = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('dec', DecisionTreeClassifier())
])
pipe_dec_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'dec__splitter': ['best', 'random'],
    'dec__min_samples_split': [2, 3, 4, 5],
    'dec__min_samples_leaf': [1, 2, 3, 4, 5]
}
# --------------------------------
pipe_ext = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('ext', ExtraTreesClassifier())
])
pipe_ext_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'ext__n_estimators': [100, 150, 200, 250],
    'ext__max_depth': [None, 1, 2, 3, 4, 5],
    'ext__min_samples_split': [2, 3, 4, 5],
    'ext__min_samples_leaf': [1, 2, 3, 4, 5]
}
# --------------------------------
pipe_bag = Pipeline([
    ('poly', PolynomialFeatures()),
    ('pca', PCA()),
    ('bag', BaggingClassifier())
])
pipe_bag_params = {
    'poly__interaction_only':[False, True],
    'pca__n_components': [5, 10, 25, 50, 75, 100],
    'bag__n_estimators': [10, 15, 20, 25],
    'bag__max_samples': [1, 2, 3, 4, 5],
    'bag__max_features': [1, 2, 3, 4, 5]
}
# --------------------------------

gs_lr = GridSearchCV(pipe_lr, pipe_lr_params, cv=3, verbose=0)
gs_rf= GridSearchCV(pipe_rf, pipe_rf_params, cv=3, verbose=0)
gs_svc= GridSearchCV(pipe_svc, pipe_svc_params, cv=3, verbose=0)
gs_xgb = GridSearchCV(pipe_xgb, pipe_xgb_params, cv =3, verbose=0)
gs_gbc = GridSearchCV(pipe_gbc, pipe_gbc_params, cv =3, verbose=0)
gs_knn = GridSearchCV(pipe_knn, pipe_knn_params, cv =3, verbose=0)
gs_dec = GridSearchCV(pipe_dec, pipe_dec_params, cv =3, verbose=0)
gs_ext = GridSearchCV(pipe_ext, pipe_ext_params, cv =3, verbose=0)
gs_bag = GridSearchCV(pipe_bag, pipe_bag_params, cv =3, verbose=0)


gs_lr.fit(X_t_train_sc, y_t_train)
gs_rf.fit(X_t_train_sc, y_t_train)
gs_svc.fit(X_t_train_sc, y_t_train)
gs_xgb.fit(X_t_train_sc, y_t_train)
gs_gbc.fit(X_t_train_sc, y_t_train)
gs_knn.fit(X_t_train_sc, y_t_train)
gs_dec.fit(X_t_train_sc, y_t_train)
gs_ext.fit(X_t_train_sc, y_t_train)
gs_bag.fit(X_t_train_sc, y_t_train)


def scores(est,X_train,X_test,y_train,y_test,y,name):
    preds = est.predict(X_test)
    ns_probs = [0 for _ in range(len(y_test))]
    est_probs = est.predict_proba(X_test)
    est_probs = est_probs[:,1]
    ns_auc = roc_auc_score(y_test, ns_probs)
    est_auc =roc_auc_score(y_test, est_probs)
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    est_fpr, est_tpr, _ = roc_curve(y_test, est_probs)
    est_precision, est_recall, _ = precision_recall_curve(y_test, est_probs)
    est_f1, pr_est_auc = f1_score(y_test, preds), auc(est_recall, est_precision)
    no_skill = len(y_test[y_test==1]) / len(y_test)
    print(f'The Null Model is: {y.mean()}')
    print('----------------------------------------------------------')
    print(f'The Training set accuracy score is: {lr.score(X_train,y_train)}')
    print('----------------------------------------------------------')
    print(f'The Testing set accuracy score is: {lr.score(X_test,y_test)}')
    print('----------------------------------------------------------')
    print(f'The Cross Validation accuracy score is: {cross_val_score(lr, X_train, y_train).mean()}')
    print('----------------------------------------------------------')
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('----------------------------------------------------------')
    print(f'{name}: ROC AUC=%.3f' % (est_auc))
    print('----------------------------------------------------------')
    print(f'{name}: f1=%.3f auc=%.3f' % (est_f1, pr_est_auc))
    c_report=pd.DataFrame(classification_report(y_test,
                                                preds,
                                                output_dict= True,
                                                target_names=['0, didn\'t subscribe','1, subscribed']))
    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0,1]).ravel()
    plot_confusion_matrix(est, X_test, y_test,
                          cmap='PuBuGn', values_format='d',
                          display_labels=['0, didn\'t subscribe','1, subscribed'], )
    plt.show()
    print('----------------------------------------------------------')
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(est_fpr, est_tpr, marker='.', label='Logistic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show();
    print('----------------------------------------------------------')
    plt.plot([0,1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(est_recall, est_precision, marker='.', label=name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.show();
    return c_report.T

lr = scores(gs_lr, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'lr')
rf = scores(gs_rf, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'rf')
svc = scores(gs_svc, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'svc')
xgb = scores(gs_xbg, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'xgb')
gbc = scores(gs_gbc, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'gbc')
knn = scores(gs_knn, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'knn')
dec = scores(gs_dec, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'dec')
ext = scores(gs_ext, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'ext')
bag = scores(gs_bag, X_t_train_sc, X_t_test_sc, y_t_train, y_t_test,y_t,name = 'bag')


lr_csv = lr.to_csv('lr_csv.csv')
rf_csv = lr.to_csv('rf_csv.csv')
svc_csv = lr.to_csv('svc_csv.csv')
xgb_csv = lr.to_csv('xgb_csv.csv')
gbc_csv = lr.to_csv('gbc_csv.csv')
knn_csv = lr.to_csv('knn_csv.csv')
dec_csv = lr.to_csv('dec_csv.csv')
ext_csv = lr.to_csv('ext_csv.csv')
bag_csv = lr.to_csv('bag_csv.csv')
