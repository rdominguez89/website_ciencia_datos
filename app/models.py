from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


def get_numeric_model(name, inputs):
    """Create and configure a numeric prediction model pipeline."""
    if len(inputs[0]) > 0 or len(inputs[1]) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), inputs[0]),
                ('num', StandardScaler(), inputs[1])
            ])
    else:
        preprocessor = None
    
    if name == 'poly':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=int(inputs[3]['degree']))),
            ('regressor', LinearRegression())
        ])
    
    elif name == 'linear':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
    
    elif name == 'ridge':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=float(inputs[3]['alpha']), random_state=inputs[2]))
        ])
    
    elif name == 'lasso':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(alpha=float(inputs[3]['alpha']), random_state=inputs[2]))
        ])
    
    elif name == 'elasticnet':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(alpha=float(inputs[3]['alpha']), l1_ratio=float(inputs[3]['l1_ratio']), random_state=inputs[2]))
        ])
    
    elif name == 'svr':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR(C=float(inputs[3]['C']), kernel=inputs[3]['kernel'], gamma=inputs[3]['gamma']))
        ])
    
    elif name == 'randomforest':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'gradientboosting':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'xgboost':
        raise ValueError(f"model '{name}' Temporarily unavailable")
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'knn':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=int(inputs[3]['n_neighbors'])))
        ])
    
    elif name == 'decisiontree':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    else:
        raise ValueError(f"model '{name}' no recognized")
    
    return model

def get_categorical_model(name, inputs):
    """Create and configure a categorical prediction model pipeline."""
    if len(inputs[0]) > 0 or len(inputs[1]) > 0:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), inputs[0]),
                ('num', StandardScaler(), inputs[1])
            ])
    else:
        preprocessor = None
    
    if name == 'logistic':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=float(inputs[3]['C']),
                penalty=inputs[3]['penalty'],
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'svc':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=float(inputs[3]['C']),
                kernel=inputs[3]['kernel'],
                gamma=inputs[3]['gamma'],
                probability=True,  # For ROC AUC
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'randomforest':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                max_depth=int(inputs[3]['n_estimators']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'gradientboosting':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif name == 'xgboost':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2],
                eval_metric=inputs[3]['eval_metric']
            ))
        ])
    
    elif name == 'knn':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=int(inputs[3]['n_neighbors']),
            ))
        ])
    
    elif name == 'decisiontree':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    else:
        raise ValueError(f"model '{name}' no reconocido")
    
    return model

