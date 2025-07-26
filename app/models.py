from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import BaggingClassifier, BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, AdaBoostRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
import pandas as pd

def get_numeric_model(name, inputs):
    """Create and configure a numeric prediction model pipeline."""
    if len(inputs[0]) > 0 or len(inputs[1]) > 0:
        preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), inputs[0]),  # ← sparse_output=False
            ('num', StandardScaler(), inputs[1])
        ],
        verbose_feature_names_out=False  # Cleaner column names
    ).set_output(transform="pandas")
    else:
        preprocessor = None
    
    model = None
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
    elif name == 'adaboost':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', AdaBoostRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                random_state=inputs[2]
            ))
        ])
    elif name == 'bagging':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', BaggingRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                random_state=inputs[2]
            ))
        ])
    
    else:
        raise ValueError(f"model '{name}' no recognized")
    
    return model, preprocessor

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
    
    model = None
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
    elif name == 'adaboost':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', AdaBoostClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                random_state=inputs[2]
            ))
        ])
    elif name == 'bagging':
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', BaggingClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                random_state=inputs[2]
            ))
        ])
    else:
        raise ValueError(f"model '{name}' no reconocido")
    
    return model

# def X_train_transform(X_train, columns_info, backwards=False):
#     if not backwards:
#         # Forward transformation (encoding)
#         info_encoder = {}
#         X_transformed = X_train.copy()
        
#         # Get categorical columns
#         cat_cols = [info['name'] for info in columns_info if info['dtype'] == 'object']
#         if len(cat_cols) == 0: return X_transformed, info_encoder
        
#         # Initialize and fit LabelEncoders for each categorical column
#         encoders = {col: LabelEncoder().fit(X_train[col]) for col in cat_cols}
        
#         # Transform categorical columns and store encoders
#         for col, encoder in encoders.items():
#             X_transformed[col] = encoder.transform(X_transformed[col])
#             info_encoder[col] = {
#                 'encoder': encoder,
#                 'original_dtype': 'object',
#                 'encoded_values': encoder.classes_
#             }
        
#         return X_transformed, info_encoder
#     else:
#         # Backward transformation (decoding)
#         X_original = X_train.copy()
#         info_encoder = columns_info  # In this case, columns_info contains the encoder info
        
#         # Inverse transform each encoded column
#         for col, enc_info in info_encoder.items():
#             encoder = enc_info['encoder']
#             X_original[col] = encoder.inverse_transform(X_train[col].astype(int))
        
#         return X_original, None

def create_balancing_pipeline(balancing_method, columns_info, random_state, X_train, y_train):
    """Create a pipeline with a balancing method and a model."""
    balance_used = False
    
    if balancing_method == 'none': 
        return X_train, y_train, balance_used
    
    balance_used = True
    
    if balancing_method == 'randomundersampler':
        sampler = RandomUnderSampler(random_state=random_state)
    elif balancing_method == 'randomoversampler':
        sampler = RandomOverSampler(random_state=random_state)
    # elif balancing_method == 'smoteenn':
    #     # Transform categorical columns before SMOTEENN
    #     X_train_transformed, info_encoder = X_train_transform(X_train, columns_info)
    #     sampler = SMOTEENN(random_state=random_state)
    else:
        return X_train, y_train, balance_used
    
    # Apply sampling
    # if balancing_method == 'smoteenn':
    #     X_train_temp, y_train_temp = sampler.fit_resample(X_train_transformed, y_train)
    #     if len(set(y_train_temp)) != len(set(y_train)):
    #         enn = EditedNearestNeighbours(n_neighbors=10, sampling_strategy='all')
    #         sampler = SMOTEENN(enn=enn, random_state=42)
    #         X_train_temp, y_train_temp = sampler.fit_resample(X_train_transformed, y_train)
    # else:
    X_train, y_train = sampler.fit_resample(X_train.copy(), y_train.copy())
    
    # if len(X_train_temp) != 0:
    #     X_train, y_train = X_train_temp, y_train_temp
    #     if balancing_method == 'smoteenn':
    #         # Transform back to original categorical values
    #         X_train, _ = X_train_transform(X_train, info_encoder, backwards=True)
    # else:
    #     balance_used = False
    
    return X_train, y_train, balance_used

