import dis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import io
import base64

def analyze_data(df, data):
    """Perform machine learning analysis with enhanced confusion matrix highlighting diagonal."""
    prediction_column = data['prediction_column']
    encoder_columns = data['onehot_columns']
    standar_scale_columns = data['standardscale_columns']
    test_size = data['test_size']
    random_state = data['random_seed']
    name = data['model']
    inputs = [encoder_columns, standar_scale_columns, random_state, data['model_params']]
    
    X = df.drop(prediction_column, axis=1).copy()
    
    # Set modern style with seaborn
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-bright')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })

    fig = plt.figure(figsize=(10, 8))
    cat_opt = False
    if prediction_column in encoder_columns:
        cat_opt = True
        opt = False
        if prediction_column in encoder_columns:
            encoder_columns.remove(prediction_column)
        y_check = set(df[prediction_column])
        
        if len(y_check) >= 2:
            opt = len(y_check) > 2
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(df[prediction_column])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        else:
            encoder = OneHotEncoder(sparse_output=False)
            y_encoded = encoder.fit_transform(df[[prediction_column]])
            y = np.argmax(y_encoded, axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        disc, y_pred = get_resultados_model_categorical(X_train, X_test, y_train, y_test, name, inputs, opt)
        cm = confusion_matrix(y_test, y_pred)
        
        # Custom confusion matrix with highlighted diagonal
        if len(y_check) >= 2:
            labels = label_encoder.classes_
        else:
            labels = encoder.categories_[0]
        
        # Create a custom colormap for the confusion matrix
        cmap = sns.color_palette("viridis", as_cmap=True)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap=cmap, values_format='d', xticks_rotation=45)
        
        # Highlight diagonal with a different color (e.g., bright green)
        for i in range(cm.shape[0]):
            disp.ax_.text(i, i, cm[i, i], ha='center', va='center', 
                         color='white' if cm[i, i] > cm.max() / 2 else 'black',
                         fontsize=14, fontweight='bold', 
                         bbox=dict(facecolor='#00cc00', edgecolor='none', alpha=0.7))
        aux_text = get_aux_text(inputs[3],encoder_columns,standar_scale_columns, prediction_column,X.columns)
        plt.title(f'{prediction_column} Confusion Matrix\n{name} {aux_text}\nAccuracy: {disc[0]:.2f} | Precision: {disc[1]:.2f} | Recall: {disc[2]:.2f} | F1: {disc[3]:.2f}', 
                 pad=20, fontsize=14, weight='bold')
        disp.ax_.set_xlabel('Predicted Label', fontsize=14)
        disp.ax_.set_ylabel('True Label', fontsize=14)
        disp.ax_.grid(False)
        fig = disp.ax_.figure  # This is crucial!
        fig = add_watermark_fig_ax(fig)
    else:
        fig = add_watermark_fig(fig)
        y = df[prediction_column].copy()
        if prediction_column in encoder_columns: encoder_columns.remove(prediction_column)
        if prediction_column in standar_scale_columns: standar_scale_columns.remove(prediction_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        disc, y_pred = get_resultados_model_numeric(X_train, X_test, y_train, y_test, name, inputs)
        aux_text = get_aux_text(inputs[3],encoder_columns,standar_scale_columns, prediction_column,X.columns)
        # Enhanced scatter plot
        plt.scatter(y_test, y_pred, alpha=0.6, s=80, c='#1f77b4', edgecolors='w', linewidth=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='crimson', linewidth=2)
        plt.title(f'{prediction_column} Actual vs Predicted\n{name} {aux_text}\nMSE: {disc[0]:.2f} | RMSE: {disc[1]:.2f} | MAE: {disc[2]:.2f} | R²: {disc[3]:.2f}', 
                 pad=20, fontsize=16, weight='bold')
        plt.xlabel(f'{prediction_column}\nActual Values', fontsize=14)
        plt.ylabel(f'{prediction_column}\nPredicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    buffer = io.BytesIO()
    if cat_opt:
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    else:
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    return {
        'success': True,
        'image_data': base64.b64encode(buffer.getvalue()).decode('utf-8')
    }

def get_resultados_model_numeric(X_train, X_test, y_train, y_test, name, inputs):
    """Evaluate numeric prediction model and return performance metrics."""
    model = get_numeric_model(name, inputs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mse, rmse, mae, r2), y_pred

def get_resultados_model_categorical(X_train, X_test, y_train, y_test, name, inputs, opt):
    """Evaluate categorical prediction model and return performance metrics."""
    model = get_categorical_model(name, inputs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    if opt:
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
    
    return (accuracy, precision, recall, f1), y_pred

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
        raise ValueError(f"model '{name}' no reconocido")
    
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

def get_resultados_model_cat(X_train, X_test, y_train, y_test, name, inputs):
    """Evaluate categorical model with bagging or boosting methods."""
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), X_train.columns)])
    
    model_configs = {
        'Random Forest (Bagging)': ('model', RandomForestClassifier(n_estimators=inputs[0], random_state=42)),
        'Gradient Boosting': ('model', GradientBoostingClassifier(
            n_estimators=inputs[0],
            learning_rate=inputs[1],
            max_depth=inputs[2],
            random_state=42
        )),
        'AdaBoost': ('model', AdaBoostClassifier(n_estimators=inputs[0], random_state=42))
    }
    
    if name not in model_configs:
        raise ValueError(f"model '{name}' no reconocido")
    
    name, model = model_configs[name]
    model = Pipeline([('preprocessor', preprocessor), (name, model)])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, confusion_matrix(y_test, y_pred)

def _format_with_paired_linebreaks(items):
    """Helper function to format items with newline after every two items."""
    if not items:
        return ""
    formatted = [
        str(item) + ('\n' if i % 2 == 1 and i < len(items) - 1 else ', ' if i < len(items) - 1 else '')
        for i, item in enumerate(items)
    ]
    return ''.join(formatted)

def get_aux_text(params, encoder_columns, standar_scale_columns, prediction_column, columns):
    sections = []
    
    # Format params
    if params:
        params_text = _format_with_paired_linebreaks([f"{k}: {v}" for k, v in params.items()])
        sections.append(params_text)
    
    # Format encoded columns
    if encoder_columns:
        sections.append("Encoded Columns: " + _format_with_paired_linebreaks(encoder_columns))
    
    # Format scaled columns
    if standar_scale_columns:
        sections.append("Scaled Columns: " + _format_with_paired_linebreaks(standar_scale_columns))
    
    # Format original columns
    remaining_cols = [col for col in columns 
                     if col not in encoder_columns 
                     and col not in standar_scale_columns 
                     and col != prediction_column]
    if remaining_cols:
        sections.append("Original Columns: " + _format_with_paired_linebreaks(remaining_cols))
    
    return '\n'.join(sections)

def add_watermark_fig(fig):
    # Add subtle diagonal watermark
    watermark_text = "github.com/rdominguez89"
    watermark_color = 'lightgray'
    watermark_alpha = 0.3  # Very subtle
    watermark_fontsize = 20
    watermark_angle = 30  # Diagonal angle
    
    # Add watermark to the figure background (zorder=0 puts it behind everything)
    # fig.text(0.45, 0.8, watermark_text, 
    #          fontsize=watermark_fontsize, color=watermark_color, alpha=watermark_alpha,
    #          rotation=watermark_angle, zorder=-100,
    #          transform=fig.transFigure)
    
    # Repeat the watermark in a grid pattern (adjust spacing as needed)
    for x in [0.05, 0.5]:
        for y in [0.1, 0.3, 0.6]:
            if not (0.45 < x < 0.55 and 0.45 < y < 0.55):  # Avoid center where main content is
                fig.text(x, y, watermark_text, 
                        fontsize=watermark_fontsize, color=watermark_color, alpha=watermark_alpha,
                        rotation=watermark_angle, zorder=0,
                        transform=fig.transFigure)
    
    return fig

def add_watermark_fig_ax(fig):
    """Add watermark to entire figure background (not just Axes)"""
    watermark_text = "github.com/rdominguez89"
    watermark_color = 'lightgray'
    watermark_alpha = 0.3
    watermark_fontsize = 15
    watermark_angle = 30
    
    # Diagonal grid of watermarks
    for x in [0.2, 0.8]:
        for y in [0.2, 0.4, 0.6]:
            fig.text(x,y, watermark_text,
                fontsize=watermark_fontsize, color=watermark_color,
                alpha=watermark_alpha, rotation=watermark_angle,
                zorder=0,
                transform=fig.transFigure, clip_on=False)
    
    return fig