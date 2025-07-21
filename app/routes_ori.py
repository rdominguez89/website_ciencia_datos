from flask import Blueprint, render_template, request, jsonify, send_file
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score,precision_score, recall_score, roc_auc_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/cleaning')
def cleaning():
    return render_template('cleaning.html')

@bp.route('/supervised')
def supervised():
    return render_template('supervised.html')

@bp.route('/nosupervised')
def nosupervised():
    return render_template('nosupervised.html')

@bp.route('/inference')
def inference():
    return render_template('inference.html')

@bp.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            df = pd.read_csv(file)

            # Apply limits
            if len(df) > 1000:
                return jsonify({'error': 'File exceeds 1000 row limit'}), 400
            if len(df.columns) > 10:
                return jsonify({'error': 'File exceeds 10 column limit'}), 400
            
            return jsonify({'success': True, 'df': df.where(pd.notnull(df), '-').to_dict(orient='records')
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@bp.route('/api/statsummary', methods=['POST'])
def statsummary():
    data = request.json
    df = pd.DataFrame.from_dict(data['df'])
    df = df.replace('-', None)
    
    # Create null counts by column
    null_counts = df.isnull().sum()
    columns_with_nulls = {col: int(count) for col, count in null_counts.items() if count > 0}
    
    stats = {
        'rows': len(df),
        'columns': len(df.columns),
        'duplicates': int(df.duplicated().sum()),
        'total_null': int(null_counts.sum()),
        'columns_with_nulls': columns_with_nulls,
        'columns_with_null_count': len(columns_with_nulls)
    }
    
    # Get column data types
    dtype_info = []
    for col in df.columns:
        dtype_info.append({
            'column': col,
            'type': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'null_values': int(null_counts[col])
        })
    
    # Get descriptive statistics for numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_stats = {}
    if not df.empty:
        # Get numerical stats
        num_stats = df.select_dtypes(include=['number']).describe().round(2).to_dict()

    return jsonify({
        'success': True,
        'data_head': df.head(5).to_html(classes='table table-striped'),
        'null_counts': null_counts.to_frame('null_count').to_html(classes='table table-striped'),
        'duplicated_rows': df[df.duplicated(keep=False)].to_html(classes='table table-striped', index=False) if stats['duplicates'] > 0 else "No duplicated rows found",
        'stats': stats,
        'dtype_info': dtype_info,
        'num_stats': num_stats,
        'columns': list(df.columns),
        'column_order':list(numerical_cols),
        'df': df.where(pd.notnull(df), '-').to_dict(orient='records')
    })

@bp.route('/api/clean_data', methods=['POST'])
def clean_data():
    try:
        data = request.json
        df = data['df']
        df = pd.DataFrame.from_dict(df)
        df = df.replace('-', None)
        
        if data.get('remove_nulls', False):
            df = df.dropna()
        
        if data.get('remove_duplicates', False):
            df = df.drop_duplicates()
        
        df = df.copy()

        return jsonify({'success': True, 'df': df.where(pd.notnull(df), '-').to_dict(orient='records')})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@bp.route('/api/create_visualizations', methods=['POST'])
def create_visualizations():
    data = request.json
    df = pd.DataFrame.from_dict(data['df'])
    df = df.replace('-', None)
    
    try:
        visualizations = {
            'boxplots': [],  # For numeric columns
            'distributions': [],  # For categorical columns
            'outliers_data': {}  # To store outlier information
        }
        
        # Numerical columns - Boxplots
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            fig, ax = plt.subplots(figsize=(7, 4))
            boxplot = sns.boxplot(data=df, y=col, ax=ax)
            
            # Get outliers from the boxplot artists
            outliers = []
            for artist in ax.get_children():
                if isinstance(artist, matplotlib.lines.Line2D):
                    # Outliers are represented as Line2D objects in boxplot
                    if len(artist.get_xdata()) == 1:  # Single point (outlier)
                        outlier_value = artist.get_ydata()[0]
                        outliers.append(outlier_value)
            
            # Get rows containing these outlier values
            outlier_rows = df[df[col].isin(outliers)].reset_index()
            
            # Convert plot to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            plt.close(fig)
            del fig
            
            visualizations['boxplots'].append({
                'title': f'Boxplot of {col}',
                'image_data': img_str,
                'type': 'boxplot',
                'outliers_count': len(outliers)
            })
            
            # Store outlier information
            visualizations['outliers_data'][col] = {
                'outlier_values': outliers,
                'outlier_rows': outlier_rows.to_dict('records'),  # Convert to list of dicts
                'total_outliers': len(outliers)
            }
        return jsonify(visualizations)
        # Categorical columns - Histograms/Count plots (unchanged)
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in df.columns:
            if df[col].nunique() < 10:
                fig, ax = plt.subplots(figsize=(8, 5))
                if col in num_cols:
                    sns.countplot(x=df[col], 
                                order=df[col].value_counts().sort_index().index, 
                                ax=ax)
                else:
                    sns.countplot(x=df[col], 
                                order=df[col].value_counts().sort_values().index, 
                                ax=ax)
                plot_type = 'countplot'
            else:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df[col], kde=True, ax=ax)
                plot_type = 'histplot'
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()
            
            plt.close(fig)
            del fig
            
            visualizations['distributions'].append({
                'title': f'Distribution of {col}',
                'image_data': img_str,
                'type': plot_type
            })
        
        return jsonify(visualizations)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/remove_outliers', methods=['POST'])
def remove_outliers():
    try:
        # Get the request data
        data = request.json
        df = pd.DataFrame.from_dict(data['df'])
        df = df.replace('-', None)
        outliers = data.get('outliers', [])
    
        
        # Create a set of (column, index) tuples for faster lookup
        outliers_to_remove = {(o['column'], o['index']) for o in outliers}
        
        # Filter out the selected outliers
        filtered_df = df[~df.apply(
            lambda row: any(
                (col, row.name) in outliers_to_remove 
                for col in df.columns
            ),
            axis=1
        )]
        
        # Reset index if needed
        filtered_df.reset_index(drop=True, inplace=True)
        
        # Convert back to dictionary for JSON response
        result = {
            'success': True,
            'df': filtered_df.where(pd.notnull(filtered_df), '-').to_dict(orient='records'),
            'removed_count': len(df) - len(filtered_df)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400


@bp.route('/api/create_correlation_plots', methods=['POST'])
def create_correlation_plots():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['df'])
        
        # Select only numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_cols) < 2:
            return jsonify({
                'error': 'Need at least 2 numerical columns for correlation plots',
                'success': False
            }), 400
        
        # Create pairplot
        plt.figure(figsize=(12, 10))
        pairplot = sns.pairplot(df[numerical_cols])
        
        # Save to buffer
        buffer = BytesIO()
        pairplot.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        heatmap = sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', center=0)
        heatmap_buffer = BytesIO()
        plt.savefig(heatmap_buffer, format='png', bbox_inches='tight')
        plt.close()
        heatmap_buffer.seek(0)
        
        return jsonify({
            'success': True,
            'plots': [
                {
                    'title': 'Pairplot of Numerical Features',
                    'image_data': base64.b64encode(buffer.getvalue()).decode('utf-8')
                },
                {
                    'title': 'Correlation Heatmap',
                    'image_data': base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
                }
            ]
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@bp.route('/api/save_data', methods=['POST'])
def save_data():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['df'])
        selected_columns = data.get('columns', df.columns.tolist())
        filename = data.get('filename', 'processed_data')
        
        # Filter columns
        df = df[selected_columns]
        
        # Save to temporary file
        temp_path = f"/tmp/{filename}.csv"
        df.to_csv(temp_path, index=False)
        
        # Return the file
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"{filename}.csv",
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@bp.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        # Get the request data
        data = request.json
        df = pd.DataFrame.from_dict(data['df'])
        
        prediction_column = data['prediction_column']
        encoder_columns = data['onehot_columns']
        standar_scale_columns = data['standardscale_columns']
        train_size = data['train_size']
        test_size = data['test_size']
        random_state = data['random_seed']
        nombre = data['model']
        inputs = [encoder_columns, standar_scale_columns,random_state, data['model_params']]
        
        # Create figure with appropriate size
        X = df.drop(prediction_column, axis=1).copy()
        plt.figure(figsize=(10, 6))
        if prediction_column in encoder_columns:
            opt = False
            if prediction_column in encoder_columns: encoder_columns.remove(prediction_column)
            y_check = set(df[prediction_column])
            # Check if binary case with 0 and 1
            # if len(y_check) == 2 and {0, 1} == y_check:
            #     y = df[prediction_column].copy()
            #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            # Multi-class case
            if len(y_check) >= 2:
                opt = True
                if len(y_check) == 2: opt = False
                # Apply LabelEncoder instead of OneHotEncoder for multi-class classification
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(df[prediction_column])
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            else:
                # Apply OneHotEncoder for multi-class case
                encoder = OneHotEncoder(sparse_output=False)
                y_encoded = encoder.fit_transform(df[[prediction_column]])
                y = pd.DataFrame(y_encoded, columns=encoder.get_feature_names_out([prediction_column]))
                y = np.argmax(y, axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            disc, y_pred = get_resultados_modelo_categorical(X_train, X_test, y_train, y_test, nombre, inputs, opt)

            cm = confusion_matrix(y_test, y_pred)
            if len(y_check) >= 2:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
                disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')
                if len(y_check) == 2:
                    plt.title(f'Confusion Matrix - {nombre}\nAccuracy: {disc[0]:.2f}, Precision: {disc[1]:.2f}, Recall: {disc[2]:.2f}, F1: {disc[3]:.2f}')
                else:
                    plt.title(f'Confusion Matrix (Multi-Class) - {nombre}\nAccuracy: {disc[0]:.2f} Precision: {disc[1]:.2f}, Recall: {disc[2]:.2f}, F1: {disc[3]:.2f}')
                plt.tight_layout()  # Adjust layout to prevent label cutoff
            else:
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
                # Plot with improved styling
                disp.plot(cmap='Blues', values_format='d')  # 'd' for integer formatting
                plt.title(f'Confusion Matrix - {nombre}\nAccuracy: {disc[0]:.2f}, Precision: {disc[1]:.2f}, Recall: {disc[2]:.2f}, F1: {disc[3]:.2f}')
                plt.grid(False)  # Typically no grid for confusion matrices
        else:
            y = df[prediction_column].copy()
            if prediction_column in encoder_columns: encoder_columns.remove(prediction_column)
            if prediction_column in standar_scale_columns: standar_scale_columns.remove(prediction_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            disc, y_pred = get_resultados_modelo_numeric(X_train, X_test, y_train, y_test, nombre, inputs)

            # Plot actual vs predicted with reference line
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r')
            plt.title(f'Actual vs Predicted - {nombre}\nMSE: {disc[0]:.2f}, RMSE: {disc[1]:.2f}, MAE: {disc[2]:.2f}, R²: {disc[3]:.2f}')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.grid(True)


            
        
        # Save plot to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        
        # Prepare result
        result = {
            'success': True,
            'image_data': base64.b64encode(buffer.getvalue()).decode('utf-8'),
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 400
    


def get_resultados_modelo_numeric(X_train, X_test, y_train, y_test, nombre, inputs):
   
    modelo = get_numeric_model(nombre, inputs)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mse, rmse, mae, r2), y_pred

def get_resultados_modelo_categorical(X_train, X_test, y_train, y_test, nombre, inputs, opt):
   
    modelo = get_categorical_model(nombre, inputs)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    #y_proba = modelo.predict_proba(X_test)[:, 1] if hasattr(modelo, 'predict_proba') else None
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    if opt:
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
    else:
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        #roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    
    return (accuracy, precision, recall, f1),  y_pred

def get_categorical_model(nombre, inputs):
    # Preprocessor for both categorical and numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), inputs[0]),  # Categorical features
            ('num', StandardScaler(), inputs[1])              # Numeric features
        ])
    
    if nombre == 'logistic':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=float(inputs[3]['C']),
                penalty=inputs[3]['penalty'],
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'svc':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', SVC(
                C=float(inputs[3]['C']),
                kernel=inputs[3]['kernel'],
                gamma=inputs[3]['gamma'],
                probability=True,  # For ROC AUC
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'randomforest':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                max_depth=int(inputs[3]['n_estimators']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'gradientboosting':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', GradientBoostingClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'xgboost':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2],
                use_label_encoder=False,
                eval_metric=inputs[3]['eval_metric']
            ))
        ])
    
    elif nombre == 'knn':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier(
                n_neighbors=int(inputs[3]['n_neighbors']),
            ))
        ])
    
    elif nombre == 'decisiontree':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    else:
        raise ValueError(f"Modelo '{nombre}' no reconocido")
    
    return modelo



def get_numeric_model(nombre, inputs):
    preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(drop='first'), inputs[0]),  # OneHot para categóricas
                ('num', StandardScaler(), inputs[1])  # Escalado para numéricas (o MinMaxScaler)
            ])
    
    if nombre == 'poly':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('poly', PolynomialFeatures(degree=int(inputs[3]['degree']))),
            ('regressor', LinearRegression())
        ])
    
    elif nombre == 'linear':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
    
    elif nombre == 'ridge':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Ridge(alpha=float(inputs[3]['alpha']), random_state=inputs[2]))
        ])
    
    elif nombre == 'lasso':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', Lasso(alpha=float(inputs[3]['alpha']), random_state=inputs[2]))
        ])
    
    elif nombre == 'elasticnet':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', ElasticNet(alpha=float(inputs[3]['alpha']), l1_ratio=float(inputs[3]['l1_ratio']), random_state=inputs[2]))
        ])
    
    elif nombre == 'svr':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', SVR(C=float(inputs[3]['C']), kernel=inputs[3]['kernel'], gamma=inputs[3]['gamma']))
        ])
    
    elif nombre == 'randomforest':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'gradientboosting':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'xgboost':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=int(inputs[3]['n_estimators']),
                learning_rate=float(inputs[3]['learning_rate']),
                max_depth=int(inputs[3]['max_depth']),
                random_state=inputs[2]
            ))
        ])
    
    elif nombre == 'knn':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', KNeighborsRegressor(n_neighbors=int(inputs[3]['n_neighbors'])))
        ])
    
    elif nombre == 'decisiontree':
        modelo = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(
                max_depth=int(inputs[3]['max_depth']),
                min_samples_split=int(inputs[3]['min_samples_split']),
                random_state=inputs[2]
            ))
        ])
    
    else:
        raise ValueError(f"Modelo '{nombre}' no reconocido")
    
    return modelo

def get_resultados_modelo_cat(X_train, X_test, y_train, y_test, nombre, inputs):
        preprocessor = ColumnTransformer(
            transformers=[
            ('num', StandardScaler(), X.columns)  # Escalado para numéricas (o MinMaxScaler)
            ])

        if nombre == 'Random Forest (Bagging)':
            modelo = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestClassifier(n_estimators=inputs[0], random_state=42))
            ])
        elif nombre == 'Gradient Boosting':
            modelo = Pipeline([
                ('preprocessor', preprocessor),
                ('model', GradientBoostingClassifier(n_estimators=inputs[0], learning_rate=inputs[1], max_depth=inputs[2], random_state=42))
            ])
        elif 'AdaBoost':
            modelo = Pipeline([
                ('preprocessor', preprocessor),
                ('model', AdaBoostClassifier(n_estimators=inputs[0], random_state=42))
            ])

        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        #print(f"{nombre}: Exactitud = {accuracy:.2f}")


        return accuracy, confusion_matrix(y_test, y_pred)






def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'csv'}