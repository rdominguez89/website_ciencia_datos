import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer, accuracy_score, f1_score, precision_score, recall_score
import io
import base64
from .models import get_numeric_model, get_categorical_model


def set_common_plot_config():
    """Configure Matplotlib's global parameters for a consistent style."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })


def _format_with_paired_linebreaks(items):
    """
    Format a list of items, inserting a newline after every two items.
    
    Args:
        items (list): List of items to format.
    
    Returns:
        str: A formatted string with paired linebreaks.
    """
    if not items:
        return ""
    formatted = [
        str(item) + ('\n' if i % 2 == 1 and i < len(items) - 1 else ', ' if i < len(items) - 1 else '')
        for i, item in enumerate(items)
    ]
    return ''.join(formatted)


def get_aux_text(params, encoder_columns, standar_scale_columns, prediction_column, columns):
    """
    Build auxiliary text information from parameters and column lists.
    
    Args:
        params (dict): Dictionary of model parameters.
        encoder_columns (list): List of encoded column names.
        standar_scale_columns (list): List of scaled column names.
        prediction_column (str): Name of the prediction column.
        columns (iterable): All column names in the feature set.
        
    Returns:
        str: Formatted auxiliary text.
    """
    sections = []
    # Format parameters
    if params:
        params_text = _format_with_paired_linebreaks([f"{k}: {v}" for k, v in params.items()])
        sections.append(params_text)
    # Format encoded columns
    if encoder_columns:
        sections.append("Encoded Columns: " + _format_with_paired_linebreaks(encoder_columns))
    # Format scaled columns (including prediction column)
    sections.append("Scaled Columns: " + _format_with_paired_linebreaks(standar_scale_columns + [prediction_column]))
    # Format remaining original columns (exclude encoded/scaled/prediction)
    remaining_cols = [col for col in columns 
                      if col not in encoder_columns 
                      and col not in standar_scale_columns 
                      and col != prediction_column]
    if remaining_cols:
        sections.append("Original Columns: " + _format_with_paired_linebreaks(remaining_cols))
    return '\n'.join(sections)


def classification_analysis(df, data, X, encoder_columns, standar_scale_columns):
    """
    Perform classification analysis including cross-validation or train-test split.
    
    This function preprocesses the target, builds the classification model, 
    computes metrics, and plots the confusion matrix and CV results.
    
    Args:
        df (DataFrame): Input data.
        data (dict): Dictionary of configuration parameters.
        X (DataFrame): Feature dataset.
        encoder_columns (list): List of one-hot encoded columns.
        standar_scale_columns (list): List of columns scaled with StandardScaler.
    """
    prediction_column = data['prediction_column']
    test_size = data['test_size']
    random_state = data['random_seed']
    name = data['model']
    params = data['model_params']
    cv_config = data['cross_validation'] if len(data['cross_validation']) > 0 else None

    # Remove prediction column from encoder-columns
    if prediction_column in encoder_columns:
        encoder_columns.remove(prediction_column)
    # Determine if multi-class classification is required
    y_check = set(df[prediction_column])
    opt = len(y_check) > 2
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[prediction_column])
    aux_text = get_aux_text(params, encoder_columns, standar_scale_columns, prediction_column, X.columns)
    model = get_categorical_model(name, [encoder_columns, standar_scale_columns, random_state, params])
    
    if cv_config is not None:
        # Cross-validation path: perform CV and plot aggregated confusion matrix and metrics trends
        cv_results = perform_cross_validation(model, X, y, cv_config, random_state, classification=True)
        fig = plt.figure(figsize=(5, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # Plot confusion matrix
        ax1 = plt.subplot(gs[0])
        plot_confusion_matrix(cv_results, label_encoder.classes_, ax=ax1)
        ax1.set_title(f'{prediction_column} {name} (Cross-Validated)\n{aux_text}', fontsize=14, weight='bold')
        
        # Plot CV results metrics
        ax2 = plt.subplot(gs[1])
        plot_cv_results(cv_results, ax=ax2)
        plt.tight_layout()
    else:
        # Train-test split path
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y)
        disc, y_pred = get_resultados_model_categorical(X_train, X_test, y_train, y_test, model, opt)
        cv_results = {'y_test': y_test, 'y_pred': y_pred}
        
        # Plot confusion matrix for test predictions
        plot_confusion_matrix(cv_results, label_encoder.classes_)
        plt.title(f'{prediction_column} Confusion Matrix\n{name} {aux_text}\n'
                  f'Accuracy: {disc[0]:.2f} | Precision: {disc[1]:.2f} | Recall: {disc[2]:.2f} | F1: {disc[3]:.2f}', 
                  pad=20, fontsize=14, weight='bold')


def regression_analysis(df, data, X, encoder_columns, standar_scale_columns):
    """
    Perform regression analysis including cross-validation or train-test split.
    
    This function sets up the regression model, scales the target,
    computes performance metrics, and plots actual vs predicted values.
    
    Args:
        df (DataFrame): Input data.
        data (dict): Dictionary of configuration parameters.
        X (DataFrame): Feature dataset.
        encoder_columns (list): List of columns that were one-hot encoded.
        standar_scale_columns (list): List of columns that require standard scaling.
    """
    prediction_column = data['prediction_column']
    test_size = data['test_size']
    random_state = data['random_seed']
    name = data['model']
    params = data['model_params']
    cv_config = data['cross_validation'] if len(data['cross_validation']) > 0 else None

    # Ensure prediction column is not in the feature set
    if prediction_column in encoder_columns:
        encoder_columns.remove(prediction_column)
    if prediction_column in standar_scale_columns:
        standar_scale_columns.remove(prediction_column)
    
    model, preprocesor = get_numeric_model(name, [encoder_columns, standar_scale_columns, random_state, params])
    
    # Scale the target variable
    y = df[prediction_column].copy()
    scaler = StandardScaler()
    y = scaler.fit_transform(y.values.reshape(-1, 1))
    aux_text = get_aux_text(params, encoder_columns, standar_scale_columns, prediction_column, X.columns)
    
    if cv_config is not None:
        # Cross-validation path: evaluate model across folds and plot predictions
        cv_results = perform_cross_validation(model, X, y, cv_config, random_state, classification=False)
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax1 = plt.subplot(gs[0])
        plot_actual_vs_predicted(cv_results, name, prediction_column, aux_text, test_size, random_state, scaler, ax=ax1)
        ax2 = plt.subplot(gs[1])
        plot_cv_results(cv_results, ax=ax2)
        fig = add_watermark_fig(fig, xs=[0.2])
    else:
        # Train-test split path
        fig = plt.figure(figsize=(6, 6))
        fig = add_watermark_fig(fig, xs=[0.3])
        if preprocesor and len(encoder_columns) > 0:
            X = preprocesor.fit_transform(X)
            inputs_cat = [col for col in X.columns if not any(num_col in col for num_col in standar_scale_columns)]
            input_list = [inputs_cat, standar_scale_columns, random_state, params]
        else:
            input_list = [encoder_columns, standar_scale_columns, random_state, params]
        model, preprocesor = get_numeric_model(name, input_list)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        disc, y_pred = get_resultados_model_numeric(X_train, X_test, y_train, y_test, model)
        
        # Inverse transform for human-friendly plotting
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_inverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
        
        plt.scatter(y_test_inverse, y_pred_inverse, alpha=0.6, s=80, c='#1f77b4',
                    edgecolors='w', linewidth=0.5)
        plt.plot([min(y_test_inverse), max(y_pred_inverse)],
                 [min(y_test_inverse), max(y_pred_inverse)],
                 '--', color='crimson', linewidth=2)
        ax = plt.gca()
        ax.set_title(f'{prediction_column} Actual vs Predicted', fontsize=12, weight='bold', loc='center')
        ax.text(0.96, 0.04, f'{name} {aux_text}\nTraining: {(100-100*test_size):.0f}% | '
                 f'Test: {100*test_size:.0f}% | seed: {random_state}', transform=ax.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(facecolor='white', alpha=0.5, pad=4))
        metrics_text = (f"MSE: {disc[0]:.2f}\nRMSE: {disc[1]:.2f}\n"
                        f"MAE: {disc[2]:.2f}\nR²: {disc[3]:.2f}")
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                fontsize=10, fontweight='bold', verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5))
        ax.tick_params(axis='both', rotation=45)
        plt.xlabel(f'{prediction_column}\nActual Values', fontsize=14)
        plt.ylabel(f'{prediction_column}\nPredicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)


def analyze_data(df, data):
    """
    Perform supervised machine learning analysis based on the provided configuration.
    
    This function computes features, routes the workflow to either classification or regression analysis,
    and returns the resulting plot as a base64-encoded PNG image.
    
    Args:
        df (DataFrame): The input dataframe.
        data (dict): Configuration parameters for preprocessing, model selection, and evaluation.
    
    Returns:
        dict: A dictionary with keys 'success' and 'image_data'.
    """
    prediction_column = data['prediction_column']
    encoder_columns = data['onehot_columns']
    standar_scale_columns = data['standardscale_columns']
    columns_to_use = data['columns_to_use']

    # Drop all columns that are not in the selected columns and the prediction column
    columns_to_drop = [col for col in df.columns if col not in columns_to_use] + [prediction_column]
    X = df.drop(columns_to_drop, axis=1).copy()

    # Set common matplotlib plotting configuration
    set_common_plot_config()

    # Route analysis based on the type specified by the prediction column
    if prediction_column in encoder_columns:
        classification_analysis(df, data, X, encoder_columns.copy(), standar_scale_columns.copy())
    else:
        regression_analysis(df, data, X, encoder_columns.copy(), standar_scale_columns.copy())

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    return {
        'success': True,
        'image_data': base64.b64encode(buffer.getvalue()).decode('utf-8')
    }


def plot_actual_vs_predicted(cv_results, name, prediction_column, aux_text, test_size, random_state, scaler, ax=None):
    """
    Plot actual vs predicted values for regression analysis.
    
    Args:
        cv_results (dict): Cross-validation results containing true and predicted values.
        name (str): Model name.
        prediction_column (str): Target column name.
        aux_text (str): Additional text for the plot.
        test_size (float): Proportion of test samples.
        random_state (int): Random seed for reproducibility.
        scaler (StandardScaler): Scaler used to inverse-transform the target values.
        ax (Axes, optional): Matplotlib Axes instance. Creates a new one if not provided.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # Extract true and predicted values from all folds
    y_true = cv_results['y_true_all_folds']
    y_pred = cv_results['y_pred_all_folds']
    cv_name = cv_results['cv_name']
    n_splits = cv_results['n_splits']

    base_style = {
        'alpha': 0.7,
        's': 80,
        'edgecolors': 'w',
        'linewidth': 0.5
    }

    # Inverse-transform to original scale
    y_true_reverse = scaler.inverse_transform(y_true.reshape(-1, 1)).ravel()
    y_pred_reverse = scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()
    
    if cv_name == 'kfold':
        colors = plt.cm.get_cmap('tab10', min(n_splits, 10))
        fold_size = len(y_true_reverse) // n_splits
        for fold in range(n_splits):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < n_splits - 1 else None
            ax.scatter(y_true_reverse[start:end], y_pred_reverse[start:end],
                       color=colors(fold % 10), label=f'Fold {fold + 1}', **base_style)
    elif cv_name == 'shufflesplit':
        ax.scatter(y_true_reverse, y_pred_reverse, color='#1f77b4', label='Predictions', **base_style)
    else:
        ax.scatter(y_true_reverse, y_pred_reverse, color='#1f77b4', label='Predictions', **base_style)
        n_splits = len(y_true_reverse)

    # Plot ideal prediction line
    min_val = min(np.min(y_true_reverse), np.min(y_pred_reverse))
    max_val = max(np.max(y_true_reverse), np.max(y_pred_reverse))
    ax.plot([min_val, max_val], [min_val, max_val],
            '--', color='crimson', linewidth=2, label='Ideal')

    # Calculate and display evaluation metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    metrics_text = (f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\n"
                    f"MAE: {mae:.2f}\nR²: {r2:.2f}")
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.5, pad=4))

    method_map = {
        'kfold': f'KFold (n_splits={n_splits})',
        'shufflesplit': f'ShuffleSplit (n_splits={n_splits}, test_size={cv_results["test_size"]})',
        'loo': f'LeaveOneOut (n_samples={n_splits})'
    }
    ax.set_title(f'{prediction_column} Actual vs Predicted', fontsize=12, weight='bold', loc='center')
    ax.text(0.96, 0.04,
            f'{method_map[cv_name]}\n{name} {aux_text}\nTraining: {(100-100*test_size):.0f}% | '
            f'Test: {100*test_size:.0f}% | seed: {random_state}', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.4, pad=4))
    ax.set_xlabel(f'{prediction_column}\nActual Values', fontsize=12)
    ax.set_ylabel(f'{prediction_column}\nPredicted Values', fontsize=12)
    ax.tick_params(axis='both', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)


def plot_confusion_matrix(cv_results, classes, ax=None):
    """
    Plot a confusion matrix using aggregated predictions.
    
    Args:
        cv_results (dict): Contains either aggregated fold data or test split results.
        classes (array-like): Class labels.
        ax (Axes, optional): Matplotlib Axes. Creates a new one if not provided.
    
    Returns:
        Figure: The figure containing the confusion matrix.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    if 'y_true_all_folds' in cv_results:
        cm = confusion_matrix(cv_results['y_true_all_folds'], cv_results['y_pred_all_folds'])
    else:
        cm = confusion_matrix(cv_results['y_test'], cv_results['y_pred'])
    
    cmap = sns.color_palette("viridis", as_cmap=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=cmap, values_format='d', xticks_rotation=45, ax=ax, colorbar=False)
    im = disp.im_
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Highlight each diagonal element with a bright background
    for i in range(cm.shape[0]):
        disp.ax_.text(i, i, cm[i, i], ha='center', va='center', 
                      color='white' if cm[i, i] > cm.max() / 2 else 'black',
                      fontsize=14, fontweight='bold', 
                      bbox=dict(facecolor='#00cc00', edgecolor='none', alpha=0.7))
    disp.ax_.set_xlabel('Predicted Label', fontsize=14)
    disp.ax_.set_ylabel('True Label', fontsize=14)
    disp.ax_.grid(False)
    fig = add_watermark_fig_ax(fig)
    return fig


def perform_cross_validation(model, X, y, cv_config, random_state, classification=True):
    """
    Perform cross-validation and aggregate predictions and metrics.
    
    Args:
        model: The machine learning model.
        X (DataFrame): Features.
        y (array-like): Target variable.
        cv_config (dict): Cross-validation configuration.
        random_state (int): Random seed.
        classification (bool): Whether the problem is classification.
        
    Returns:
        dict: Aggregated CV metrics and predictions.
    """
    cv_name = cv_config['method']
    n_splits = cv_config.get('n_splits', 5)
    test_size = cv_config.get('test_size', 0.2)

    if cv_name == 'kfold':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_name == 'shufflesplit':
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    elif cv_name == 'loo':
        raise ValueError(f"LOO disabled: {cv_name}")
    else:
        raise ValueError(f"Unknown CV strategy: {cv_name}")

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1': make_scorer(f1_score, average='macro')
    } if classification else {
        'neg_mse': 'neg_mean_squared_error',
        'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))),
        'neg_mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    y_true_all_folds = []
    y_pred_all_folds = []

    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, 
        return_train_score=True, return_estimator=False
    )

    # Aggregate predictions manually for all CV folds.
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_true_all_folds.extend(y_test)
        y_pred_all_folds.extend(y_pred)

    results = {
        'cv_name': cv_name,
        'n_splits': n_splits,
        'test_size': test_size if cv_name == 'shufflesplit' else None,
        'metrics': {},
        'y_true_all_folds': np.array(y_true_all_folds),
        'y_pred_all_folds': np.array(y_pred_all_folds)
    }

    for metric in scoring.keys():
        train_key = f'train_{metric}'
        test_key = f'test_{metric}'
        if metric.startswith('neg_'):
            clean_metric = metric[4:]
            results['metrics'][clean_metric] = {
                'train': -cv_results[train_key],
                'test': -cv_results[test_key]
            }
        else:
            results['metrics'][metric] = {
                'train': cv_results[train_key],
                'test': cv_results[test_key]
            }
    return results


def plot_cv_results(cv_results, ax=None):
    """
    Plot cross-validation results with error bars for each metric.
    
    Args:
        cv_results (dict): Contains metrics for both training and testing.
        ax (Axes, optional): Matplotlib Axes instance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    metrics = cv_results['metrics']
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    shift = 0.15
    all_values = [v for m in metrics for split in ['train', 'test'] for v in metrics[m][split]]
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_padding = (y_max - y_min) * 0.1

    # Draw background boxes for each metric
    for i in range(n_metrics):
        ax.axvspan(
            i - 0.5, i + 0.5,
            facecolor='lightgray' if i % 2 == 0 else 'whitesmoke',
            alpha=0.3,
            zorder=-1
        )

    # Plot train and test points with error bars
    for i, metric in enumerate(metrics):
        ax.errorbar(
            x[i] - shift,
            np.mean(metrics[metric]['train']),
            yerr=np.std(metrics[metric]['train']),
            fmt='o',
            markersize=10,
            color='#1f77b4',
            alpha=0.9,
            capsize=5,
            capthick=2,
            label='Train' if i == 0 else None,
            zorder=10
        )
        ax.errorbar(
            x[i] + shift,
            np.mean(metrics[metric]['test']),
            yerr=np.std(metrics[metric]['test']),
            fmt='o',
            markersize=10,
            color='#ff7f0e',
            alpha=0.9,
            capsize=5,
            capthick=2,
            label='Test' if i == 0 else None,
            zorder=10
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right', fontsize=12)
    ax.set_xlim(-0.5, n_metrics - 0.5)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    for i in range(1, n_metrics):
        ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)

    ax.legend(loc='best', framealpha=1)
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    # Annotate each metric with the mean values
    for i, metric in enumerate(metrics):
        ax.text(
            x[i] - shift - 0.05,
            np.mean(metrics[metric]['train']),
            f'{np.mean(metrics[metric]["train"]):.3f}',
            ha='right',
            va='center',
            fontsize=10,
            color='#1f77b4',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
            zorder=20
        )
        ax.text(
            x[i] + shift + 0.05,
            np.mean(metrics[metric]['test']),
            f'{np.mean(metrics[metric]["test"]):.3f}',
            ha='left',
            va='center',
            fontsize=10,
            color='#ff7f0e',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
            zorder=20
        )


def get_resultados_model_numeric(X_train, X_test, y_train, y_test, model):
    """
    Evaluate a numeric prediction model.
    
    Args:
        X_train, X_test, y_train, y_test: train-test split data.
        model: Regression model.
    
    Returns:
        tuple: (mse, rmse, mae, r2) and predictions.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mse, rmse, mae, r2), y_pred


def get_resultados_model_categorical(X_train, X_test, y_train, y_test, model, opt):
    """
    Evaluate a classification model.
    
    Args:
        X_train, X_test, y_train, y_test: train-test split data.
        model: Classification model.
        opt (bool): Option for using average='macro' in metrics for multi-class problems.
    
    Returns:
        tuple: (accuracy, precision, recall, f1) and predictions.
    """
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


def add_watermark_fig(fig, xs=[0.05, 0.5]):
    """
    Add a diagonal watermark grid to the figure.
    
    Args:
        fig (Figure): The matplotlib figure.
        xs (list): X positions for watermark texts.
    
    Returns:
        Figure: The figure with watermark.
    """
    watermark_text = "github.com/rdominguez89"
    watermark_color = 'lightgray'
    watermark_alpha = 0.3
    watermark_fontsize = 20
    watermark_angle = 30
    for x in xs:
        for y in [0.1, 0.3, 0.6]:
            if not (0.45 < x < 0.55 and 0.45 < y < 0.55):
                fig.text(x, y, watermark_text, fontsize=watermark_fontsize,
                         color=watermark_color, alpha=watermark_alpha,
                         rotation=watermark_angle, zorder=0, transform=fig.transFigure)
    return fig


def add_watermark_fig_ax(fig):
    """
    Add watermark to the entire figure background.
    
    Args:
        fig (Figure): The matplotlib figure.
    
    Returns:
        Figure: The watermarked figure.
    """
    watermark_text = "github.com/rdominguez89"
    watermark_color = 'lightgray'
    watermark_alpha = 0.3
    watermark_fontsize = 15
    watermark_angle = 30
    for x in [0.15, 0.55]:
        for y in [0.2, 0.4, 0.6]:
            fig.text(x, y, watermark_text, fontsize=watermark_fontsize, color=watermark_color,
                     alpha=watermark_alpha, rotation=watermark_angle, zorder=0,
                     transform=fig.transFigure, clip_on=False)
    return fig