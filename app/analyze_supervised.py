import dis
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit, LeaveOneOut, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import io
import base64
from .models import get_numeric_model, get_categorical_model


def analyze_data(df, data):
    """Perform machine learning analysis with enhanced confusion matrix highlighting diagonal."""
    prediction_column = data['prediction_column']
    encoder_columns = data['onehot_columns']
    standar_scale_columns = data['standardscale_columns']
    test_size = data['test_size']
    random_state = data['random_seed']
    name = data['model']
    inputs = [encoder_columns, standar_scale_columns, random_state, data['model_params']]
    cv_config = data['cross_validation'] if len(data['cross_validation']) > 0 else None

    X = df.drop(prediction_column, axis=1).copy()
    
    # Set modern style with seaborn
    # sns.set_style("whitegrid")
    # plt.style.use('seaborn-v0_8-bright')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })

    cat_opt = False
    if prediction_column in encoder_columns:
        cat_opt = True
        opt = False
        if prediction_column in encoder_columns:
            encoder_columns.remove(prediction_column)
        y_check = set(df[prediction_column])
        opt = len(y_check) > 2
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[prediction_column])
        aux_text = get_aux_text(inputs[3],encoder_columns,standar_scale_columns, prediction_column,X.columns)
        model = get_categorical_model(name, inputs)
        if cv_config is not None:
            cv_results = perform_cross_validation(
                            model, X, y, cv_config, random_state, classification=True)
            # Create a figure with subplots
            fig = plt.figure(figsize=(5, 8))
            gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])  
            
            # First subplot for confusion matrix
            ax1 = plt.subplot(gs[0])
            plot_confusion_matrix(cv_results, label_encoder.classes_, ax=ax1)
           
            ax1.set_title(f'{prediction_column} {name} (Cross-Validated)\n{aux_text}', fontsize=14, weight='bold')#\nAccuracy: {mean_test_accuracy:.2f} | Precision: {mean_test_precision:.2f} | Recall: {mean_test_recall:.2f} | F1: {mean_test_f1:.2f}', fontsize=14, weight='bold')
            #plt.title(f'{prediction_column} {name}\n{aux_text}\nAccuracy: {disc[0]:.2f} | Precision: {disc[1]:.2f} | Recall: {disc[2]:.2f} | F1: {disc[3]:.2f}', fontsize=14, weight='bold')
            # Second subplot for CV results
            ax2 = plt.subplot(gs[1])
            plot_cv_results(cv_results, ax=ax2)
            plt.tight_layout()
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
            
            disc, y_pred = get_resultados_model_categorical(X_train, X_test, y_train, y_test, model, opt)
            cv_results = {'y_test': y_test, 'y_pred': y_pred}
            plot_confusion_matrix(cv_results, label_encoder.classes_)
            
            plt.title(f'{prediction_column} Confusion Matrix\n{name} {aux_text}\nAccuracy: {disc[0]:.2f} | Precision: {disc[1]:.2f} | Recall: {disc[2]:.2f} | F1: {disc[3]:.2f}', 
                    pad=20, fontsize=14, weight='bold')
            
    else:
        model = get_numeric_model(name, inputs)
        y = df[prediction_column].copy()
        if prediction_column in encoder_columns: encoder_columns.remove(prediction_column)
        if prediction_column in standar_scale_columns: standar_scale_columns.remove(prediction_column)
        aux_text = get_aux_text(inputs[3],encoder_columns,standar_scale_columns, prediction_column,X.columns)
        if cv_config is not None:
            # cv_results = perform_cross_validation(model, X, y, cv_config, random_state, classification=False)
            #  # Create a figure with subplots
            # fig = plt.figure(figsize=(5, 8))
            # gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
            # # First subplot for actual vs predicted
            # ax1 = plt.subplot(gs[0])
            # plot_actual_vs_predicted(cv_results, name, prediction_column, aux_text, ax=ax1)
            # # Second subplot for CV results
            # ax2 = plt.subplot(gs[1])
            # plot_cv_results(cv_results, ax=ax2)
            # fig = add_watermark_fig(fig,xs=[0.2])

            cv_results = perform_cross_validation(model, X, y, cv_config, random_state, classification=False)
             # Create a figure with subplots
            fig = plt.figure(figsize=(6,6))
            ax = plt.gca()
            plot_actual_vs_predicted(cv_results, name, prediction_column, aux_text, ax=ax)
            fig = add_watermark_fig(fig,xs=[0.2])

        else:
            fig = plt.figure(figsize=(6, 6))
            fig = add_watermark_fig(fig,xs=[0.3])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            disc, y_pred = get_resultados_model_numeric(X_train, X_test, y_train, y_test, model)
            # Enhanced scatter plot
            plt.scatter(y_test, y_pred, alpha=0.6, s=80, c='#1f77b4', edgecolors='w', linewidth=0.5)
            plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='crimson', linewidth=2)
            # plt.title(f'{prediction_column} Actual vs Predicted\n{name} {aux_text}\nMSE: {disc[0]:.2f} | RMSE: {disc[1]:.2f} | MAE: {disc[2]:.2f} | R²: {disc[3]:.2f}', 
            #         pad=20, fontsize=16, weight='bold')
            plt.title(f'{prediction_column} Actual vs Predicted\n{name} {aux_text}', pad=20, fontsize=16, weight='bold')
            ax = plt.gca()
            metrics_text = (f"MSE: {disc[0]:.2f}\n"
                f"RMSE: {disc[1]:.2f}\n"
                f"MAE: {disc[2]:.2f}\n"
                f"R²: {disc[3]:.2f}")
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8))
            ax.tick_params(axis='both', rotation=45)
            plt.xlabel(f'{prediction_column}\nActual Values', fontsize=14)
            plt.ylabel(f'{prediction_column}\nPredicted Values', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    buffer = io.BytesIO()
    # if cat_opt:
    #     plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    # else:
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    #plt.close(fig)
    
    return {
        'success': True,
        'image_data': base64.b64encode(buffer.getvalue()).decode('utf-8')
    }

def plot_actual_vs_predicted(cv_results, name, prediction_column, aux_text, ax=None):
    """Plot actual vs predicted values for regression with CV-method-specific styling."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure
    
    # Extract values
    y_true = cv_results['y_true_all_folds']
    y_pred = cv_results['y_pred_all_folds']
    cv_name = cv_results['cv_name']
    n_splits = cv_results['n_splits']

    # Base styling
    base_style = {
        'alpha': 0.7,
        's': 80,
        'edgecolors': 'w',
        'linewidth': 0.5
    }

    # Handle each CV method differently
    if cv_name == 'kfold':
        # KFold - color by fold
        colors = plt.cm.get_cmap('tab10', min(n_splits, 10))  # Max 10 colors
        fold_size = len(y_true) // n_splits
        
        for fold in range(n_splits):
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < n_splits - 1 else None
            ax.scatter(y_true[start:end], y_pred[start:end],
                      color=colors(fold % 10),  # Cycle if >10 folds
                      label=f'Fold {fold + 1}',
                      **base_style)
        
    elif cv_name == 'shufflesplit':
        # ShuffleSplit - single color (points may overlap across folds)
        ax.scatter(y_true, y_pred, color='#1f77b4', label='Predictions', **base_style)
        
    else:  # LOO
        # LeaveOneOut - single color (each point is its own fold)
        ax.scatter(y_true, y_pred, color='#1f77b4', label='Predictions', **base_style)
        n_splits = len(y_true)  # Override to show sample count

    # Ideal prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val],
            '--', color='crimson', linewidth=2, label='Ideal')

    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Metrics box
    metrics_text = (f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\n"
                   f"MAE: {mae:.2f}\nR²: {r2:.2f}")
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0.8, pad=4))

    # Titles and labels
    method_map = {
        'kfold': f'KFold (n_splits={n_splits})',
        'shufflesplit': f'ShuffleSplit (n_splits={n_splits}, test_size={cv_results["test_size"]})',
        'loo': f'LeaveOneOut (n_samples={n_splits})'
    }
    ax.set_title(f'{prediction_column} Actual vs Predicted\n{method_map[cv_name]}\n{name} {aux_text}',
                pad=20, fontsize=12, weight='bold')
    ax.set_xlabel(f'{prediction_column}\nActual Values', fontsize=12)
    ax.set_ylabel(f'{prediction_column}\nPredicted Values', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, linestyle='--', alpha=0.7)

    return 

def plot_confusion_matrix(cv_results, classes, ax=None):
    """Plot confusion matrix using pre-aggregated predictions."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    if 'y_true_all_folds' in cv_results:
        # Compute final aggregated CM
        cm = confusion_matrix(
            cv_results['y_true_all_folds'],
            cv_results['y_pred_all_folds']
        )
    else:
        cm = confusion_matrix(
            cv_results['y_test'],
            cv_results['y_pred']
        )
    cmap = sns.color_palette("viridis", as_cmap=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    #disp.plot(cmap=cmap, values_format='d', xticks_rotation=45)
    
    disp.plot(cmap=cmap, values_format='d', xticks_rotation=45, ax=ax, colorbar=False)
    #Get the current image from the plot
    im = disp.im_
    # Create a colorbar with the same height as the confusion matrix
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Highlight diagonal with a different color (e.g., bright green)
    for i in range(cm.shape[0]):
        disp.ax_.text(i, i, cm[i, i], ha='center', va='center', 
                    color='white' if cm[i, i] > cm.max() / 2 else 'black',
                    fontsize=14, fontweight='bold', 
                    bbox=dict(facecolor='#00cc00', edgecolor='none', alpha=0.7))
    
    disp.ax_.set_xlabel('Predicted Label', fontsize=14)
    disp.ax_.set_ylabel('True Label', fontsize=14)
    disp.ax_.grid(False)
    fig = disp.ax_.figure  # This is crucial!
    fig = add_watermark_fig_ax(fig)
    
    return fig

def perform_cross_validation(model, X, y, cv_config, random_state, classification=True):
    """Perform cross-validation and return metrics + aggregated predictions."""
    cv_name = cv_config['method']
    n_splits = cv_config.get('n_splits', 5)
    test_size = cv_config.get('test_size', 0.2)

    # Create cross-validator
    if cv_name == 'kfold':
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_name == 'shufflesplit':
        cv = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    elif cv_name == 'loo':
        raise ValueError(f"LOO disabled: {cv_name}")
        cv = LeaveOneOut()
    else:
        raise ValueError(f"Unknown CV strategy: {cv_name}")

    # Define scoring metrics
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

    # Store aggregated predictions
    y_true_all_folds = []
    y_pred_all_folds = []
    #confusion_matrices = []

    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y, cv=cv, scoring=scoring, 
        return_train_score=True, 
        return_estimator=False
    )

    # Manually collect predictions for aggregation
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_true_all_folds.extend(y_test)
        y_pred_all_folds.extend(y_pred)
        #if classification: confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Process metrics
    results = {
        'cv_name': cv_name,
        'n_splits': n_splits,
        'test_size': test_size if cv_name == 'shufflesplit' else None,
        'metrics': {},
        'y_true_all_folds': np.array(y_true_all_folds),
        'y_pred_all_folds': np.array(y_pred_all_folds),
        #'confusion_matrices': confusion_matrices  # Optional
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
    """Plot CV results with metrics enclosed in vertical boxes."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    metrics = cv_results['metrics']
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    shift = 0.15  # Horizontal offset between train/test points
    
    # Get y-limits for consistent boxes
    all_values = [v for m in metrics for split in ['train', 'test'] for v in metrics[m][split]]
    y_min, y_max = np.min(all_values), np.max(all_values)
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    
    # Draw metric boxes first (so points appear on top)
    for i in range(n_metrics):
        ax.axvspan(
            i - 0.5, i + 0.5,  # Full width for each metric
            facecolor='lightgray' if i % 2 == 0 else 'whitesmoke',
            alpha=0.3,
            zorder=-1  # Send to background
        )
    
    # Plot points and error bars
    for i, metric in enumerate(metrics):
        # Train data
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
            zorder=10  # Ensure points are on top
        )
        
        # Test data
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
    
    # Customize axes
    ax.set_xticks(x)
    ax.set_xticklabels(
        [m.replace('_', ' ').title() for m in metrics],
        rotation=45,
        ha='right',
        fontsize=12
    )
    ax.set_xlim(-0.5, n_metrics - 0.5)  # Ensure full width usage
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    # Add dividing lines between metrics
    for i in range(1, n_metrics):
        ax.axvline(i - 0.5, color='gray', linestyle=':', alpha=0.5, zorder=1)
    
    # Add legend and labels
    ax.legend(loc='best', framealpha=1)
    ax.set_ylabel('Score', fontsize=12)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Add metric values
    for i, metric in enumerate(metrics):
        # Train value
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
        
        # Test value
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
    
    return

def get_resultados_model_numeric(X_train, X_test, y_train, y_test, model):
    """Evaluate numeric prediction model and return performance metrics."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return (mse, rmse, mae, r2), y_pred

def get_resultados_model_categorical(X_train, X_test, y_train, y_test, model, opt):
    """Evaluate categorical prediction model and return performance metrics."""
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

def add_watermark_fig(fig, xs=[0.05, 0.5]):
    # Add subtle diagonal watermark
    watermark_text = "github.com/rdominguez89"
    watermark_color = 'lightgray'
    watermark_alpha = 0.3  # Very subtle
    watermark_fontsize = 20
    watermark_angle = 30  # Diagonal angle
    
    # Repeat the watermark in a grid pattern (adjust spacing as needed)
    for x in xs:
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
    for x in [0.15, 0.55]:
        for y in [0.2, 0.4, 0.6]:
            fig.text(x,y, watermark_text,
                fontsize=watermark_fontsize, color=watermark_color,
                alpha=watermark_alpha, rotation=watermark_angle,
                zorder=0,
                transform=fig.transFigure, clip_on=False)
    
    return fig