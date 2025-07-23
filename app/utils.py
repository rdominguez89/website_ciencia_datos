import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import io
import base64
import os
from flask import send_file
from .analyze_supervised import add_watermark_fig, add_watermark_fig_ax


matplotlib.use('Agg')

def allowed_file(filename):
    """Check if the file has an allowed extension (CSV)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def load_dataframe(file):
    """Load and validate CSV file with row and column limits."""
    df = pd.read_csv(file)
    if len(df) > 1000:
        raise ValueError('File exceeds 1000 row limit')
    if len(df.columns) > 10:
        raise ValueError('File exceeds 10 column limit')
    return df

def get_stats_summary(df):
    """Generate statistical summary and data type information for the dataset."""
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
    
    dtype_info = [
        {
            'column': col,
            'type': str(df[col].dtype),
            'unique_values': int(df[col].nunique()),
            'null_values': int(null_counts[col])
        } for col in df.columns
    ]
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    num_stats = df[numerical_cols].describe().round(2).to_dict() if not df.empty else {}
    
    return {
        'success': True,
        'data_head': df.head(5).to_html(classes='table table-striped'),
        'null_counts': null_counts.to_frame('null_count').to_html(classes='table table-striped'),
        'duplicated_rows': df[df.duplicated(keep=False)].to_html(classes='table table-striped', index=False) if stats['duplicates'] > 0 else "No duplicated rows found",
        'stats': stats,
        'dtype_info': dtype_info,
        'num_stats': num_stats,
        'columns': list(df.columns),
        'column_order': list(numerical_cols),
        'df': df.where(pd.notnull(df), '-').to_dict(orient='records')
    }

def clean_dataframe(df, remove_nulls=False, remove_duplicates=False):
    """Clean the dataset by removing nulls and duplicates as specified."""
    df = df.copy()
    if remove_nulls:
        df = df.dropna()
    if remove_duplicates:
        df = df.drop_duplicates()
    return df

def create_visualizations_function(df):
    """Generate boxplots for numerical columns and distribution plots for categorical columns with enhanced styling."""
    visualizations = {'boxplots': [], 'distributions': [], 'outliers_data': {}}
    
    # Set modern style with seaborn
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-bright')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })
    
    # Numerical columns - Boxplots
    num_cols = df.select_dtypes(include=['number']).columns
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 6))
        fig = add_watermark_fig(fig)
        sns.boxplot(data=df, y=col, ax=ax, color='#1f77b4', boxprops=dict(edgecolor='black', linewidth=1.5),
                    whiskerprops=dict(color='black', linewidth=1.5), flierprops=dict(marker='o', color='crimson', 
                    markersize=8, alpha=0.6))
        
        # Extract outliers
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = df[col][(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
        outlier_rows = df[df[col].isin(outliers)].reset_index()
        
        # Add title and labels
        ax.set_title(f'Boxplot of {col}', pad=20, fontsize=16, weight='bold')
        ax.set_ylabel(col, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        # Convert plot to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        visualizations['boxplots'].append({
            'title': f'Boxplot of {col}',
            'image_data': img_str,
            'type': 'boxplot',
            'outliers_count': len(outliers)
        })
        visualizations['outliers_data'][col] = {
            'outlier_values': outliers.tolist(),
            'outlier_rows': outlier_rows.to_dict('records'),
            'total_outliers': len(outliers)
        }
    return visualizations
    # Categorical and numerical columns - Distributions
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        if df[col].nunique() < 10:
            sns.countplot(x=df[col], order=df[col].value_counts().sort_index().index, ax=ax)
            plot_type = 'countplot'
        else:
            sns.histplot(df[col], kde=True, ax=ax)
            plot_type = 'histplot'
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        visualizations['distributions'].append({
            'title': f'Distribution of {col}',
            'image_data': img_str,
            'type': plot_type
        })
    
    return visualizations

def remove_outliers_function(df, outliers):
    """Remove specified outliers from the dataset."""
    outliers_to_remove = {(o['column'], o['index']) for o in outliers}
    filtered_df = df[~df.apply(
        lambda row: any((col, row.name) in outliers_to_remove for col in df.columns),
        axis=1
    )].reset_index(drop=True)
    
    return {
        'success': True,
        'df': filtered_df.where(pd.notnull(filtered_df), '-').to_dict(orient='records'),
        'removed_count': len(df) - len(filtered_df)
    }

def create_correlation_plots_function(df):
    """Generate pairplot and correlation heatmap for numerical features with enhanced styling."""
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if len(numerical_cols) < 2:
        raise ValueError('Need at least 2 numerical columns for correlation plots')
    
    # Set modern style with seaborn
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8-bright')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 100
    })
    
    # Pairplot
    pairplot = sns.pairplot(df[numerical_cols], 
                           plot_kws={'alpha': 0.6, 's': 80, 'color': '#1f77b4', 'edgecolor': 'w', 'linewidth': 0.5},
                           diag_kws={'color': '#1f77b4', 'alpha': 0.8})
    pairplot.fig.suptitle('Pairplot of Numerical Features', y=1.02, fontsize=16, weight='bold')
    for ax in pairplot.axes.flatten():
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        fig = add_watermark_fig_ax(pairplot.fig)
        
    buffer = io.BytesIO()
    pairplot.savefig(buffer, format='png', bbox_inches='tight', dpi=80)
    buffer.seek(0)
    pairplot_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(pairplot.fig)
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    fig = add_watermark_fig(fig)
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', center=0, fmt='.2f', 
                annot_kws={'size': 12, 'weight': 'bold'}, 
                square=True, cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_title('Correlation Heatmap', pad=20, fontsize=16, weight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    heatmap_buffer = io.BytesIO()
    fig.savefig(heatmap_buffer, format='png', bbox_inches='tight', dpi=100)
    heatmap_buffer.seek(0)
    heatmap_img = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return {
        'success': True,
        'plots': [
            {'title': 'Pairplot of Numerical Features', 'image_data': pairplot_img},
            {'title': 'Correlation Heatmap', 'image_data': heatmap_img}
        ]
    }

def save_dataframe(df, selected_columns, filename):
    """Save the dataset as a CSV file and return it for download."""
    df = df[selected_columns]
    temp_path = f"/tmp/{filename}.csv"
    df.to_csv(temp_path, index=False)
    try:
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=f"{filename}.csv",
            mimetype='text/csv'
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)