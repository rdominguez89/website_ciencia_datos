# filepath: /media/raul3/M2SDD/website_ciencia_datos_dev/app/analyze_inference.py
from scipy import stats
import numpy as np
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson, uniform, norm, multinomial

def perform_one_sample_ttest(data, population_mean, significance_level):
    """
    Perform a one-sample t-test and generate a plot.

    Args:
        data (list): List of numeric data.
        population_mean (float): Hypothesized population mean.
        significance_level (float): Significance level (alpha).

    Returns:
        dict: Results including test statistic, p-value, and plot.
    """
    try:
        data = np.array(data, dtype=float)
        t_statistic, p_value = stats.ttest_1samp(data, population_mean)

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=15, alpha=0.7, label='Sample Data')
        plt.axvline(data.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Sample Mean: {data.mean():.2f}')
        plt.axvline(population_mean, color='green', linestyle='dashed', linewidth=2, label=f'Population Mean: {population_mean:.2f}')
        plt.title('One-Sample t-Test')
        plt.xlabel('Data Values')
        plt.ylabel('Frequency')
        plt.legend()

        # Save plot to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        # Determine if the null hypothesis is rejected
        if p_value < significance_level:
            message = f"The null hypothesis is rejected. p-value: {p_value:.3f}"
        else:
            message = f"The null hypothesis is not rejected. p-value: {p_value:.3f}"

        return {
            'success': True,
            'statistic': t_statistic,
            'p_value': p_value,
            'message': message,
            'plot': plot_data
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e),
            'plot': None
        }

def perform_correlation(data, column1, column2, method, produce_plot=False):
    """
    Perform a correlation test based on the specified method.
    
    Supported methods:
        - 'pearson'
        - 'spearman'
        - 'kendall'
        - 'pointbiserial'
        - 'cramer'
    
    Args:
        data (dict): Data object with columns.
        column1 (str): Name of the first column.
        column2 (str): Name of the second column.
        method (str): The correlation method.
        
    Returns:
        dict: Result with the correlation statistic, p-value, message and a plot (if applicable).
    """
    try:
        col1_data = np.array(data[column1], dtype=float)
        col2_data = np.array(data[column2], dtype=float)
        
        # Choose the correlation method
        if method == 'pearson':
            corr, p_value = stats.pearsonr(col1_data, col2_data)
            test_name = "Pearson Correlation"
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(col1_data, col2_data)
            test_name = "Spearman’s Rank Correlation"
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(col1_data, col2_data)
            test_name = "Kendall’s Tau"
        elif method == 'pointbiserial':
            # For point-biserial, one variable must be dichotomous.
            corr, p_value = stats.pointbiserialr(col1_data, col2_data)
            test_name = "Point-Biserial Correlation"
        elif method == 'cramer':
            # Cramér's V is computed from a contingency table. Here we assume categorical data.
            contingency_table = pd.crosstab(pd.Series(data[column1]), pd.Series(data[column2]))
            chi2 = stats.chi2_contingency(contingency_table)[0]
            n = contingency_table.to_numpy().sum()
            min_dim = min(contingency_table.shape) - 1
            corr = np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else np.nan
            p_value = np.nan  # p-value is not typically computed with Cramér's V
            test_name = "Cramér’s V"
        else:
            raise ValueError("Unsupported correlation method")
        
        # Plotting (scatter plot for continuous methods; contingency plot for Cramér's V)
        # if method == 'cramer':
        #     contingency_table.plot(kind='bar', stacked=True)
        #     plt.xlabel(column1)
        #     plt.ylabel('Count')
        #     plt.title(f'{test_name} between {column1} and {column2}')
        # else:
        plot_data = None
        if method != 'cramer' and produce_plot:
            plt.figure(figsize=(5,5))  # Let matplotlib handle the default size
            plt.scatter(col1_data, col2_data, alpha=0.7, edgecolors='w', linewidth=0.5)
            plt.xlabel(column1, fontsize=10)
            plt.ylabel(column2, fontsize=10)
            plt.title(f'{test_name}: {column1} vs {column2}', fontsize=12, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)  # Add a subtle grid
            plt.xticks(fontsize=8)  # Adjust tick label size
            plt.yticks(fontsize=8)
            plt.tight_layout()  # Ensure labels don't overlap
        
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

        message = f"{test_name}: correlation = {corr:.3f}" + (f", p-value = {p_value:.3e}" if not np.isnan(p_value) else "")        
        if np.isnan(p_value): p_value = 0
        
        return {
            'success': True,
            'correlation': corr,
            'p_value': p_value,
            'message': message,
            'plot': plot_data
        }
    except Exception as e:
        return {
            'success': False,
            'message': str(e),
            'plot': None
        }

def analyze_distribution(outcomes, distribution, params):
    """Analyze a probability distribution and generate a plot."""
    try:
        # # Validate that probabilities sum to 1 (within tolerance)
        # if not np.isclose(sum(probabilities), 1.0, atol=1e-2):
        #     raise ValueError("Probabilities must sum to 1.0 for a valid multinomial distribution.")
        
        if distribution == 'multinomial':
            # Extract successes and probabilities from outcomes.
            # outcomes is a list of dicts including 'name', 'probability' and 'successes'
            success_counts = [int(item['successes']) for item in outcomes]
            probabilities = [float(item['probability']) for item in outcomes]
            
            n_trials = sum(success_counts) #Future implementation int(params['trials'])
            sum_successes = sum(success_counts)
 
            # Calculate remaining trials (if any)
            remaining_trials = n_trials - sum_successes
            
            if remaining_trials > 0:
                if not np.isclose(sum(probabilities), 1.0, atol=1e-2):
                    raise ValueError("Number of trials greater than sum of successes, not implemented yet.")
        
                # Create a "remainder" category for unspecified outcomes
                success_counts.append(remaining_trials)
                probabilities.append(1 - sum(probabilities))  # Should be 0 if probabilities sum to 1
            elif sum(probabilities) != 1.0: 
                 success_counts.append(0)
                 probabilities.append(1- sum(probabilities))  # Should be 0 if probabilities sum to 1
            
            # Calculate probability
            p_value = multinomial.pmf(success_counts, n_trials, probabilities)
            
            # Format output message
            # Format the outcomes list in a natural language way
            outcomes_list = " and ".join([
                f"{item['successes']} {item['name']}" 
                for item in outcomes
            ])

            # Add remaining trials if they exist
            if remaining_trials > 0:
                outcomes_list += f" and {remaining_trials} other"

            message = (
                f"Multinomial Distribution: The probability of observing {outcomes_list} "
                f"in {n_trials} trial{'s' if n_trials != 1 else ''} is {100*p_value:.2f}%."
            )
                    # (Optional) You may add plotting code if desired.
            return {
                'success': True,
                'message': message,
                'plot': None
            }
        # Existing branches for other distributions:
        if distribution == 'binomial':
            n = int(params['trials'])
            p = float(outcomes[0]['probability'])
            x = int(params['successes'])
            y = binom.pmf(x, n, p)
            message = f"Binomial Distribution: The probability to get {x} success{'es' if x != 1 else ''} in {n} trials is {(100*y):.2f}%."
            return {'success': True, 'message': message, 'plot': None}
        elif distribution == 'poisson':
            lam = float(params['lambda'])
            x = float(params['timeFrequency'])
            y = poisson.pmf(x, lam)
            message = f"Poisson Distribution: The probability to get {x} event{'s' if x != 1 else ''} in the period is {(100*y):.2f}%."
            return {'success': True, 'message': message, 'plot': None}
        elif distribution == 'uniform':
            b = float(params['total'])
            a = float(params['favorable'])
            x = np.linspace(a, b, 100)
            y = uniform.pdf(x, loc=a, scale=b - a)
            message = f"Uniform Distribution analyzed for {outcomes[0]['name']} from {a} to {b} is {(a/b):.2f}%."
            # (Optional plotting code)
            return {'success': True, 'message': message, 'plot': None}
        elif distribution == 'normal':
            mu = float(params['mean'])
            sigma = float(params['stdDev'])
            if params['condition'] == 'range':
                x1 = float(params['lowerBound'])
                x2 = float(params['upperBound'])
                y1 = norm.cdf(x1, loc=mu, scale=sigma)
                y2 = norm.cdf(x2, loc=mu, scale=sigma)
                if x1 < 0: y1 = 1 - y1  # Adjust for left tail
                if x2 < 0: y2 = 1 - y2  # Adjust for left tail
                y = abs(y2 - y1)
                message = f"Normal Distribution: The probability of observing {outcomes[0]['name']} between {x1} and {x2} is {(100*y):.2f}%."
            else:
                x = float(params['seekValue'])
                y = norm.cdf(x, loc=mu, scale=sigma)
                if params['condition'] == '<=':
                    message = f"Normal Distribution: The probability of observing {outcomes[0]['name']} less than or equal to {params['seekValue']} is {(100*y):.2f}%."
                elif params['condition'] == '>=':
                    message = f"Normal Distribution: The probability of observing {outcomes[0]['name']} greater than or equal to {params['seekValue']} is {(100*(1-y)):.2f}%."
            return {'success': True, 'message': message, 'plot': None}
        else:
            raise ValueError("Invalid distribution type")
    except Exception as e:
        return {'success': False, 'message': str(e), 'plot': None}