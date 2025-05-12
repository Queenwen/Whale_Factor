import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import ttest_1samp

class TradingStrategyAgent:
    """Trading strategy agent class for executing multifactor strategy analysis"""
    
    def __init__(self, output_dir=None):
        """Initialize trading strategy agent"""
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def perform_lagged_regression(self, data_df, common_tokens, factor_suffix, return_suffix, lag=1, 
                                 plot_results=True, plot_top_n=20, output_dir=None, config_label="default_config"):
        """Perform lagged regression analysis for each token to validate factor's predictive power"""
        if output_dir is None:
            output_dir = self.output_dir
            
        print(f"\nStarting lagged regression analysis: config='{config_label}', factor_suffix='{factor_suffix}'")
        results_list = []

        for token in common_tokens:
            factor_col = f"{token}{factor_suffix}"
            return_col = f"{token}{return_suffix}"

            if factor_col not in data_df.columns:
                continue
            if return_col not in data_df.columns:
                continue

            # Prepare Y (future returns) and X (current factor)
            y = data_df[return_col].shift(-lag)
            X = data_df[factor_col]

            # Merge and remove NaN values
            reg_df = pd.DataFrame({'y': y, 'X': X}).dropna()
            
            if len(reg_df) < 30:  # Need sufficient data points for meaningful regression
                continue

            X_with_const = sm.add_constant(reg_df['X'])
            
            try:
                model = sm.OLS(reg_df['y'], X_with_const)
                result = model.fit()
                
                coefficient = result.params.get('X', np.nan)
                p_value = result.pvalues.get('X', np.nan)
                t_value = result.tvalues.get('X', np.nan)
                r_squared = result.rsquared
                n_obs = result.nobs
                
                results_list.append({
                    'token': token,
                    'factor_suffix': factor_suffix,
                    'return_suffix': return_suffix,
                    'lag_for_y_shift': lag,
                    'config_label': config_label,
                    'coefficient': coefficient,
                    'p_value': p_value,
                    't_value': t_value,
                    'r_squared': r_squared,
                    'n_observations': n_obs,
                    'significant': p_value < 0.05
                })
                
            except Exception as e:
                print(f"Error: Failed to perform OLS regression for token {token}: {e}")
                continue
        
        if not results_list:
            print(f"Warning: No valid regression results generated for any token.")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results_list)
        
        # Sort by absolute coefficient value
        results_df['abs_coefficient'] = results_df['coefficient'].abs()
        results_df = results_df.sort_values('abs_coefficient', ascending=False)
        
        # Print results summary
        print(f"\nRegression results summary (sorted by absolute coefficient):")
        print(f"Total tokens analyzed: {len(results_df)}")
        print(f"Significant results (p < 0.05): {results_df['significant'].sum()} tokens")
        print(f"Average coefficient: {results_df['coefficient'].mean():.4f}")
        print(f"Average R²: {results_df['r_squared'].mean():.4f}")
        
        # Save results to CSV
        if output_dir:
            csv_filename = os.path.join(output_dir, f"regression_results_{config_label}.csv")
            results_df.to_csv(csv_filename, index=False)
            print(f"Regression results saved to: {csv_filename}")
        
        # Plot regression results
        if plot_results and output_dir and not results_df.empty:
            # Limit number of tokens to plot
            plot_df = results_df.head(plot_top_n) if len(results_df) > plot_top_n else results_df
            
            # Plot coefficient bar chart
            plt.figure(figsize=(14, 8))
            bars = plt.bar(plot_df['token'], plot_df['coefficient'], color=[
                'green' if x < 0.05 else 'gray' for x in plot_df['p_value']
            ])
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.title(f'Factor Coefficients ({config_label})\nGreen = Statistically Significant (p < 0.05)')
            plt.xlabel('Token')
            plt.ylabel('Coefficient Value')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # Save plot
            plot_filename = os.path.join(output_dir, f"regression_coefficients_{config_label}.png")
            plt.savefig(plot_filename)
            plt.close()
            print(f"Coefficient plot saved to: {plot_filename}")
            
            # Plot R² bar chart
            plt.figure(figsize=(14, 8))
            plt.bar(plot_df['token'], plot_df['r_squared'], color='skyblue')
            plt.title(f'Regression R² Values ({config_label})')
            plt.xlabel('Token')
            plt.ylabel('R²')
            plt.xticks(rotation=90)
            plt.tight_layout()
            
            # Save plot
            r2_plot_filename = os.path.join(output_dir, f"regression_r2_{config_label}.png")
            plt.savefig(r2_plot_filename)
            plt.close()
            print(f"R² plot saved to: {r2_plot_filename}")
        
        return results_df
    
    def perform_pooled_lagged_regression(self, data_df, common_tokens, factor_suffix, return_suffix, lag, config_label="pooled_config"):
        """Perform lagged regression analysis on pooled data for all tokens"""
        print(f"\nStarting pooled lagged regression analysis: config='{config_label}', factor_suffix='{factor_suffix}', return_suffix='{return_suffix}', lag_for_y_shift='{lag}'")
        
        all_y_data = []
        all_x_data = []

        for token in common_tokens:
            factor_col = f"{token}{factor_suffix}"
            return_col = f"{token}{return_suffix}"

            if factor_col not in data_df.columns or return_col not in data_df.columns:
                continue

            y_token = data_df[return_col].shift(-lag)  # Y is future return
            x_token = data_df[factor_col]              # X is current factor
            
            all_y_data.append(y_token)
            all_x_data.append(x_token)

        if not all_x_data or not all_y_data:
            print("Warning: No data collected for pooled regression.")
            return None

        pooled_y = pd.concat(all_y_data)
        pooled_x = pd.concat(all_x_data)

        pooled_df = pd.DataFrame({'y': pooled_y, 'X': pooled_x}).dropna()

        if len(pooled_df) < 50:  # Need sufficient data points for meaningful pooled regression
            print(f"Warning: Insufficient pooled data points ({len(pooled_df)}) for effective regression.")
            return None

        X_with_const = sm.add_constant(pooled_df['X'])

        try:
            model = sm.OLS(pooled_df['y'], X_with_const)
            result = model.fit()

            coefficient = result.params.get('X', np.nan)
            p_value = result.pvalues.get('X', np.nan)
            t_value = result.tvalues.get('X', np.nan)
            r_squared = result.rsquared
            n_obs = result.nobs

            pooled_result = {
                'factor_suffix': factor_suffix,
                'return_suffix': return_suffix,
                'lag_for_y_shift': lag,
                'config_label': config_label,
                'coefficient': coefficient,
                'p_value': p_value,
                't_value': t_value,
                'r_squared': r_squared,
                'n_observations': n_obs
            }
            print(f"Pooled regression results: Coeff={coefficient:.4f}, P-value={p_value:.4f}, T-value={t_value:.2f}, R2={r_squared:.3f}, N={n_obs}")
            return pooled_result
            
        except Exception as e:
            print(f"Error: Pooled OLS regression failed: {e}")
            return None
    
    def plot_pooled_regression_summary_table(self, df_summary, factor_configs, output_dir=None):
        """Plot summary table for pooled regression results"""
        if output_dir is None:
            output_dir = self.output_dir
            
        if df_summary.empty:
            print("Warning: Summary data is empty, cannot plot table.")
            return
            
        # Create pivot table with factors as rows and regression configs as columns
        pivot_coef = df_summary.pivot_table(
            index='factor_suffix', 
            columns='config_label', 
            values='coefficient',
            aggfunc='first'
        )
        
        pivot_pval = df_summary.pivot_table(
            index='factor_suffix', 
            columns='config_label', 
            values='p_value',
            aggfunc='first'
        )
        
        # Create better factor name mapping
        factor_name_map = {config['suffix']: config['display_name'] for config in factor_configs}
        pivot_coef.index = [factor_name_map.get(idx, idx) for idx in pivot_coef.index]
        pivot_pval.index = [factor_name_map.get(idx, idx) for idx in pivot_pval.index]
        
        # Plot coefficient heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_coef, annot=True, cmap='RdBu_r', center=0, fmt='.4f')
        plt.title('Factor Coefficient Heatmap (All Configurations)')
        plt.tight_layout()
        
        if output_dir:
            coef_heatmap_file = os.path.join(output_dir, 'pooled_regression_coefficient_heatmap.png')
            plt.savefig(coef_heatmap_file)
            plt.close()
            print(f"Coefficient heatmap saved to: {coef_heatmap_file}")
        
        # Plot p-value heatmap
        plt.figure(figsize=(14, 8))
        # Use custom color mapping - green for significant (p<0.05), red for non-significant
        p_values_array = pivot_pval.values
        colors = np.zeros(p_values_array.shape + (3,))
        
        # Set colors: green for significant (p<0.05), red for non-significant
        for i in range(p_values_array.shape[0]):
            for j in range(p_values_array.shape[1]):
                if pd.notna(p_values_array[i, j]):
                    if p_values_array[i, j] < 0.05:
                        colors[i, j] = [0.2, 0.8, 0.2]  # green
                    else:
                        colors[i, j] = [0.8, 0.2, 0.2]  # red
        
        # Plot heatmap
        ax = plt.gca()
        ax.imshow(colors)
        
        # Add text annotations
        for i in range(len(pivot_pval.index)):
            for j in range(len(pivot_pval.columns)):
                value = pivot_pval.iloc[i, j]
                if pd.notna(value):
                    text_color = 'white'
                    ax.text(j, i, f'{value:.4f}', ha='center', va='center', color=text_color)
        
        # Set axis labels
        ax.set_xticks(np.arange(len(pivot_pval.columns)))
        ax.set_yticks(np.arange(len(pivot_pval.index)))
        ax.set_xticklabels(pivot_pval.columns)
        ax.set_yticklabels(pivot_pval.index)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        plt.title('Factor P-value Heatmap (Green = Significant p<0.05)')
        plt.tight_layout()
        
        if output_dir:
            pval_heatmap_file = os.path.join(output_dir, 'pooled_regression_pvalue_heatmap.png')
            plt.savefig(pval_heatmap_file)
            plt.close()
            print(f"P-value heatmap saved to: {pval_heatmap_file}")