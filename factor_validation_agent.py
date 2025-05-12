import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates

class FactorValidationAgent:
    """Factor validation agent class for performing factor effectiveness analysis"""
    
    def __init__(self, output_dir=None):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "factor_validation_outputs")
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.n_quantiles = 5
        
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        
    def load_common_tokens(self, filepath):
        try:
            with open(filepath, 'r') as f:
                tokens = [line.strip() for line in f if line.strip()]
            return tokens
        except FileNotFoundError:
            print(f"Error: Common tokens file not found {filepath}")
            return []

    def load_data(self, filepath):
        try:
            df = pd.read_csv(filepath)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
            return df
        except FileNotFoundError:
            print(f"Error: Analysis results file not found {filepath}")
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
            
    @staticmethod
    def load_analysis_results(file_path):
        try:
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()
            return df
        except FileNotFoundError:
            print(f"Error: Analysis results file not found {file_path}")
            return None

    def calculate_information_coefficient(self, df, common_tokens, factor_suffix, return_col_suffix, factor_display_name, return_display_name, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir
            
        print(f"\n--- IC Analysis: Factor='{factor_display_name}', Return='{return_display_name}' ---")
        daily_ics = []
        dates = []

        for date, daily_data in df.iterrows():
            factor_values = []
            return_values = []
            valid_tokens_for_day = 0
            for token in common_tokens:
                factor_col = f"{token}{factor_suffix}"
                return_col = f"{token}{return_col_suffix}"

                if factor_col in daily_data and return_col in daily_data and \
                pd.notna(daily_data[factor_col]) and pd.notna(daily_data[return_col]):
                    factor_values.append(daily_data[factor_col])
                    return_values.append(daily_data[return_col])
                    valid_tokens_for_day +=1
            
            if len(factor_values) >= 5:
                ic, p_value = spearmanr(factor_values, return_values)
                if pd.notna(ic):
                    daily_ics.append(ic)
                    dates.append(date)

        if not daily_ics:
            print("No daily IC values calculated.")
            return None

        ic_series = pd.Series(daily_ics, index=pd.to_datetime(dates))
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        icir = mean_ic / std_ic if std_ic != 0 else np.nan

        print(f"IC Mean: {mean_ic:.4f}")
        print(f"IC Std: {std_ic:.4f}")
        print(f"ICIR: {icir:.4f}")

        plt.figure(figsize=(14, 7))
        ic_series.plot(marker='o', linestyle='-', markersize=4)
        plt.axhline(mean_ic, color='r', linestyle='--', label=f'Mean IC: {mean_ic:.4f}')
        plt.axhline(0, color='k', linestyle='-', linewidth=0.8)
        plt.title(f"Daily Information Coefficient (IC): {factor_display_name} vs {return_display_name}")
        plt.xlabel("Date")
        plt.ylabel("Spearman IC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // (12*20))))

        factor_name_file = factor_suffix.replace("_lag1_zscore", "").replace("_","")
        return_name_file = return_col_suffix.replace("_return_","").replace("_","")
        plot_filename = os.path.join(output_dir, f"ic_series_{factor_name_file}_vs_{return_name_file}.png")
        plt.savefig(plot_filename)
        print(f"IC time series plot saved to: {plot_filename}")
        plt.close()

        return {"factor": factor_display_name, "return_period": return_display_name, 
                "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir, "num_observations": len(ic_series)}

    def perform_quantile_analysis(self, df, common_tokens, factor_suffix, return_col_suffix, n_quantiles=None, factor_display_name=None, return_display_name=None, output_dir=None):
        if n_quantiles is None:
            n_quantiles = self.n_quantiles
        if output_dir is None:
            output_dir = self.output_dir
            
        print(f"\n--- Quantile Analysis (N={n_quantiles}): Factor='{factor_display_name}', Return='{return_display_name}' ---")
        
        daily_quantile_returns = {f'Q{i+1}': [] for i in range(n_quantiles)}

        for date, daily_data in df.iterrows():
            token_factor_return_pairs = []
            for token in common_tokens:
                factor_col = f"{token}{factor_suffix}"
                return_col = f"{token}{return_col_suffix}"

                if factor_col in daily_data and return_col in daily_data and \
                pd.notna(daily_data[factor_col]) and pd.notna(daily_data[return_col]):
                    token_factor_return_pairs.append({
                        'token': token,
                        'factor_value': daily_data[factor_col],
                        'return_value': daily_data[return_col]
                    })
            
            if len(token_factor_return_pairs) < n_quantiles:
                continue

            temp_df = pd.DataFrame(token_factor_return_pairs)
            try:
                temp_df['quantile_group'] = pd.qcut(temp_df['factor_value'], n_quantiles, labels=False, duplicates='drop')
            except ValueError as e:
                continue
                
            for q_idx in range(n_quantiles):
                quantile_label = f'Q{q_idx+1}'
                tokens_in_quantile = temp_df[temp_df['quantile_group'] == q_idx]
                if not tokens_in_quantile.empty:
                    avg_return_for_quantile_today = tokens_in_quantile['return_value'].mean()
                    daily_quantile_returns[quantile_label].append(avg_return_for_quantile_today)

        avg_returns_per_quantile = {}
        for q_label, returns_list in daily_quantile_returns.items():
            if returns_list:
                avg_returns_per_quantile[q_label] = np.nanmean(returns_list)
            else:
                avg_returns_per_quantile[q_label] = np.nan 

        print("Average returns per quantile:")
        for q_label, avg_ret in avg_returns_per_quantile.items():
            print(f"{q_label}: {avg_ret:.6f}")

        quantile_labels_sorted = sorted(avg_returns_per_quantile.keys())
        avg_returns_values_sorted = [avg_returns_per_quantile[q] for q in quantile_labels_sorted]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(quantile_labels_sorted, avg_returns_values_sorted, color='skyblue')
        plt.title(f"Quantile Average Returns: {factor_display_name} vs {return_display_name}\n(Q1: Low factor value, Q{n_quantiles}: High factor value)")
        plt.xlabel("Quantile")
        plt.ylabel(f"Average {return_display_name}")
        plt.grid(axis='y', linestyle='--')
        
        for bar in bars:
            yval = bar.get_height()
            if pd.notna(yval):
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom' if yval >=0 else 'top', ha='center')

        plt.tight_layout()
        factor_name_file = factor_suffix.replace("_lag1_zscore", "").replace("_","")
        return_name_file = return_col_suffix.replace("_return_","").replace("_","")
        plot_filename = os.path.join(output_dir, f"quantile_returns_{factor_name_file}_vs_{return_name_file}_N{n_quantiles}.png")
        plt.savefig(plot_filename)
        print(f"Quantile returns plot saved to: {plot_filename}")
        plt.close()

        return {"factor": factor_display_name, "return_period": return_display_name, "n_quantiles": n_quantiles,
                "avg_returns_per_quantile": avg_returns_per_quantile}

    def plot_factor_validation_summary_table(self, summary_df, factor_configs, n_quantiles=None, output_dir=None):
        if n_quantiles is None:
            n_quantiles = self.n_quantiles
        if output_dir is None:
            output_dir = self.output_dir
            
        print("\n--- Generating factor validation summary table ---")
        
        unique_factors_in_df = summary_df['factor'].unique()

        for factor_display_name in unique_factors_in_df:
            factor_file_name_part = "UnknownFactor"
            for fc in factor_configs:
                if fc['display_name'] == factor_display_name:
                    factor_file_name_part = fc['name']
                    break
            
            factor_data = summary_df[summary_df['factor'] == factor_display_name].copy()
            
            if factor_data.empty:
                print(f"No validation data found for factor {factor_display_name}, skipping table generation.")
                continue

            table_rows = []
            processed_return_periods = []

            for ret_cfg in self.return_configs:
                return_period_name = ret_cfg['name']
                if return_period_name in processed_return_periods:
                    continue
                
                ic_data = factor_data[(factor_data['return_period'] == return_period_name) & (factor_data['analysis_type'] == 'IC')]
                quantile_data = factor_data[(factor_data['return_period'] == return_period_name) & (factor_data['analysis_type'] == 'QuantileSort')]

                row = {'Return Period': return_period_name}
                
                if not ic_data.empty:
                    row['IC Mean'] = f"{ic_data['mean_ic'].iloc[0]:.4f}" if pd.notna(ic_data['mean_ic'].iloc[0]) else "N/A"
                    row['ICIR'] = f"{ic_data['icir'].iloc[0]:.4f}" if pd.notna(ic_data['icir'].iloc[0]) else "N/A"
                else:
                    row['IC Mean'] = "N/A"
                    row['ICIR'] = "N/A"

                if not quantile_data.empty:
                    for i in range(1, n_quantiles + 1):
                        col_name = f'avg_return_Q{i}'
                        q_ret_val = quantile_data[col_name].iloc[0]
                        row[f'Q{i} Ret'] = f"{q_ret_val:.4f}" if pd.notna(q_ret_val) else "N/A"
                else:
                    for i in range(1, n_quantiles + 1):
                        row[f'Q{i} Ret'] = "N/A"
                
                table_rows.append(row)
                processed_return_periods.append(return_period_name)

            if not table_rows:
                print(f"No data available for factor {factor_display_name} to create table.")
                continue
                
            table_df_for_plot = pd.DataFrame(table_rows)
            
            column_order = ['Return Period', 'IC Mean', 'ICIR'] + [f'Q{i} Ret' for i in range(1, n_quantiles + 1)]
            table_df_for_plot = table_df_for_plot[column_order]

            plot_data = table_df_for_plot.values.tolist()
            col_labels = table_df_for_plot.columns.tolist()

            fig, ax = plt.subplots(figsize=(max(12, n_quantiles*1.5), max(3, len(plot_data) * 0.6)))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=plot_data, colLabels=col_labels, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.1, 1.1)

            for (i, j), cell in table.get_celld().items():
                if i == 0: 
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2c5985')
                cell.set_edgecolor('grey')

            plt.title(f"Factor Validation Summary: {factor_display_name}", fontsize=14, weight='bold', pad=20)
            plt.tight_layout(pad=1.5)
            
            filename = os.path.join(output_dir, f"summary_table_factor_validation_{factor_file_name_part}.png")
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Factor validation summary table saved to: {filename}")
            plt.close(fig)

    def run_factor_validation(self, data_df, common_tokens, factor_configs, return_configs):
        print("Starting factor validation analysis...")
        
        self.return_configs = return_configs
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        all_validation_results = []

        for factor_cfg in factor_configs:
            factor_suffix = factor_cfg["suffix"]
            factor_display_name = factor_cfg["display_name"]
            factor_file_name = factor_cfg["name"]

            factor_output_dir = os.path.join(self.output_dir, factor_file_name)
            os.makedirs(factor_output_dir, exist_ok=True)

            for ret_cfg in return_configs:
                return_col_suffix = ret_cfg["return_col_suffix"]
                return_display_name = ret_cfg["name"]

                ic_result = self.calculate_information_coefficient(
                    df=data_df, 
                    common_tokens=common_tokens, 
                    factor_suffix=factor_suffix, 
                    return_col_suffix=return_col_suffix,
                    factor_display_name=factor_display_name, 
                    return_display_name=return_display_name, 
                    output_dir=factor_output_dir
                )
                if ic_result:
                    all_validation_results.append({**ic_result, "analysis_type": "IC"})

                quantile_result = self.perform_quantile_analysis(
                    df=data_df, 
                    common_tokens=common_tokens, 
                    factor_suffix=factor_suffix, 
                    return_col_suffix=return_col_suffix, 
                    n_quantiles=self.n_quantiles,
                    factor_display_name=factor_display_name, 
                    return_display_name=return_display_name, 
                    output_dir=factor_output_dir
                )
                if quantile_result:
                    flat_quantile_result = {"analysis_type": "QuantileSort", 
                                            "factor": quantile_result["factor"],
                                            "return_period": quantile_result["return_period"],
                                            "n_quantiles": quantile_result["n_quantiles"]}
                    for q, r in quantile_result["avg_returns_per_quantile"].items():
                        flat_quantile_result[f"avg_return_{q}"] = r
                    all_validation_results.append(flat_quantile_result)
        
        if all_validation_results:
            results_df = pd.DataFrame(all_validation_results)
            summary_filename = os.path.join(self.output_dir, "factor_validation_summary.csv")
            results_df.to_csv(summary_filename, index=False)
            print(f"\nAll validation results saved to: {summary_filename}")
            print(results_df)

            self.plot_factor_validation_summary_table(results_df, factor_configs, self.n_quantiles, self.output_dir)

        print("\nFactor validation analysis completed.")
        return results_df if all_validation_results else None