import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr
import os
import argparse


class BacktestAgent:
    """Backtest agent class for executing multifactor strategy backtesting"""
    
    def __init__(self, output_dir=None):
        """Initialize backtest agent"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "multifactor_strategy_outputs")
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def calculate_performance_metrics(self, daily_returns, days_per_year=252):
        """Calculate strategy performance metrics"""
        if daily_returns.empty or daily_returns.std() == 0:
            return {
                'Cumulative Return': 0,
                'Annualized Return': 0,
                'Annualized Volatility': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0
            }

        cumulative_return = (1 + daily_returns).prod() - 1
        annualized_return = daily_returns.mean() * days_per_year
        annualized_volatility = daily_returns.std() * np.sqrt(days_per_year)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

        cumulative_returns_series = (1 + daily_returns).cumprod()
        peak = cumulative_returns_series.expanding(min_periods=1).max()
        drawdown = (cumulative_returns_series - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'Cumulative Return': cumulative_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }
        
    def plot_net_value_curve(self, net_value_df, title, filename):
        """Plot net value curve"""
        plt.figure(figsize=(14, 7))
        for col in net_value_df.columns:
            plt.plot(net_value_df.index, net_value_df[col], label=col)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (Normalized to 1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Net value curve saved to: {filename}")
        
    def run_multifactor_backtest(self, data_df, common_tokens, N, holding_period, factor_suffix_template):
        """Execute multifactor strategy backtesting"""
        print(f"\nStarting multifactor backtest: N={N}, Holding Period={holding_period} days, Factor Suffix Template: {factor_suffix_template}")

        if not isinstance(data_df.index, pd.DatetimeIndex):
            if 'datetime' in data_df.columns:
                data_df['datetime'] = pd.to_datetime(data_df['datetime'])
                data_df = data_df.set_index('datetime')
            else:
                print("Error: data_df must have 'datetime' as index or contain 'datetime' column.")
                return None, None
        
        data_df = data_df.sort_index()

        token_specific_factor_cols = [
            f"{token}{factor_suffix_template}"
            for token in common_tokens
            if f"{token}{factor_suffix_template}" in data_df.columns
        ]
        
        valid_tokens_for_this_factor = [
            token for token in common_tokens
            if f"{token}{factor_suffix_template}" in data_df.columns and f"{token}_return_1d" in data_df.columns
        ]

        if not valid_tokens_for_this_factor:
            print(f"Error: For factor suffix {factor_suffix_template}, no valid tokens found with both factor and return data.")
            return None, None

        if len(valid_tokens_for_this_factor) < 2 * N:
            print(f"Error: For factor suffix {factor_suffix_template}, insufficient valid tokens ({len(valid_tokens_for_this_factor)}) for long-short strategy (minimum {2*N} required).")
            return None, None

        print(f"Using {len(valid_tokens_for_this_factor)} valid tokens for backtesting based on factor {factor_suffix_template}.")

        daily_portfolio_returns = pd.DataFrame(index=data_df.index)
        daily_portfolio_returns['long_return'] = 0.0
        daily_portfolio_returns['short_return'] = 0.0
        daily_portfolio_returns['long_short_return'] = 0.0
        
        unique_dates = data_df.index.unique().sort_values()

        for i in range(0, len(unique_dates), holding_period):
            rebalance_date = unique_dates[i]
            
            token_factor_values = {}
            for token in valid_tokens_for_this_factor:
                token_factor_col = f"{token}{factor_suffix_template}" 
                if rebalance_date in data_df.index and pd.notna(data_df.loc[rebalance_date, token_factor_col]):
                    token_factor_values[token] = data_df.loc[rebalance_date, token_factor_col]
            
            if not token_factor_values or len(token_factor_values) < 2 * N:
                continue

            sorted_tokens = sorted(token_factor_values.items(), key=lambda item: item[1], reverse=True)
                
            long_portfolio_tokens = [token for token, factor_val in sorted_tokens[:N]]
            short_portfolio_tokens = [token for token, factor_val in sorted_tokens[-N:]]

            actual_holding_end_date_idx = min(i + holding_period, len(unique_dates))
            
            for date_idx_in_holding_period in range(i, actual_holding_end_date_idx):
                current_holding_date = unique_dates[date_idx_in_holding_period]
                
                daily_long_returns_for_date = []
                for token in long_portfolio_tokens:
                    ret_col = f"{token}_return_1d"
                    if current_holding_date in data_df.index and ret_col in data_df.columns and pd.notna(data_df.loc[current_holding_date, ret_col]):
                        daily_long_returns_for_date.append(data_df.loc[current_holding_date, ret_col])
                
                daily_short_returns_for_date = []
                for token in short_portfolio_tokens:
                    ret_col = f"{token}_return_1d"
                    if current_holding_date in data_df.index and ret_col in data_df.columns and pd.notna(data_df.loc[current_holding_date, ret_col]):
                        daily_short_returns_for_date.append(data_df.loc[current_holding_date, ret_col])

                avg_long_return = np.mean(daily_long_returns_for_date) if daily_long_returns_for_date else 0
                avg_short_return = np.mean(daily_short_returns_for_date) if daily_short_returns_for_date else 0
                
                daily_portfolio_returns.loc[current_holding_date, 'long_return'] = avg_long_return
                daily_portfolio_returns.loc[current_holding_date, 'short_return'] = avg_short_return
                daily_portfolio_returns.loc[current_holding_date, 'long_short_return'] = avg_long_return - avg_short_return
                
        returns_df = daily_portfolio_returns.loc[(daily_portfolio_returns!=0).any(axis=1)]
                
        if returns_df.empty:
            print("No portfolio returns generated during backtest period.")
            return None, None

        metrics_to_show = {}

        print("\n--- Long Portfolio Performance ---")
        long_metrics = self.calculate_performance_metrics(returns_df['long_return'])
        for k, v in long_metrics.items(): print(f"{k}: {v:.4f}")
        metrics_to_show['Long_Portfolio'] = long_metrics

        print("\n--- Short Portfolio Performance ---")
        short_metrics = self.calculate_performance_metrics(returns_df['short_return'])
        for k, v in short_metrics.items(): print(f"{k}: {v:.4f}")
        metrics_to_show['Short_Portfolio'] = short_metrics
        
        print("\n--- Long-Short Portfolio Performance ---")
        long_short_metrics = self.calculate_performance_metrics(returns_df['long_short_return'])
        for k, v in long_short_metrics.items(): print(f"{k}: {v:.4f}")
        metrics_to_show['Long_Short_Portfolio'] = long_short_metrics

        performance_summary_df = pd.DataFrame(metrics_to_show)
        factor_name_for_file = factor_suffix_template.replace("_lag1_zscore", "").replace("_","")
        perf_filename = os.path.join(self.output_dir, f'performance_{factor_name_for_file}_N{N}_H{holding_period}.csv')
        performance_summary_df.to_csv(perf_filename)
        print(f"Performance summary saved to: {perf_filename}")

        net_value_df = pd.DataFrame(index=returns_df.index)
        net_value_df['Long Portfolio'] = (1 + returns_df['long_return']).cumprod()
        net_value_df['Short Portfolio'] = (1 + returns_df['short_return']).cumprod() 
        net_value_df['Long-Short Portfolio'] = (1 + returns_df['long_short_return']).cumprod()
        
        plot_title = f'Strategy Performance (Factor: {factor_name_for_file}, N={N}, Hold={holding_period}d)'
        plot_filename = os.path.join(self.output_dir, f'net_value_{factor_name_for_file}_N{N}_H{holding_period}.png')
        self.plot_net_value_curve(net_value_df, plot_title, plot_filename)

        return returns_df, net_value_df
    
    def run_backtest_analysis(self, data_df, common_tokens, factor_configs, n_value=5, holding_period=3):
        """Execute backtest analysis"""
        print("\n=== Starting Backtest Analysis ===")
        
        backtest_results = {}
        performance_metrics = {}
        
        for factor_config in factor_configs:
            print(f"\nRunning backtest for factor '{factor_config['display_name']}'...")
            
            daily_returns, net_value = self.run_multifactor_backtest(
                data_df=data_df,
                common_tokens=common_tokens,
                N=n_value,
                holding_period=holding_period,
                factor_suffix_template=factor_config['suffix']
            )
            
            if daily_returns is not None:
                backtest_results[factor_config['name']] = {
                    'daily_returns': daily_returns,
                    'net_value': net_value
                }
                
                long_short_metrics = self.calculate_performance_metrics(daily_returns['long_short_return'])
                performance_metrics[factor_config['name']] = long_short_metrics
                
                plot_filename = os.path.join(self.output_dir, f"net_value_{factor_config['name']}.png")
                self.plot_net_value_curve(
                    net_value_df=net_value,
                    title=f"Net Value Curve: {factor_config['display_name']} (N={n_value}, Holding Period={holding_period} days)",
                    filename=plot_filename
                )
        
        if len(backtest_results) > 1:
            combined_net_value = pd.DataFrame()
            for factor_name, result in backtest_results.items():
                combined_net_value[factor_name] = result['net_value']['Long-Short Portfolio']
            
            plot_filename = os.path.join(self.output_dir, "all_factors_comparison.png")
            self.plot_net_value_curve(
                net_value_df=combined_net_value,
                title=f"Multifactor Strategy Comparison (N={n_value}, Holding Period={holding_period} days)",
                filename=plot_filename
            )
        
        if performance_metrics:
            metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
            metrics_file = os.path.join(self.output_dir, "performance_metrics_summary.csv")
            metrics_df.to_csv(metrics_file)
            print(f"Performance metrics summary saved to: {metrics_file}")
            print(f"Performance metrics summary saved to: {metrics_file}")
            print("\nPerformance Metrics Summary:")
            print(metrics_df)
        
        print("Backtest analysis completed")
        return backtest_results, performance_metrics


def load_common_tokens(file_path):
    """Load common tokens from CSV file"""
    try:
        tokens_df = pd.read_csv(file_path)
        if 'token' not in tokens_df.columns:
            print(f"Error: 'token' column not found in file {file_path}.")
            return []
        return tokens_df['token'].unique().tolist()
    except FileNotFoundError:
        print(f"Error: Common tokens file not found {file_path}")
        return []


def main():
    """Command line entry function"""
    parser = argparse.ArgumentParser(description='Multifactor Backtest Tool')
    parser.add_argument('--data-file', type=str, required=True, help='Analysis results data file path')
    parser.add_argument('--tokens-file', type=str, required=True, help='Common tokens list file path')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--n-value', type=int, default=5, help='Select top N and bottom N tokens')
    parser.add_argument('--holding-period', type=int, default=3, help='Holding period (days)')
    
    args = parser.parse_args()
    
    backtest_agent = BacktestAgent(output_dir=args.output_dir)
    
    try:
        common_tokens = load_common_tokens(args.tokens_file)
        print(f"Loaded {len(common_tokens)} common tokens")
    except Exception as e:
        print(f"Error: Failed to load common tokens list: {e}")
        return
    
    try:
        data_df = pd.read_csv(args.data_file)
        if 'datetime' in data_df.columns:
            data_df['datetime'] = pd.to_datetime(data_df['datetime'])
            data_df = data_df.set_index('datetime')
        print(f"Loaded analysis results data, total {len(data_df)} records")
    except Exception as e:
        print(f"Error: Failed to load analysis results data: {e}")
        return
    
    factor_configs = [
        {"suffix": "_amount_top_holders_lag1_zscore", "name": "AmountTopHolders", "display_name": "Amount in Top Holders (Lag1 Z-Score)"},
        {"suffix": "_txn_count_100k_lag1_zscore", "name": "TxnCount100k", "display_name": "Transaction Count >100k (Lag1 Z-Score)"},
        {"suffix": "_txn_vol_100k_lag1_zscore", "name": "TxnVol100k", "display_name": "Transaction Volume >100k (Lag1 Z-Score)"},
    ]
    
    backtest_agent.run_backtest_analysis(
        data_df=data_df,
        common_tokens=common_tokens,
        factor_configs=factor_configs,
        n_value=args.n_value,
        holding_period=args.holding_period
    )


# if __name__ == "__main__":
#     main()