import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import spearmanr

from analysisagent import AnalysisAgent
from backtest_agent import BacktestAgent
from factor_validation_agent import FactorValidationAgent
from trading_strategy_agent import TradingStrategyAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_TOKENS_FILE = os.path.join(BASE_DIR, 'whale_effect_common_tokens.csv')
OUTPUT_BASE_DIR = os.path.join(BASE_DIR, 'factor_validation_outputs')
BACKTEST_OUTPUT_DIR = os.path.join(BASE_DIR, 'multifactor_strategy_outputs')

os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
os.makedirs(BACKTEST_OUTPUT_DIR, exist_ok=True)

FACTOR_CONFIGS = [
    {"suffix": "_amount_top_holders_lag1_zscore", "name": "AmountTopHolders", "display_name": "Amount in Top Holders (Lag1 Z-Score)"},
    {"suffix": "_txn_count_100k_lag1_zscore", "name": "TxnCount100k", "display_name": "Transaction Count >100k (Lag1 Z-Score)"},
    {"suffix": "_txn_vol_100k_lag1_zscore", "name": "TxnVol100k", "display_name": "Transaction Volume >100k (Lag1 Z-Score)"},
]

RETURN_CONFIGS = [
    {'return_col_suffix': '_return_1d', 'name': '1D Forward Return', 'lag': 1},
    {'return_col_suffix': '_return_7d', 'name': '7D Forward Acc. Return', 'lag': 7},
    {'return_col_suffix': '_return_14d', 'name': '14D Forward Acc. Return', 'lag': 14},
    {'return_col_suffix': '_return_21d', 'name': '21D Forward Acc. Return', 'lag': 21},
]

N_VALUE = 5
HOLDING_PERIOD = 3

def run_factor_validation(data_df, common_tokens):
    print("\n=== Starting factor validation analysis ===")
    
    factor_validation_agent = FactorValidationAgent(output_dir=OUTPUT_BASE_DIR)
    
    validation_results = factor_validation_agent.run_factor_validation(
        data_df=data_df,
        common_tokens=common_tokens,
        factor_configs=FACTOR_CONFIGS,
        return_configs=RETURN_CONFIGS
    )
    
    print("Factor validation analysis completed")
    return validation_results

def run_backtest_analysis(data_df, common_tokens):
    print("\n=== Starting backtest analysis ===")
    
    backtest_agent = BacktestAgent(output_dir=BACKTEST_OUTPUT_DIR)
    
    backtest_results = backtest_agent.run_backtest_analysis(
        data_df=data_df,
        common_tokens=common_tokens,
        factor_configs=FACTOR_CONFIGS,
        n_value=N_VALUE,
        holding_period=HOLDING_PERIOD
    )
    
    print("Backtest analysis completed")
    return backtest_results

def run_trading_strategy_analysis(data_df, common_tokens):
    print("\n=== Starting multifactor strategy analysis ===")
    
    trading_strategy_agent = TradingStrategyAgent(output_dir=OUTPUT_BASE_DIR)
    
    factor_configs_to_test = FACTOR_CONFIGS
    
    regression_configs_to_test = [
        {'return_suffix': '_return_1d', 'y_shift_lag': 1, 'name': '1D_Return_Lag1'},
        {'return_suffix': '_return_7d', 'y_shift_lag': 0, 'name': '7D_Acc_Return_Lag0'},
        {'return_suffix': '_return_14d', 'y_shift_lag': 0, 'name': '14D_Acc_Return_Lag0'},
        {'return_suffix': '_return_21d', 'y_shift_lag': 0, 'name': '21D_Acc_Return_Lag0'},
        {'return_suffix': '_return_1d', 'y_shift_lag': 7, 'name': '1D_Return_Lag7'},
    ]

    all_factors_pooled_regression_summary = []

    for factor_config in factor_configs_to_test:
        current_factor_suffix = factor_config["suffix"]
        current_factor_name = factor_config["name"]
        current_factor_display_name = factor_config["display_name"]

        print(f"\n\n{'='*60}")
        print(f"===== Processing factor: {current_factor_display_name} ({current_factor_suffix}) =====")
        print(f"{'='*60}")

        print(f"\n--- 1.1 Running backtest for factor '{current_factor_display_name}' ---")
        
        backtest_agent = BacktestAgent(output_dir=BACKTEST_OUTPUT_DIR)
        returns_df, net_value_df = backtest_agent.run_multifactor_backtest(
            data_df=data_df.copy(),
            common_tokens=common_tokens,
            N=N_VALUE,
            holding_period=HOLDING_PERIOD,
            factor_suffix_template=current_factor_suffix
        )
        
        if returns_df is not None and not returns_df.empty:
            print(f"Backtest completed for factor {current_factor_display_name}.")
            
            metrics = backtest_agent.calculate_performance_metrics(returns_df['long_short_return'])
            print("\n--- Long-short portfolio performance ---")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
                
            output_file = os.path.join(BACKTEST_OUTPUT_DIR, f"net_value_curve_{current_factor_name}.png")
            backtest_agent.plot_net_value_curve(net_value_df, f"Net Value Curve - {current_factor_display_name}", output_file)
        else:
            print(f"Backtest failed to generate results for factor {current_factor_display_name} or insufficient valid tokens.")

        print(f"\n--- 1.2 Running lagged regression analysis for factor '{current_factor_display_name}' (token-specific and pooled) ---")
        all_regression_results_for_current_factor = []
        current_factor_pooled_results = []
        
        regression_plot_output_dir = os.path.join(OUTPUT_BASE_DIR, f"regression_plots_{current_factor_name}")
        os.makedirs(regression_plot_output_dir, exist_ok=True)

        for reg_config in regression_configs_to_test:
            config_label_individual = f"{reg_config['name']}_token_specific"
            config_label_pooled = f"{reg_config['name']}_pooled"

            print(f"\n--- Running Token-Specific Regressions for: Factor={current_factor_name}, Config={reg_config['name']} ---")
            results_df_single_config = trading_strategy_agent.perform_lagged_regression(
                data_df=data_df.copy(),
                common_tokens=common_tokens,
                factor_suffix=current_factor_suffix,
                return_suffix=reg_config['return_suffix'],
                lag=reg_config['y_shift_lag'],
                plot_results=True, 
                plot_top_n=15,
                output_dir=regression_plot_output_dir, 
                config_label=config_label_individual
            )
            if results_df_single_config is not None and not results_df_single_config.empty:
                all_regression_results_for_current_factor.append(results_df_single_config)

            print(f"\n--- Running Pooled Regression for: Factor={current_factor_name}, Config={reg_config['name']} ---")
            pooled_reg_result = trading_strategy_agent.perform_pooled_lagged_regression(
                data_df=data_df.copy(),
                common_tokens=common_tokens,
                factor_suffix=current_factor_suffix,
                return_suffix=reg_config['return_suffix'],
                lag=reg_config['y_shift_lag'],
                config_label=config_label_pooled
            )
            if pooled_reg_result:
                current_factor_pooled_results.append(pooled_reg_result)
                all_factors_pooled_regression_summary.append(pooled_reg_result) 
        
        if all_factors_pooled_regression_summary:
            df_all_pooled_summary = pd.DataFrame(all_factors_pooled_regression_summary)
            summary_pooled_csv_filename = os.path.join(OUTPUT_BASE_DIR, "summary_all_factors_pooled_regressions.csv")
            df_all_pooled_summary.to_csv(summary_pooled_csv_filename, index=False)
            print(f"\n\nAll factors pooled regression results summary saved to: {summary_pooled_csv_filename}")
            print(df_all_pooled_summary)
    
            trading_strategy_agent.plot_pooled_regression_summary_table(df_all_pooled_summary, factor_configs_to_test, OUTPUT_BASE_DIR)
    
        print("\n\nAll factors processed.")
        print("Multifactor strategy analysis completed")
        return df_all_pooled_summary if all_factors_pooled_regression_summary else None

def main():
    print("=== Whale Effect Analysis Main Program Started ===")
    
    parser = argparse.ArgumentParser(description='Whale Effect Analysis Tool')
    parser.add_argument('--data-file', type=str, help='Analysis results data file path')
    parser.add_argument('--tokens-file', type=str, help='Common tokens list file path')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--stage', type=str, choices=['data_prep', 'analysis', 'all'], 
                        default='all', help='Execution stage: data_prep, analysis, all')
    args = parser.parse_args()
    
    data_file = args.data_file if args.data_file else ANALYSIS_RESULTS_FILE
    tokens_file = args.tokens_file if args.tokens_file else COMMON_TOKENS_FILE
    
    common_tokens = AnalysisAgent.load_common_tokens(tokens_file)
    if not common_tokens:
        print("Error: Failed to load common tokens list, program terminated.")
        return
    print(f"Loaded {len(common_tokens)} common tokens")
    
    if args.stage in ['data_prep', 'all']:
        print("\n=== Stage 1: Data Preparation and Factor Construction ===")
        analysis_agent = AnalysisAgent()
        data_df = analysis_agent.run_analysis()
        if data_df is None:
            print("Error: Data processing and factor construction failed, program terminated.")
            return
        print(f"Data processing and factor construction completed, results saved to {ANALYSIS_RESULTS_FILE}")
    
    if args.stage in ['analysis', 'all']:
        print("\n=== Stage 2: Factor Validation and Backtest Analysis ===")
        data_df = FactorValidationAgent.load_analysis_results(data_file)
        if data_df is None:
            print("Error: Failed to load analysis results data, program terminated.")
            return
        print(f"Loaded analysis results data, total {len(data_df)} records")
        
        run_factor_validation(data_df, common_tokens)
        run_backtest_analysis(data_df, common_tokens)
        run_trading_strategy_analysis(data_df, common_tokens)
    
    print("\n=== Whale Effect Analysis Main Program Completed ===")

if __name__ == "__main__":
    main()