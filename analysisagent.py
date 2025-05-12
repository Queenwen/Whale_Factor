import pandas as pd
import numpy as np
from scipy.stats import zscore
import os
import argparse
import sys
from pathlib import Path

class AnalysisAgent:
    """Whale Effect Analysis Agent Class"""
    
    def __init__(self, base_dir=None, output_dir=None):
        """Initialize the analysis agent"""
        # Set base directory
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(self.base_dir, "intermediate_steps")
        else:
            self.output_dir = output_dir
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
    
    @staticmethod
    def load_common_tokens(file_path):
        """Load common tokens from a CSV file"""
        try:
            tokens_df = pd.read_csv(file_path)
            if 'token' not in tokens_df.columns:
                print(f"Error: 'token' column not found in file {file_path}.")
                return []
            return tokens_df['token'].unique().tolist()
        except FileNotFoundError:
            print(f"Error: Common tokens file not found {file_path}")
            return []

    def run_analysis(self, price_file=None, holders_file=None, count_file=None, volume_file=None, final_output=None):
        """
        Execute whale effect analysis
        
        Parameters:
        price_file: Path to the price data file, use default if None
        holders_file: Path to the whale holdings data file, use default if None
        count_file: Path to the whale transaction count data file, use default if None
        volume_file: Path to the whale transaction volume data file, use default if None
        final_output: Path to the final output file, use default if None
        """
        # Set default file paths
        if price_file is None:
            price_file = os.path.join(self.base_dir, "token_prices", "merged_token_prices.csv")
        if holders_file is None:
            holders_file = os.path.join(self.base_dir, "processed_no_high_missing", "amount_in_top_holders.csv")
        if count_file is None:
            count_file = os.path.join(self.base_dir, "processed_no_high_missing", "whale_transaction_count_100k_usd_to_inf.csv")
        if volume_file is None:
            volume_file = os.path.join(self.base_dir, "processed_no_high_missing", "whale_transaction_volume_100k_usd_to_inf.csv")
        if final_output is None:
            final_output = os.path.join(self.base_dir, "analysis_results.csv")
        
        # Load price data
        print(f"Loading price data: {price_file}")
        price_df = pd.read_csv(price_file)

        # Load whale holdings data
        print(f"Loading whale holdings data: {holders_file}")
        whale_holders_df = pd.read_csv(holders_file)

        # Load whale transaction count data
        try:
            print(f"Attempting to load whale transaction count data: {count_file}")
            whale_count_df = pd.read_csv(count_file)
            print(f"Loaded whale transaction count data from: {count_file}")
        except FileNotFoundError:
            print(f"Warning: Whale transaction count data file not found: {count_file}")
            whale_count_df = None

        # Load whale transaction volume data
        try:
            print(f"Attempting to load whale transaction volume data: {volume_file}")
            whale_volume_df = pd.read_csv(volume_file)
            print(f"Loaded whale transaction volume data from: {volume_file}")
        except FileNotFoundError:
            print(f"Warning: Whale transaction volume data file not found: {volume_file}")
            whale_volume_df = None

        # Convert data to wide format, each token as a column
        price_wide = price_df.copy()
        price_wide.to_csv(os.path.join(self.output_dir, '01_price_wide_initial.csv'), index=False)
        print("Saved: 01_price_wide_initial.csv")

        # Step 1: Factor preprocessing (lag + normalization)
        # Lag processing
        lag_columns = {}
        for token in price_wide.columns[1:]:  # Skip 'datetime' column
            lag_columns[f'{token}_lag1'] = price_wide[token].shift(1)

        # Add all lag columns to DataFrame at once
        lag_df = pd.DataFrame(lag_columns)
        price_wide = pd.concat([price_wide, lag_df], axis=1)
        price_wide.to_csv(os.path.join(self.output_dir, '02_price_wide_with_lag.csv'), index=False)
        print("Saved: 02_price_wide_with_lag.csv")

        # Normalization (cross-sectional z-score)
        for date in price_wide['datetime'].unique():
            mask = price_wide['datetime'] == date
            for token in price_wide.columns[1:]:
                if not token.endswith('_lag1') and token != 'datetime':  # Only normalize original price columns, exclude datetime
                    try:
                        if pd.api.types.is_numeric_dtype(price_wide.loc[mask, token]):
                            price_wide.loc[mask, token] = zscore(price_wide.loc[mask, token])
                    except Exception as e:
                        pass
        price_wide.to_csv(os.path.join(self.output_dir, '03_price_wide_normalized.csv'), index=False)
        print("Saved: 03_price_wide_normalized.csv")

        # Step 2: Construct target variables (future returns)
        return_columns = {}
        for token in price_df.columns[1:]:  # Skip 'datetime' column
            return_columns[f'{token}_return_1d'] = price_df[token].pct_change(fill_method=None).shift(-1)
            return_columns[f'{token}_return_7d'] = price_df[token].pct_change(periods=7, fill_method=None).shift(-7)
            return_columns[f'{token}_return_14d'] = price_df[token].pct_change(periods=14, fill_method=None).shift(-14)
            return_columns[f'{token}_return_21d'] = price_df[token].pct_change(periods=21, fill_method=None).shift(-21)

        # Add all return columns to DataFrame at once
        return_df = pd.DataFrame(return_columns)
        price_wide = pd.concat([price_wide, return_df.set_index(price_wide.index[:len(return_df)])], axis=1)
        price_wide.to_csv(os.path.join(self.output_dir, '04_price_wide_with_returns.csv'), index=False)
        print("Saved: 04_price_wide_with_returns.csv")

        # Step 3: Validity check
        # Construct a whale factor scoring system

        # Process whale holdings data
        whale_holders_wide = None
        try:
            whale_holders_df_loaded = pd.read_csv(holders_file)
            print(f"Loaded whale holdings data from: {holders_file}")
            whale_holders_df_loaded['datetime'] = pd.to_datetime(whale_holders_df_loaded['datetime'])
            
            # Rename columns
            renamed_columns_holders = {}
            for col in whale_holders_df_loaded.columns:
                if col == 'datetime':
                    renamed_columns_holders[col] = col
                else:
                    renamed_columns_holders[col] = col.replace('_amount_in_top_holders', '_amount_top_holders')
            whale_holders_df_loaded.columns = [renamed_columns_holders.get(col, col) for col in whale_holders_df_loaded.columns]
            whale_holders_wide = whale_holders_df_loaded
            print("Whale holdings data columns renamed.")
        except FileNotFoundError:
            print(f"Warning: Whale holdings data file not found: {holders_file}")
        except Exception as e:
            print(f"Error processing whale holdings data: {e}")

        # Process whale transaction count data
        whale_count_wide = None
        try:
            if whale_count_df is not None:
                print(f"Processing loaded whale transaction count data.")
                whale_count_df['datetime'] = pd.to_datetime(whale_count_df['datetime'])
                
                # Rename columns
                renamed_columns_count = {}
                for col in whale_count_df.columns:
                    if col == 'datetime':
                        renamed_columns_count[col] = col
                    else:
                        renamed_columns_count[col] = col.replace('_whale_transaction_count_100k_usd_to_inf', '_txn_count_100k')
                whale_count_df.columns = [renamed_columns_count.get(col, col) for col in whale_count_df.columns]
                whale_count_wide = whale_count_df
                print("Whale transaction count data columns renamed.")
            elif not os.path.exists(count_file):
                print(f"Warning: Whale transaction count data file not found at: {count_file}")
        except Exception as e:
            print(f"Error processing whale transaction count data: {e}")

        # Process whale transaction volume data
        whale_volume_wide = None
        try:
            if whale_volume_df is not None:
                print(f"Processing loaded whale transaction volume data.")
                whale_volume_df['datetime'] = pd.to_datetime(whale_volume_df['datetime'])

                # Rename columns
                renamed_columns_volume = {}
                for col in whale_volume_df.columns:
                    if col == 'datetime':
                        renamed_columns_volume[col] = col
                    else:
                        renamed_columns_volume[col] = col.replace('_whale_transaction_volume_100k_usd_to_inf', '_txn_vol_100k')
                whale_volume_df.columns = [renamed_columns_volume.get(col, col) for col in whale_volume_df.columns]
                whale_volume_wide = whale_volume_df
                print("Whale transaction volume data columns renamed.")
            elif not os.path.exists(volume_file):
                print(f"Warning: Whale transaction volume data file not found at: {volume_file}")
        except Exception as e:
            print(f"Error processing whale transaction volume data: {e}")

        price_wide['datetime'] = pd.to_datetime(price_wide['datetime'])
        # Merge price data and all whale holdings data
        merged_df = price_wide.copy()

        if whale_holders_wide is not None:
            merged_df = pd.merge(merged_df, whale_holders_wide, on='datetime', how='left')
        if whale_count_wide is not None:
            merged_df = pd.merge(merged_df, whale_count_wide, on='datetime', how='left')
        if whale_volume_wide is not None:
            merged_df = pd.merge(merged_df, whale_volume_wide, on='datetime', how='left')

        merged_df.to_csv(os.path.join(self.output_dir, '05_merged_df_before_whale_factors.csv'), index=False)
        print("Saved: 05_merged_df_before_whale_factors.csv")

        # Step 4: Construct whale factors (lag + Z-score)
        whale_metric_suffixes = []
        if whale_holders_wide is not None:
            whale_metric_suffixes.append('amount_top_holders')
        if whale_count_wide is not None:
            whale_metric_suffixes.append('txn_count_100k')
        if whale_volume_wide is not None:
            whale_metric_suffixes.append('txn_vol_100k')

        whale_lag_cols_for_zscore = []
        if not merged_df.empty and whale_metric_suffixes:
            lag_whale_dict = {}
            for token_col_base in price_df.columns[1:]:
                for suffix in whale_metric_suffixes:
                    original_whale_col = f"{token_col_base}_{suffix}"
                    if original_whale_col in merged_df.columns:
                        lagged_col_name = f"{original_whale_col}_lag1"
                        lag_whale_dict[lagged_col_name] = merged_df[original_whale_col].shift(1)
                        whale_lag_cols_for_zscore.append(lagged_col_name)
            
            if lag_whale_dict:
                lag_whale_df = pd.DataFrame(lag_whale_dict)
                merged_df = pd.concat([merged_df, lag_whale_df.set_index(merged_df.index[:len(lag_whale_df)])], axis=1)

            merged_df.to_csv(os.path.join(self.output_dir, '06_merged_df_with_whale_lag.csv'), index=False)
            print("Saved: 06_merged_df_with_whale_lag.csv")
        else:
            print("merged_df is empty or no whale metric suffixes, skipping whale factor lag processing.")

        # Calculate Z-score
        if whale_lag_cols_for_zscore:
            zscore_dict = {}
            for col in whale_lag_cols_for_zscore:
                try:
                    if pd.api.types.is_numeric_dtype(merged_df[col]) and merged_df[col].notna().sum() > 1:
                        col_filled = merged_df[col].fillna(merged_df[col].median())
                        zscore_dict[f'{col}_zscore'] = zscore(col_filled, nan_policy='omit')
                    else:
                        zscore_dict[f'{col}_zscore'] = np.nan
                except Exception as e:
                    zscore_dict[f'{col}_zscore'] = np.nan
            
            if zscore_dict:
                zscore_df = pd.DataFrame(zscore_dict, index=merged_df.index)
                merged_df = pd.concat([merged_df, zscore_df], axis=1)
            merged_df.to_csv(os.path.join(self.output_dir, '07_merged_df_with_whale_zscore.csv'), index=False)
            print("Saved: 07_merged_df_with_whale_zscore.csv")
        else:
            print("No available whale metric lag columns, unable to calculate z-score factors for tokens.")

        # Print processed DataFrame
        print(merged_df.head())

        # Save final result
        merged_df.to_csv(final_output, index=False)
        print(f"Analysis complete, final result saved to {final_output}")
        
        return merged_df

def main():
    """Command line entry function"""
    parser = argparse.ArgumentParser(description='Whale Effect Analysis Tool')
    parser.add_argument('--base-dir', type=str, help='Base directory path')
    parser.add_argument('--output-dir', type=str, help='Intermediate output directory')
    parser.add_argument('--price-file', type=str, help='Price data file path')
    parser.add_argument('--holders-file', type=str, help='Whale holdings data file path')
    parser.add_argument('--count-file', type=str, help='Whale transaction count data file path')
    parser.add_argument('--volume-file', type=str, help='Whale transaction volume data file path')
    parser.add_argument('--final-output', type=str, help='Final output file path')
    
    args = parser.parse_args()
    
    # Create an instance of the analysis agent
    agent = AnalysisAgent(base_dir=args.base_dir, output_dir=args.output_dir)
    
    # Run analysis
    agent.run_analysis(
        price_file=args.price_file,
        holders_file=args.holders_file,
        count_file=args.count_file,
        volume_file=args.volume_file,
        final_output=args.final_output
    )

# if __name__ == "__main__":
#     main()