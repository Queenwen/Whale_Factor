# CoinGecko API base URL
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"

# API request interval in seconds
API_DELAY = 1.5

# Number of tokens per page
PER_PAGE = 250

# Output filename
OUTPUT_FILE = "active_defi_tokens.csv"

# Santiment API key
SANTIMENT_API_KEY = "22kc57wsi2xiofmb_hbgpmfzd65pfy72t"

# GraphQL API endpoint
SANTIMENT_GRAPHQL_URL = "https://api.santiment.net/graphql"

# Token list
TOKENS = [
    "uniswap", "aave", "curve-dao-token", "maker", "lido-dao", "compound",
    "yearn-finance", "synthetix-network-token", "frax-share", "dydx", "balancer",
    "bancor", "sushiswap", "rocketpool", "pancakeswap"
]

# Sentiment metrics
SENTIMENT_METRICS = [
    "weighted_sentiment",
    "positive_sentiment",
    "negative_sentiment"
]

# Other metrics
OTHER_METRICS = [
    "social_volume_total",
    "dev_activity",
    "top_holders_percent_of_total_supply"
]

# All metrics
ALL_METRICS = SENTIMENT_METRICS + OTHER_METRICS

# Data date range
START_DATE = "2020-01-01T00:00:00Z"
END_DATE = "2024-12-31T23:59:59Z"

# Data storage paths
SENTIMENT_DATA_DIR = "santiment_data"
OTHER_DATA_DIR = "other_data"

# Output base directory
OUTPUT_BASE_DIR = '/Users/queenwen/Desktop/UIUC/580/DeFi/whale_effect/multifactor_strategy_outputs'