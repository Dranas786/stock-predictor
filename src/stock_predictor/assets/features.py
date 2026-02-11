from dagster import asset
import pandas as pd


@asset
def price_features(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple features from raw stock prices.

    This asset depends on `raw_prices` and will automatically
    run after it in Dagster.
    """
    df = raw_prices.copy()

    # Simple feature: daily price change
    df["daily_change"] = df["close_price"].diff()

    return df
