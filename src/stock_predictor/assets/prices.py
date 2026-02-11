from dagster import asset
import pandas as pd


@asset
def raw_prices() -> pd.DataFrame:
    """
    A simple asset that returns toy stock price data.

    In a real system, this would ingest data from an external API
    or data warehouse.
    """
    data = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "close_price": [100.0, 101.5, 102.0],
    }

    return pd.DataFrame(data)
