from dagster import Definitions

from stock_predictor.assets.prices import raw_prices
from stock_predictor.assets.features import price_features

defs = Definitions(
    assets=[raw_prices, price_features],
)
