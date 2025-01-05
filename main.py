import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

training = pl.read_parquet(r"border_crossings\data\train_set.parquet")


class DateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column: str, date_format="%b %Y"):
        self.date_format = date_format
        self.column = column

    def fit(self, X: pl.DataFrame, y=None) -> pl.DataFrame:
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:

        if isinstance(X, pd.DataFrame):
            # Convert pandas to polars
            X = pl.from_pandas(X)

        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in the DataFrame.")
        X = X.with_columns(pl.col(self.column).str.to_datetime("%b %Y"))
        epoch = pl.lit(datetime(1950, 1, 1))
        X = X.with_columns(
            (pl.col(self.column) - epoch).cast(pl.Int64).alias("Days Since Epoch")
        )
        return X.to_pandas()


transformer = ColumnTransformer(
    transformers=[
        ("date_transformation", DateTransformer(column="Date"), ["Date"]),
        ("border_encoder", OneHotEncoder(), ["Border"]),
        ("state_encoding", OneHotEncoder(), ["State"]),
    ]
)

pandas_training = training.to_pandas()
transformed_data = transformer.fit_transform(pandas_training)

# Allows us to see all the columns
with pl.Config(tbl_cols=-1):
    print(transformed_data)

print(transformed_data)
