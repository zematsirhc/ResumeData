import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np

training = pd.read_parquet(r"border_crossings\data\train_set.parquet")

# Split the training data from the labels
y = ["Value"]
X = training.drop(y, axis=1)
y = training.copy(y)


class DateTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms a string date column into K-Hot encoding
    """

    def __init__(self, column: str, date_format="%b %Y"):
        self.date_format = date_format
        self.column = column

    def fit(self, X, y=None):
        """
        Fit only needs to return self
        """
        return self

    def transform(self, X):
        X["Date"] = pd.to_datetime(X["Date"], format=self.date_format)
        X["Year"] = X[self.column].dt.year
        X["Month"] = X[self.column].dt.month
        print(X)


datetransformer = DateTransformer("Date")
datetransformer.transform(X)


transformer = ColumnTransformer(
    transformers=[
        ("date_transformation", DateTransformer(column="Date"), ["Date"]),
        # ("border_encoder", OneHotEncoder(), ["Border"]),
        # ("state_encoding", OneHotEncoder(), ["State"]),
    ]
)

pandas_training = training.to_pandas()
transformed_data = transformer.fit_transform(pandas_training)

# Allows us to see all the columns
with pl.Config(tbl_cols=-1):
    print(transformed_data)

print(transformed_data)

# To do:
# drop the values column and make a separate dataframe that has the values
# We can split the numeric columns and the categorical columns into separate
# pipelines, and then transform them both in a ColumnTransformer or using from sklearn.pipeline.FeatureUnion
# e.g., from sklear.compose import ColumnTransformer
# full_pipeline = ColumnTransformer([
#        ("num", num_pipeline, num_attribs), # Where num_pipeline is its own pipeline
#        ("cat", OneHotEncoder(), cat_attribs),
#    ])
