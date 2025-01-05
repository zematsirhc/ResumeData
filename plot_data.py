import matplotlib.pyplot as plt
import polars as pl
from sklearn.preprocessing import (
    MinMaxScaler,
)  # Transform the value column into a normalized 0-1

train_set = pl.read_parquet(r"border_crossings\data\train_set.parquet")

# Group the data
train_set = train_set.group_by(["Latitude", "Longitude"]).agg(pl.col("Value").sum())

# Instantiate a scaler so that we can utilize the value as the alpha
scaler = MinMaxScaler()

# Scale the value column
scaled_values = scaler.fit_transform(train_set["Value"].to_numpy().reshape(-1, 1))

# Add scaled values back to the DataFrame, adjusting slightly for alpha
adjusted_alpha = scaled_values.flatten()
adjustment = 1e-6
adjusted_alpha = adjusted_alpha * (1 - 2 * adjustment) + adjustment

train_set = train_set.with_columns(pl.Series("Value", adjusted_alpha))
print(train_set)

# Examine the data by latitute and longitude
fig, ax = plt.subplots()
ax.scatter(x=train_set["Longitude"], y=train_set["Latitude"], alpha=train_set["Value"])


ax.set_title("Border Crossing Locations by Latitude and Longitude")
fig.show()
fig.savefig(r"border_crossings\data\plot.png")
