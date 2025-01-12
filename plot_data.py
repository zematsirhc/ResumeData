import matplotlib.pyplot as plt
import polars as pl

train_set = pl.read_parquet(r"border_crossings\data\train_set.parquet")

# Group the data
train_set = train_set.group_by(["Latitude", "Longitude"]).agg(pl.col("Value").sum())

# Examine the data by latitute and longitude
fig, ax = plt.subplots()
ax.scatter(x=train_set["Longitude"], y=train_set["Latitude"], alpha=0.1)


ax.set_title("Border Crossing Locations by Latitude and Longitude")
fig.show()
fig.savefig(r"border_crossings\data\plot.png")
