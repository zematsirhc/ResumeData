import polars as pl
from sklearn.model_selection import ShuffleSplit
import pandas as pd

border_crossings = pl.read_csv(
    r"C:\Users\Chris\Downloads\Border_Crossing_Entry_Data.csv"
)
print(border_crossings)
# Read pickle file with pandas
government_control = pd.read_pickle(
    r"border_crossings\data\government_control_processed.pkl"
)

# Convert to Polars DF
government_control = pl.from_pandas(government_control)

# Change years to ints
government_control = government_control.with_columns(
    government_control["Start Year"].cast(pl.Int64).alias("Start Year"),
    government_control["End Year"].cast(pl.Int64).alias("End Year"),
)

# Add the middle year
government_control = government_control.with_columns(
    (pl.col("Start Year") + 1).alias("Middle Year")
)

# Make sure our assumptions about the year columns are accurate
assert ((government_control["End Year"] - government_control["Middle Year"]) == 1).all()
assert (
    (government_control["Middle Year"] - government_control["Start Year"]) == 1
).all()

# Get a year column which matches (int64) in the border crossing data set
border_crossings = border_crossings.with_columns(
    (pl.col("Date").str.tail(4)).cast(pl.Int64).alias("Year")
)

# Unpivot the dates so that they're in a single column
government_control = government_control.unpivot(
    on=["Start Year", "Middle Year", "End Year"],
    index=["President Control", "House Control", "Senate Control"],
    variable_name="Year Type",
    value_name="Year",
)

# Ensure there are no duplicate years
government_control = government_control.unique("Year")

government_control = government_control.sort("Year")

# Join on the year column
border_crossings = border_crossings.join(
    government_control,
    left_on="Year",
    right_on="Year",
    how="left",
)

# Drop extra columns
border_crossings = border_crossings.drop(["Year Type", "Year"])


border_crossings = border_crossings.with_row_index()

# Reserve 20% of the data for testing purposes
split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Establish the training and testing sets
for test_index, train_index in split.split(border_crossings):
    test_set = border_crossings.filter(pl.col("index").is_in(test_index))
    train_set = border_crossings.filter(pl.col("index").is_in(train_index))

uniques = {k: v[0] for k, v in train_set.select(pl.all().n_unique()).to_dict().items()}

print(uniques)

print(f"Test set: {len(test_set)}  Train set: {len(train_set)}")

print(train_set.describe())

# train_set.write_parquet(r"border_crossings\data\train_set.parquet")
# test_set.write_parquet(r"border_crossings\data\test_set.parquet")
