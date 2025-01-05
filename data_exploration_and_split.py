import polars as pl
from sklearn.model_selection import ShuffleSplit

border_crossings = pl.read_csv(
    r"C:\Users\Chris\Downloads\Border_Crossing_Entry_Data.csv"
)
border_crossings = border_crossings.with_row_index()

# Allows us to see all the columns
with pl.Config(tbl_cols=-1):
    print(border_crossings)

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

""" train_set.write_parquet(
    r"border_crossings\data\train_set.parquet"
)
test_set.write_parquet(
    r"border_crossings\data\test_set.parquet"
) """
