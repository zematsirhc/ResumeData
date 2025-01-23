import pandas as pd

url = "https://en.wikipedia.org/wiki/Party_divisions_of_United_States_Congresses"

webpage_tables = pd.read_html(url)

# Check which table we want
for index, df in enumerate(webpage_tables):
    print(f"Head {index}: \n {df.head(5)} Tail {index}: \n {df.tail(5)}")

# Get the first table
governmental_control = webpage_tables[0]

# Load to pickle file for faster processing while we work
governmental_control.to_pickle(
    r"border_crossings\data\government_data_preprocessed.pkl"
)
