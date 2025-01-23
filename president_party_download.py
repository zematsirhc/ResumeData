import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States"

webpage_tables = pd.read_html(url)

# Check which table we want
for index, df in enumerate(webpage_tables):
    print(f"Head {index}: \n {df.head(5)} Tail {index}: \n {df.tail(5)}")

# Get the first table
president_party = webpage_tables[0]

# Load to pickle file for faster processing while we work
president_party.to_pickle(r"border_crossings\data\president_party_preprocessed.pkl")
