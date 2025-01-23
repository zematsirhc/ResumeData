import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

path = r"border_crossings\data\government_data_preprocessed.pkl"

government_control = pd.read_pickle(path)

# The data is in multi-index format
multi_index = government_control.columns

# Get a list of the tuples
multi_index = multi_index.to_list()

# Convert the list to only the first tuple instance
new_index = pd.Index([e[0] for e in multi_index])

# Set the header to that
government_control.columns = new_index

# Drop NaN rows
government_control = government_control.dropna(subset=["Congress"])

# Reset the index and drop the old index column
government_control = government_control.reset_index().drop("index", axis=1)

# The true headers are on row 124 (except a few on the header row)
actual_header = government_control.columns
second_header = government_control.iloc[124]

# Zip the two rows together
header_tuples = zip(actual_header, second_header)

# Create a list for the final headers.
final_header = [
    h[0] if type(h[1]) != str else h[0] + " - " + h[1] if h[0] != h[1] else h[1]
    for h in header_tuples
]

# Override the header values
government_control.columns = final_header

# Drop the "Total" rows
government_control = government_control[government_control["Congress"] != "Congress"]

# Create a start year column
government_control["Start Year"] = government_control["Years"].str[0:4]
government_control["End Year"] = government_control["Years"].str[5:10]

# Normalize the numeric data notations:
government_control = government_control.replace("/", "-", regex=True)
# Remove all of the wiki footnote hyperlinks enclosed in brackets
government_control = government_control.replace("\[\s*.*?\s*\]", "", regex=True)

president_party = pd.read_pickle(r"border_crossings\data\president_party_processed.pkl")

# Merge with the president_party data to get the presidents' party affiliation
government_control = government_control.merge(
    president_party,
    left_on="House of Representatives - President",
    right_on="Name (birth–death)",
    how="inner",
)

# Rename the affiliation column
government_control = government_control.rename(
    columns={
        "Party[b][17].1": "President Control",
        "Name (birth–death)": "President",
    }
)

# Drop excess columns
government_control = government_control.drop(
    columns=[
        "Congress",
        "Years",
        "President - Trifecta",
        "Trifecta",
        "House of Representatives - President",
        "Unnamed: 7_level_0 - Total",
    ]
)


def normalize_years(value):
    """
    This function is to handle the cases where there are e.g., 52-53 republicans (or dems) in a given timeframe
    """
    try:
        # Check if value is not empty or None
        if pd.isna(value) or value == "":
            return float(0)
        # Replace various dash-like characters with a standard dash
        value = str(value).replace("–", "-").replace("—", "-")
        # Split the string into parts and convert to float
        parts = [float(n) for n in value.split("-")]
        # Calculate the average if multiple parts exist, otherwise return the single number
        return sum(parts) / len(parts) if len(parts) > 1 else parts[0]
    except ValueError:
        return float(0)


columns_to_normalize = [
    "Senate - Democrats",
    "Senate - Republicans",
    "Senate - Others",
    "Senate - Vacancies",
    "House of Representatives - Democrats",
    "House of Representatives - Republicans",
    "House of Representatives - Others",
    "House of Representatives - Vacancies",
]

# Normalize colummns
government_control[columns_to_normalize] = government_control[
    columns_to_normalize
].applymap(normalize_years)


# Calculate who controls the senate
government_control["Senate Control"] = np.where(
    government_control["Senate - Republicans"]
    > government_control["Senate - Democrats"],
    "Republican",
    np.where(
        government_control["Senate - Democrats"]
        > government_control["Senate - Republicans"],
        "Democrats",
        "Tie",
    ),
)

# Calculate who controlls the house
government_control["House Control"] = np.where(
    government_control["House of Representatives - Republicans"]
    > government_control["House of Representatives - Democrats"],
    "Republican",
    np.where(
        government_control["House of Representatives - Democrats"]
        > government_control["House of Representatives - Republicans"],
        "Democrats",
        "Tie",
    ),
)

# Keep only these columns
government_control = government_control.loc[
    :,
    ["Start Year", "End Year", "President Control", "Senate Control", "House Control"],
]

# Export final processed governmental control data
government_control.to_pickle(r"border_crossings\data\government_control_processed.pkl")
print(government_control)
