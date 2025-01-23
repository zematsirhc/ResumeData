import pandas as pd

pd.set_option("display.max_columns", None)

path = r"border_crossings\data\president_party_preprocessed.pkl"

president_party = pd.read_pickle(path)

president_party = president_party.drop(
    columns=[
        "No.[a]",
        "Party[b][17]",
        "Election",
        "Vice President[18]",
        "Portrait",
        "Term[16]",
    ]
)

# Normalize the numeric data notations:
president_party = president_party.replace("/", "-", regex=True)
# Remove all of the wiki footnote hyperlinks enclosed in brackets
president_party = president_party.replace("\[\s*.*?\s*\]", "", regex=True)
# Remove the dates to get just the presidents' names
president_party = president_party.replace("\ (\(.*\))", "", regex=True)
# Ensure no white space
president_party["Name (birth–death)"] = president_party[
    "Name (birth–death)"
].str.strip()

# Output data to a pickle file
president_party.to_pickle(r"border_crossings\data\president_party_processed.pkl")
print(president_party)
