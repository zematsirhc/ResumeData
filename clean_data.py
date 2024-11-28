import polars as pl
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfTransformer


# First time you run it, uncomment these downloads
# nltk.download("stopwords")
# nltk.download("wordnet")
people = pl.read_csv(r"data\01_people.csv")
abilities = pl.read_csv(r"data\02_abilities.csv")
education = pl.read_csv(r"data\03_education.csv")
experience = pl.read_csv(r"data\04_experience.csv")
person_skills = pl.read_csv(r"data\05_person_skills.csv")
skills = pl.read_csv(r"data\06_skills.csv")

# Make the job titles all lower case
experience = experience.with_columns(pl.col("title").str.to_lowercase().alias("title"))

# Convert the job column to 1 hot encoding
experience = experience.with_columns(
    pl.when(pl.col("title").str.contains_any(["data scien"]))
    .then(1)
    .otherwise(0)
    .alias("title")
)
experience_grouped = experience.group_by(pl.col("person_id")).agg(pl.col("title").max())

# This portion will standardize the skills data
lemitizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def prepocessed_skills(text: str) -> str:
    """
    Remove unnessary words and make the skills barebones
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
    tokens = word_tokenize(text)
    return " ".join(
        [lemitizer.lemmatize(word) for word in tokens if word not in stop_words]
    )


person_skills = person_skills.with_columns(
    cleaned_skills=pl.col("skill").map_elements(
        prepocessed_skills, return_dtype=pl.String
    )
)

skills_corpus = person_skills.get_column(pl.col("skill")).to_list()

vectorizer = TfidfTransformer(max_features=1000)
tfidf_matrix = vectorizer.fit_transform(skills_corpus)

print(person_skills)


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
