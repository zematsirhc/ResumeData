import polars as pl

people = pl.read_csv(r"data\01_people.csv")
abilities = pl.read_csv(r"data\02_abilities.csv")
education = pl.read_csv(r"data\03_education.csv")
experience = pl.read_csv(r"data\04_experience.csv")
person_skills = pl.read_csv(r"data\05_person_skills.csv")
skills = pl.read_csv(r"data\06_skills.csv")

experience = experience.filter(pl.col("title") == "Data Scientist")

print(f"Number of Data Scientists: {len(experience)}")

print(f"People \n {people.head(10)}")
print(f"Abilities \n {abilities.head(10)}")
print(f"Education \n {education.head(10)}")
print(f"Experience \n {experience.head(10)}")
print(f"Person Skills \n {person_skills.head(10)}")
print(f"Skills \n {skills.head(10)}")

print(people.describe())
