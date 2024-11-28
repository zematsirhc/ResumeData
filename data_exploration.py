import pandas as pd

people = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\01_people.csv"
)
abilities = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\02_abilities.csv"
)

education = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\03_education.csv"
)

experience = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\04_experience.csv"
)

person_skills = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\05_person_skills.csv"
)

skills = pd.read_csv(
    r"C:\Users\Chris\.cache\kagglehub\datasets\suriyaganesh\resume-dataset-structured\versions\2\06_skills.csv"
)
print(people.head())
