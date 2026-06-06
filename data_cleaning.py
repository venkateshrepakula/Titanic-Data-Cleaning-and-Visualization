import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("data/titanic.csv")

print("Dataset Loaded Successfully!")

# Dataset Shape
print("\nDataset Shape:")
print(df.shape)

# Missing Values Before Cleaning
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Handle Missing Values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Remove Duplicates
df = df.drop_duplicates()

# Missing Values After Cleaning
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# Save Cleaned Dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Graph 1 - Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# Graph 2 - Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"], bins=20)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# Graph 3 - Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("class_survival.png")
plt.show()

print("\nAll graphs saved successfully!")
print("Project Completed Successfully!")