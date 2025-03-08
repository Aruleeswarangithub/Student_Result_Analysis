import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/student_scores.csv")

# Data Preprocessing
df = df.drop("Unnamed: 0", axis=1, errors="ignore")

# Gender Distribution
plt.figure(figsize=(5, 5))
ax = sns.countplot(data=df, x="Gender")
plt.title("Gender Distribution")
ax.bar_label(ax.containers[0])
plt.savefig("images/gender_distribution.png")
plt.show()

# Relationship between Parent's Education and Student's Score
gb = df.groupby("ParentEduc")[["MathScore", "ReadingScore", "WritingScore"]].mean()
plt.figure(figsize=(8, 5))
sns.heatmap(gb, annot=True, cmap="coolwarm")
plt.title("Impact of Parent's Education on Student's Score")
plt.savefig("images/parent_education_vs_scores.png")
plt.show()

# Relationship between Parent's Marital Status and Student's Score
gb1 = df.groupby("ParentMaritalStatus")[["MathScore", "ReadingScore", "WritingScore"]].mean()
plt.figure(figsize=(8, 5))
sns.heatmap(gb1, annot=True, cmap="coolwarm")
plt.title("Impact of Parent's Marital Status on Student's Score")
plt.savefig("images/parent_marital_status_vs_scores.png")
plt.show()

# Boxplots for Score Distributions
for col in ["MathScore", "ReadingScore", "WritingScore"]:
    sns.boxplot(data=df, x=col)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"images/{col}_boxplot.png")
    plt.show()

# Ethnic Group Distribution
ethnic_counts = df["EthnicGroup"].value_counts()
plt.pie(ethnic_counts, labels=ethnic_counts.index, autopct="%1.2f%%")
plt.title("Distribution of Ethnic Groups")
plt.savefig("images/ethnic_distribution.png")
plt.show()

# Ethnic Group Count Plot
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="EthnicGroup")
ax.bar_label(ax.containers[0])
plt.title("Ethnic Group Count")
plt.savefig("images/ethnic_group_count.png")
plt.show()
