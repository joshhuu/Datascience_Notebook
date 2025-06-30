# 📘 Python for Data Science Workshop Notebook
# Author: Josh S
# Dataset used: Titanic Dataset (uploaded)

# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For ML Demo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Plot styles
sns.set(style="whitegrid")

# ✅ Step 2: Load Dataset
df = pd.read_csv("7e88493f-1ae4-4559-b969-d2dc4c3e6820.csv")
df.head()

# ✅ Step 3: Explore Dataset
print("Dataset Info:\n")
df.info()

print("\nStatistical Summary:\n")
df.describe()

# ✅ Step 4: Data Cleaning
# Checking nulls
df.isnull().sum()

# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin for simplicity
df.drop(columns=['Cabin'], inplace=True)

# ✅ Step 5: Pandas Operations
# Count of survivors
df['Survived'].value_counts()

# Group by class and survival
df.groupby(['Pclass', 'Survived'])['PassengerId'].count()

# ✅ Step 6: Visualization
# Survival count
sns.countplot(x='Survived', data=df)
plt.title('Survival Count')
plt.show()

# Age distribution
sns.histplot(df['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot by Class
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.show()

# Heatmap for correlation
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# ✅ Step 7: NumPy Quick Math
print("Mean Age:", np.mean(df['Age']))
print("Standard Deviation:", np.std(df['Age']))
print("Correlation between Age and Fare:", np.corrcoef(df['Age'], df['Fare'])[0,1])

# ✅ Step 8: Mini ML - Logistic Regression
# Convert Sex to binary
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Feature selection
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ✅ Step 9: Wrap Up
# Try modifying the model or adding more features like SibSp, Parch, etc.
# Feel free to fork this notebook and play with it!
