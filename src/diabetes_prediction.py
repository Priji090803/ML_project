import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imbPipeline
import pickle

# Load dataset
df = pd.read_csv('../data/diabetes.csv')

# Exploratory Data Analysis
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print("Number of duplicate rows:", df[df.duplicated()].shape)
df = df.drop_duplicates()
print(df['Outcome'].value_counts())

# Visualizations
plt.hist(df['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.savefig('age_distribution.png')
plt.close()

sns.histplot(df['BMI'], bins=30)
plt.title('BMI Distribution')
plt.savefig('bmi_distribution.png')
plt.close()

sns.countplot(x='Outcome', data=df)
plt.title('Outcome Distribution')
plt.savefig('outcome_distribution.png')
plt.close()

sns.boxplot(x='Outcome', y='BMI', data=df)
plt.title('BMI vs Outcome')
plt.savefig('bmi_vs_outcome.png')
plt.close()

sns.boxplot(x='Outcome', y='Age', data=df)
plt.title('Age vs Outcome')
plt.savefig('age_vs_outcome.png')
plt.close()

sns.boxplot(x='Outcome', y='Glucose', data=df)
plt.title('Glucose vs Outcome')
plt.savefig('glucose_vs_outcome.png')
plt.close()

# Data Preprocessing
Q1 = df['BMI'].quantile(0.25)
Q3 = df['BMI'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
df.loc[df['BMI'] > upper, 'BMI'] = upper

sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.png')
plt.close()

# Split data
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.22, random_state=42)

# Model Training
logreg = LogisticRegression(random_state=42, max_iter=1000)
logreg.fit(X_train, Y_train)

nb = GaussianNB()
nb.fit(X_train, Y_train)

dectree = DecisionTreeClassifier(criterion='entropy', random_state=42)
dectree.fit(X_train, Y_train)

ranfor = RandomForestClassifier(random_state=42)
ranfor.fit(X_train, Y_train)

# Predictions
Y_pred_logreg = logreg.predict(X_test)
Y_pred_nb = nb.predict(X_test)
Y_pred_dectree = dectree.predict(X_test)
Y_pred_ranfor = ranfor.predict(X_test)

# Model Evaluation
print("Logistic Regression:", accuracy_score(Y_test, Y_pred_logreg) * 100)
print("Naive Bayes:", accuracy_score(Y_test, Y_pred_nb) * 100)
print("Decision Tree:", accuracy_score(Y_test, Y_pred_dectree) * 100)
print("Random Forest:", accuracy_score(Y_test, Y_pred_ranfor) * 100)

# Advanced Model with Pipeline and GridSearchCV
over = SMOTE(sampling_strategy=0.7)  # Oversample minority to 70% of majority (350 samples)
under = RandomUnderSampler(sampling_strategy=1.0)  # Balance classes equally (350 each)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
    ])

X1 = df.drop('Outcome', axis=1)
Y1 = df['Outcome']
clf = imbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('over', over),
    ('under', under),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(clf, param_grid, cv=5, error_score='raise')
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.2, random_state=42)
grid_search.fit(X1_train, Y1_train)

print("Best Parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X1_test)
print("Model Accuracy:", accuracy_score(Y1_test, y_pred))
print(classification_report(Y1_test, y_pred))

cm = confusion_matrix(Y1_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Save the model
with open('../classifier1.pkl', 'wb') as pickle_out:
    pickle.dump(grid_search, pickle_out)
