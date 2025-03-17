# Diabetes Prediction Model

This project aims to predict diabetes using machine learning techniques based on a dataset containing patient health metrics.

## Project Structure
- `data/`: Contains the dataset (`diabetes_prediction_dataset.csv`).
- `src/`: Contains the main script (`diabetes_prediction.py`).
- `classifier1.pkl`: Trained model file (optional).
- `requirements.txt`: List of dependencies.
- `README.md`: Project documentation.

## Dataset
The dataset (`diabetes_prediction_dataset.csv`) includes the following features:
- `Pregnancies`: Number of pregnancies.
- `Glucose`: Plasma glucose concentration.
- `BloodPressure`: Diastolic blood pressure (mm Hg).
- `SkinThickness`: Triceps skin fold thickness (mm).
- `Insulin`: 2-hour serum insulin (mu U/ml).
- `BMI`: Body mass index (weight in kg/(height in m)^2).
- `DiabetesPedigreeFunction`: Diabetes pedigree function (a measure of genetic influence).
- `Age`: Age in years.
- `Outcome`: Target variable (0 = non-diabetic, 1 = diabetic).

The dataset contains 768 samples with no missing values.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diabetes_prediction.git
   cd diabetes_prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the `data/` folder:
   Ensure `diabetes_prediction_dataset.csv` is located at `data/diabetes_prediction_dataset.csv`.

## Usage
Run the script to perform exploratory data analysis (EDA), preprocess the data, train models, and evaluate performance:

```bash
python src/diabetes_prediction.py
```

The script will:
- Generate visualizations saved as PNG files (e.g., `age_distribution.png`, `confusion_matrix.png`).
- Output model accuracies and a detailed classification report to the console.
- Save the trained model as `classifier1.pkl`.

## Methodology
### Exploratory Data Analysis (EDA):
- Visualizations of feature distributions (e.g., histograms, boxplots) and their relationships with Outcome.
- Correlation matrix to identify feature relationships.

### Data Preprocessing:
- Remove duplicate rows (if any).
- Cap BMI outliers using the interquartile range (IQR) method.
- Split data into training (78%) and testing (22%) sets.

### Modeling:
- Train basic models: Logistic Regression, Naive Bayes, Decision Tree, and Random Forest.
- Train an advanced Random Forest model using a pipeline with:
  - SMOTE (Synthetic Minority Oversampling Technique) to oversample the minority class.
  - RandomUnderSampler to balance the classes.
  - GridSearchCV to optimize hyperparameters.

### Evaluation:
- Assess models using accuracy, precision, recall, F1-score, and a confusion matrix.

## Results
### Basic Models:
- Logistic Regression: ~75.74% accuracy
- Naive Bayes: ~76.92% accuracy
- Decision Tree: ~73.37% accuracy
- Random Forest: ~75.15% accuracy

### Optimized Random Forest:
- Uses SMOTE (sampling_strategy=0.7) and RandomUnderSampler (sampling_strategy=1.0) for class balancing.
- Hyperparameters tuned via GridSearchCV (e.g., n_estimators, max_depth).
- Detailed performance metrics (precision, recall, F1-score) are printed in the console output.
- Visualizations are saved as PNG files for further analysis.

## Visualizations
The following plots are generated and saved in the project directory:
- `age_distribution.png`: Histogram of age distribution.
- `bmi_distribution.png`: Histogram of BMI distribution.
- `outcome_distribution.png`: Count plot of diabetes outcomes.
- `bmi_vs_outcome.png`: Boxplot of BMI vs. Outcome.
- `age_vs_outcome.png`: Boxplot of Age vs. Outcome.
- `glucose_vs_outcome.png`: Boxplot of Glucose vs. Outcome.
- `correlation_matrix.png`: Heatmap of feature correlations.
- `confusion_matrix.png`: Confusion matrix for the optimized Random Forest model.

## License
This project is licensed under the MIT License.

## Acknowledgments
- **Dataset Source**: The dataset is sourced from Kaggle (Pima Indians Diabetes Database).
- **Tools**: Built with Python, scikit-learn, pandas, matplotlib, seaborn, and imbalanced-learn.

