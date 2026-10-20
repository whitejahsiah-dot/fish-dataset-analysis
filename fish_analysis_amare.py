import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load the CSV file into a Pandas DataFrame
df_amare = pd.read_csv('Fish.csv')

print("="*80)
print("1. EXPLORE THE DATA")
print("="*80)

# 1.1 Print the names of columns
print("\n1.1 Column Names:")
print(df_amare.columns.tolist())

# 1.2 Print the types of columns
print("\n1.2 Column Types:")
print(df_amare.dtypes)

# 1.3 Print the count of each unique values in the 'Species' column
print("\n1.3 Count of Unique Values in 'Species' Column:")
print(df_amare['Species'].value_counts())

# 1.4 Print statistics (count, min, mean, std, 25%, 50%, 75%, max) of all numeric columns
print("\n1.4 Statistics of All Numeric Columns:")
print(df_amare.describe())

# 1.5 Print the first three records
print("\n1.5 First Three Records:")
print(df_amare.head(3))

# 1.6 Print a summary of all missing values in all columns
print("\n1.6 Missing Values Summary:")
print(df_amare.isnull().sum())

print("\n" + "="*80)
print("2. VISUALIZE THE DATA")
print("="*80)

# 2.1 Plot a histogram for the Weight column with 10 bins
plt.figure(figsize=(10, 6))
plt.hist(df_amare['Weight'], bins=10, color='skyblue', edgecolor='black')
plt.xlabel('Weight (grams)')
plt.ylabel('Frequency')
plt.title('amare_Weight')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('amare_Weight_histogram.png', dpi=100)
plt.show()
print("\n2.1 Histogram for Weight column created and saved as 'amare_Weight_histogram.png'")

# 2.2 Create a scatterplot showing "Length1" versus "Weight"
plt.figure(figsize=(10, 6))
plt.scatter(df_amare['Length1'], df_amare['Weight'], color='coral', alpha=0.6, edgecolors='black')
plt.xlabel('Length1 (cm)')
plt.ylabel('Weight (grams)')
plt.title('amare_WL_scatter')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('amare_WL_scatter.png', dpi=100)
plt.show()
print("2.2 Scatterplot for Length1 vs Weight created and saved as 'amare_WL_scatter.png'")

# 2.3 Create a boxplot for the Weight column
plt.figure(figsize=(8, 6))
plt.boxplot(df_amare['Weight'])
plt.ylabel('Weight (grams)')
plt.title('amare_boxplot')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('amare_boxplot.png', dpi=100)
plt.show()
print("2.3 Boxplot for Weight column created and saved as 'amare_boxplot.png'")

print("\n" + "="*80)
print("3. PRE-PROCESS THE DATA")
print("="*80)

# Create a copy of the dataframe for preprocessing
df_amare_processed = df_amare.copy()

# 3.1 Add a new column 'Avg_len_amare' with mean of Length1, Length2, Length3
df_amare_processed['Avg_len_amare'] = df_amare_processed[['Length1', 'Length2', 'Length3']].mean(axis=1)
print("\n3.1 Added column 'Avg_len_amare' with average of Length columns")

# 3.2 Drop the three Length columns
df_amare_processed = df_amare_processed.drop(['Length1', 'Length2', 'Length3'], axis=1)
print("3.2 Dropped columns: Length1, Length2, Length3")

# 3.3 Drop the Height column
df_amare_processed = df_amare_processed.drop('Height', axis=1)
print("3.3 Dropped column: Height")

# 3.4 Convert categorical columns to numeric using get_dummies
df_amare_processed = pd.get_dummies(df_amare_processed, columns=['Species'], drop_first=False)
print("3.4 Converted categorical column 'Species' to numeric using get_dummies")
print(f"   New columns after get_dummies: {df_amare_processed.columns.tolist()}")

# 3.5 Rename the processed dataframe to df_amare_numeric
df_amare_numeric = df_amare_processed.copy()

print("\n3.5 Final Preprocessed DataFrame (df_amare_numeric):")
print(f"   Shape: {df_amare_numeric.shape}")
print(f"   Data types:\n{df_amare_numeric.dtypes}")
print(f"\n   First few rows:")
print(df_amare_numeric.head())

print("\n" + "="*80)
print("4. BUILD A MODEL AND VALIDATE")
print("="*80)

# 4.1 Prepare data for modeling
X = df_amare_numeric.drop('Weight', axis=1)  # Features
y = df_amare_numeric['Weight']  # Target

# 4.2 Split data: 70% training, 30% testing with random_state = last two digits of student number
# Using random_state = 43 (CHANGE THIS TO YOUR STUDENT NUMBER'S LAST TWO DIGITS)
random_seed = 43
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=random_seed)

print(f"\n4.1-4.2 Data split (70% train, 30% test) with random_state={random_seed}")
print(f"   Training set size: {X_train.shape[0]}")
print(f"   Testing set size: {X_test.shape[0]}")

# 4.3 Build and fit the linear regression model
model_amare = LinearRegression()
model_amare.fit(X_train, y_train)

print("\n4.3 Linear Regression Model Created and Fitted")

# 4.4 Print the parameters of the model
print("\n4.4 Model Parameters:")
print(f"   Intercept: {model_amare.intercept_:.4f}")
print(f"   Coefficients:")
for feature, coef in zip(X.columns, model_amare.coef_):
    print(f"      {feature}: {coef:.6f}")

# 4.5 Make predictions on test data
y_pred = model_amare.predict(X_test)

# 4.6 Calculate and print scores
score = model_amare.score(X_test, y_test)
r2 = r2_score(y_test, y_pred)

print("\n4.6 Model Performance on Test Data:")
print(f"   R-squared Score: {score:.4f}")
print(f"   R-squared (from r2_score): {r2:.4f}")
print(f"   Mean Squared Error: {np.mean((y_test - y_pred)**2):.4f}")

print("\n" + "="*80)
print("ANALYSIS QUESTIONS")
print("="*80)

print("\n5.1 Key highlights of the original dataset:")
print(f"   - Total number of records: {len(df_amare)}")
print(f"   - Total number of columns: {len(df_amare.columns)}")
print(f"   - Species included: {df_amare['Species'].unique().tolist()}")
print(f"   - Species distribution: {dict(df_amare['Species'].value_counts())}")
print(f"   - Weight range: {df_amare['Weight'].min()} to {df_amare['Weight'].max()} grams")
print(f"   - Missing values: {df_amare.isnull().sum().sum()} total")

print(f"\n5.2 After pre-processing:")
print(f"   - Original number of rows: {len(df_amare)}")
print(f"   - New number of rows: {len(df_amare_numeric)}")
print(f"   - Original number of columns: {len(df_amare.columns)}")
print(f"   - New number of columns: {len(df_amare_numeric.columns)}")
print(f"   - Columns in new dataframe: {df_amare_numeric.columns.tolist()}")

print(f"\n5.3 Model Performance Analysis:")
print(f"   - R-squared Score: {score:.4f} or {score*100:.2f}%")
if score > 0.8:
    print(f"   - Assessment: This is an EXCELLENT model")
    print(f"   - Interpretation: The model explains {score*100:.2f}% of the variance in fish weight")
    print(f"   - The model is very reliable for predicting fish weight based on the features")
elif score > 0.6:
    print(f"   - Assessment: This is a GOOD model")
    print(f"   - Interpretation: The model explains {score*100:.2f}% of the variance in fish weight")
    print(f"   - The model is reasonably reliable for predictions")
elif score > 0.4:
    print(f"   - Assessment: This is an ACCEPTABLE model")
    print(f"   - Interpretation: The model explains {score*100:.2f}% of the variance")
    print(f"   - The model shows moderate predictive power")
else:
    print(f"   - Assessment: This model needs improvement")
    print(f"   - Interpretation: The model only explains {score*100:.2f}% of the variance")
    print(f"   - Consider feature engineering or different algorithms")

print("\n" + "="*80)
