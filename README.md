### IMPORT ALL THE LIBRARIES
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.preprocessing import StandardScaler
###IMPORT THE DATASET
# Load the data
uploaded = files.upload()
filename = list(uploaded.keys())[0]

# Read the uploaded file into a DataFrame
data = pd.read_csv(filename)
#Convert into a dataframe
df = pd.DataFrame(data)
df
#Datatypes overview of attributes
data.info()
# Display missing values for each attribute
missing_values = data.isnull().sum()

missing_summary = pd.DataFrame({
    'Attribute': missing_values.index,
    'Missing Values': missing_values.values
})

# Display the result
print(missing_summary)
data.describe()
# Check skewness of the Age column
print(f"Skewness of Age: {data['Age'].skew()}")
# Display numerical attributes
numerical_attributes = data.select_dtypes(include=['int64', 'float64'])
print("Numerical Attributes:\n", numerical_attributes.columns.tolist())

# Display categorical attributes
categorical_attributes = data.select_dtypes(include=['object', 'category'])
print("Categorical Attributes:\n", categorical_attributes.columns.tolist())
##DATA PREPROCESSING
**1. DATA CLEANING**
####i) DROP COLUMNS  & HANDLING MISSING VALUES
# Load the dataset
df = pd.read_csv("Accidental_Drug_Related_Deaths.csv")

# Dropping redundant and futile columns
df.drop(
    labels=["Other Opioid","Ethnicity","Location if Other","Cause of Death", "Any Opioid", "Residence State", "Injury State", "Death State", "Manner of Death", "Injury Place"],
    axis=1,
    inplace=True
)

# Fill missing values for specific columns
missing_fill_values = {
    "Sex": "Unknown",
    "Race": "Unknown",
    "Location": "Unknown",
    "Other": "Not Specified",
    "ResidenceCityGeo": "Unknown",
    "InjuryCityGeo": "Unkown",
    "DeathCityGeo": "Unknown",
    "Other Significant Conditions ": "Not Specified",
    "Residence City": "Unknown",
    "Injury City": "Unknown",
    "Death City": "Unknown",
    "Residence County": "Unknown",
    "Injury County": "Unknown",
    "Death County": "Unknown"
}

# Apply the filling operation for the selected columns
for column, fill_value in missing_fill_values.items():
    if column in data.columns:
        df[column] = df[column].fillna(fill_value)

# Fill missing values for Age with median
if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Confirm missing values are handled
print("Missing Values after handling:\n", df.isnull().sum())

# Save the cleaned dataset
df.to_excel("Dropped_Handled_Drug_Overdose_Data.xlsx", index=False)
####ii) HANDLING DUPLICATES
# Identify Duplicated Rows
duplicates= data.duplicated()

# Return true for duplicated rows
num_duplicates = duplicates.sum()

# Print the number of duplicated rows
print(f"Number of Duplicated Rows {num_duplicates}")
####iii) REMOVING WHITESPACES
# Load the dataset
data_nowspace = pd.read_excel("Dropped_Handled_Drug_Overdose_Data.xlsx")

#Removing WhiteSpace
#Step 1: Trim Whitespace from col
data_nowspace =data.copy()

#Step 2: Trim Whitespace from Rows
data_nowspace.replace(to_replace={'Title':r'\s+'}, value='_', regex=True, inplace=True)
####iv) RENAME COLUMNS
# Load the dataset
data_nowspace = pd.read_excel("Dropped_Handled_Drug_Overdose_Data.xlsx")

# Rename the "Location" column to "Death Place"
data_nowspace.rename(columns={"Location": "Death Place"}, inplace=True)

# Save the cleaned dataset
data_nowspace.to_excel("Renamed_Accidental_drug_deaths.xlsx", index=False)
v) FILL MISSING VALUES IN SUBSTANCE DETAILS WITH "N/F"
# Load the dataset
df = pd.read_excel("Renamed_Accidental_drug_deaths.xlsx")

# List of substance columns
substance_columns = [
    "Heroin", "Heroin death certificate (DC)", "Cocaine", "Fentanyl", "Fentanyl Analogue",
    "Oxycodone", "Oxymorphone", "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone",
    "Meth/Amphetamine", "Amphet", "Tramad", "Hydromorphone", "Morphine (Not Heroin)",
    "Xylazine", "Gabapentin", "Opiate NOS", "Heroin/Morph/Codeine"
]

# Replace blank cells with 'N/F'
df[substance_columns] = df[substance_columns].fillna("N/F")

# Save the cleaned dataset
df.to_excel("Filled_Accidental_drug_deaths.xlsx", index=False)

# Confirm missing values are handled
print("Missing Values after handling:\n", df.isnull().sum())
vi) OUTLIER DETECTION
# Selecting the numerical column 'Age'
numerical_columns = ['Age']

# Convert 'Age' column to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')

# Compute IQR for Outlier Detection
Q1 = df['Age'].quantile(0.25)  # First quartile (25th percentile)
Q3 = df['Age'].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile range

# Define lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detecting outliers
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]

# Display the youngest and oldest ages
youngest = df['Age'].min()
oldest = df['Age'].max()

print(f"Youngest Victim: {youngest}")
print(f"Oldest Victim: {oldest}")

# Display detected outliers
print("Outliers Detected:")
print(outliers[['Age']])
vii) DATA MAPPING:
# Load the dataset
df = pd.read_excel("Filled_Accidental_drug_deaths.xlsx")

# Define the mapping dictionary for "Race"
mapping_race = {
    "Asian/Indian": "Asian",
    "Asian, Other": "Asian",
    "Asian Indian": "Asian",
    "Black or African American": "Black",
    "Black or African American / American Indian Lenni Lenape": "Black",
    "Native American, Other": "American",
    "American Indian or Alaska Native": "American or Alaskan",
    "Other (Specify) Haitian": "Other",
    "Other (Specify) portugese, Cape Verdean": "Other",
    "Other (Specify) Puerto Rican": "Other",
    "Other Asian": "Asian",
    "Other Asian (Specify)": "Asian",
    "Blanks": "Unknown"
}

# Apply the mapping to the "Race" column
df["Race"] = df["Race"].replace(mapping_race)

# Rename "Location" column to "Death Place"
df.rename(columns={"Location": "Death Place"}, inplace=True)

# Define the mapping dictionary for "Death Place"
mapping_death_place = {
    "Decedentâ€™s Home": "Decedent's Home",
    "Decedent’s Home": "Decedent's Home",
    "Hiospital": "Hospital",
    "Hospice Facility": "Hospice",
    "Other (Specify)": "Other"
}

# Apply the mapping to the "Death Place" column
df["Death Place"] = df["Death Place"].replace(mapping_death_place)

# List of drug presence columns
drug_columns = [
    "Heroin", "Heroin death certificate (DC)", "Cocaine", "Fentanyl", "Fentanyl Analogue",
    "Oxycodone", "Oxymorphone", "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone",
    "Meth/Amphetamine", "Amphet", "Tramad", "Hydromorphone", "Morphine (Not Heroin)",
    "Xylazine", "Gabapentin", "Opiate NOS", "Heroin/Morph/Codeine"
]

# Replace blank cells with 0 and 'Y' with 1, all else becomes NaN → convert to 0
df[drug_columns] = df[drug_columns].replace({"Y": 1}).fillna(0)

# Convert all values to integers, setting non-numeric values to 0
df[drug_columns] = df[drug_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

# Apply the mapping to the "Description of Injury" column
df["Race"] = df["Race"].replace(mapping_race)


# Save the cleaned dataset
df.to_excel("Cleaned_v1_Accidental_drug_deaths.xlsx", index=False)

# Load the dataset
df = pd.read_excel("Cleaned_v1_Accidental_drug_deaths.xlsx")

# Define mapping for different categories of drug use and related features
Description_of_Injury_mapping = {
    # Drug Use (General)
    "Drug use": "Drug Use (General)",
    "Used Drugs": "Drug Use (General)",
    "Drug Abuse": "Drug Use (General)",
    "drug use": "Drug Use (General)",
    "multiple drug use": "Drug Use (General)",
    "Used illicit drugs": "Drug Use (General)",
    "Used Opiates": "Drug Use (General)",
    "DrugUse": "Drug Use (General)",
    "Used illicit and prescription drugs": "Drug Use (General)",
    "Substance use disorder": "Drug Use (General)",

    # Specific Drug Use - Cocaine
    "Used Cocaine": "Cocaine Use",
    "Cocaine use": "Cocaine Use",
    "Used cocaine": "Cocaine Use",
    "Recent cocaine use": "Cocaine Use",
    "Usage of Cocaine and Heroin": "Cocaine & Heroin Use",
    "Used Cocaine and Heroin": "Cocaine & Heroin Use",
    "Took cocaine": "Cocaine Use",

    # Specific Drug Use - Heroin
    "Used Heroin": "Heroin Use",
    "Heroin Overdose": "Heroin Use",
    "Heroin use": "Heroin Use",
    "Acute Heroin Toxicity": "Heroin Use",

    # Specific Drug Use - Fentanyl
    "Used fentanyl": "Fentanyl Use",
    "Took fentanyl": "Fentanyl Use",
    "Fentanyl Use": "Fentanyl Use",
    "Took Fentanyl and Ethanol": "Fentanyl & Alcohol Use",
    "Misuse of Fentanyl Patch": "Fentanyl Use",

    # Specific Drug Use - Methadone
    "Used Methadone": "Methadone Use",
    "Used methadone": "Methadone Use",
    "Used Methanone and Ethanol": "Methadone & Alcohol Use",

    # Specific Drug Use - Oxycodone
    "Used Oxycodone": "Oxycodone Use",
    "Ingested Oxycodone and Alcohol": "Oxycodone & Alcohol Use",
    "Used oxycodone": "Oxycodone Use",

    # Specific Drug Use - Morphine
    "Used Morphine and Ethanol": "Morphine & Alcohol Use",
    "USED MORPHINE": "Morphine Use",

    # Specific Drug Use - Hydrocodone
    "Used Hydrocodone": "Hydrocodone Use",
    "Used Methadone and Hydrocodone": "Methadone & Hydrocodone Use",

    # Specific Drug Use - Benzodiazepines
    "Fentanyl and Benzodiazepine Use": "Benzodiazepine Use",
    "Alcohol and Benzodiazepine use": "Benzodiazepine & Alcohol Use",

    # Prescription/Medication Abuse
    "Prescription Medicine Abuse": "Prescription Abuse",
    "Prescription drug use": "Prescription Abuse",
    "Abused prescription medications": "Prescription Abuse",
    "Prescription Medication Abuse": "Prescription Abuse",
    "prescription medication abuse": "Prescription Abuse",
    "Prescription Medicine Misuse": "Prescription Abuse",
    "Medication Misuse": "Prescription Abuse",
    "Took prescription medication": "Prescription Abuse",
    "Used prescription medications": "Prescription Abuse",
    "Took prescribed medications": "Prescription Abuse",
    "Took prescription medications, synthetic opioid, and ethanol": "Prescription & Synthetic Opioid Use",
    "Used multiple medications": "Multiple Medication Use",
    "Used Multiple Medications": "Multiple Medication Use",
    "Took multiple medications": "Multiple Medication Use",
    "Ingested multiple medications": "Multiple Medication Use",

    # Substance Abuse (General)
    "Substance abuse": "Substance Abuse",
    "Substance Abuse": "Substance Abuse",
    "Substance sue": "Substance Abuse",
    "Substance Use": "Substance Abuse",
    "Substances Abuse": "Substance Abuse",
    "Substance Abuse Including Intravenous Injection": "Substance Abuse & Injection",
    "Substance Abuse Including Injection of Heroin": "Substance Abuse & Heroin Injection",
    "Acute and Chronic Substance Use": "Chronic Substance Abuse",
    "Acute and chronic substance use disorder": "Chronic Substance Abuse",

    # Alcohol and Drug Combination
    "Used alcohol and mitragynine": "Alcohol & Drug Combination",
    "Consumed ethanol with prescription medications": "Alcohol & Drug Combination",
    "Combined Alcohol and Medications": "Alcohol & Drug Combination",
    "Took ethanol and fentanyl": "Alcohol & Drug Combination",
    "Combined medication and substance ingestion": "Alcohol & Drug Combination",
    "Alcohol and substance abuse": "Alcohol & Drug Combination",
    "Ingested Multiple Medications and Alcohol": "Alcohol & Drug Combination",
    "Took medications and alcohol": "Alcohol & Drug Combination",
    "Alcohol and Medication Ingestion": "Alcohol & Drug Combination",
    "Combined alcohol and medication": "Alcohol & Drug Combination",

    # Route of Administration
    "Ingestion": "Ingestion",
    "Ingested drugs": "Ingestion",
    "Ingested medications": "Ingestion",
    "Ingested prescription medication": "Ingestion",
    "Injection": "Injection",
    "substance abuse (injection)": "Injection",
    "Intravenous drug abuse": "Injection",
    "Inhalation": "Inhalation",
    "Inhalation/Ingestion": "Inhalation & Ingestion",
    "Huffed Propellant": "Inhalation",
    "Used Fentanyl Patches": "Dermal Absorption",
    "Transdermal Absorption": "Dermal Absorption",

    # Overdose/Toxicity
    "Accidental Drug Overdose": "Toxicity",
    "Overdose": "Toxicity",
    "Acute and Chronic Alcohol/Substance Use Disorder": "Toxicity",
    "Toxic effects of ethanol and fentanyl": "Toxicity",
    "Toxic effects of ethanol and prescription medications": "Toxicity",

    # Unusual Cases
    "Drowned In Hot Tub While Intoxicated": "Unusual Case",
    "Drowned In Bathtub": "Unusual Case",
    "Submerged in bathtub while intoxicated": "Unusual Case",
    "Collapsed with trash can and plastic bag over face while intoxicated": "Unusual Case",
    "Swallowed bag of drug while in police custody": "Unusual Case",

    # Unknown/Missing Data
    "Unknown": "Reason Unknown",
    None: "Reason Unknown",
    float("nan"): "Reason Unknown"
}

# Update the feature column with mapped values
df["Description of Injury"] = df["Description of Injury"].map(lambda x: Description_of_Injury_mapping.get(str(x).strip(), "Reason Unknown"))
df.head()

# Save the cleaned dataset
df.to_excel("Cleaned_v2_Accidental_drug_deaths.xlsx", index=False)
**2. DATA INTEGRATION**
i) Handling redundancy using a Correlation Heatmap
# Load dataset
file_path = "Filled_Accidental_drug_deaths.xlsx"
data = pd.read_excel(file_path)

# Convert "Y" values in substance columns to 1
substance_columns = ["Heroin", "Heroin death certificate (DC)","Cocaine","Fentanyl","Fentanyl Analogue","Oxycodone","Oxymorphone","Ethanol","Hydrocodone","Benzodiazepine","Methadone","Meth/Amphetamine","Amphet","Tramad","Hydromorphone","Morphine (Not Heroin)","Xylazine","Gabapentin","Opiate NOS","Heroin/Morph/Codeine","Other Opioid","Any Opioid"]
for col in substance_columns:
    if col in data.columns:
        data[col] = data[col].astype(str).str.strip().str.upper().map({'Y': 1}).fillna(0)

# Select numeric columns
numeric_data = data.select_dtypes(include=['number'])

if len(numeric_data.columns) > 1:
    # Compute correlation matrix
    correlation_matrix = numeric_data.corr()

    # Heatmap
    plt.figure(figsize=(12, 10))  # Increase figure size
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})  # Reduce font size
    plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=8)  # Reduce y-axis font size
    plt.title("Correlation Heatmap", fontsize=14)  # Adjust title size
    plt.show()
else:
    print("Not enough numeric columns for correlation analysis.")
**3. DATA TRANSFORMATION**
i) FEATURE ENGINEERING
# Load the dataset
df = pd.read_excel("Cleaned_v2_Accidental_drug_deaths.xlsx")

# Function to extract latitude and longitude
def extract_lat_lon(geo_column):
    if pd.isna(geo_column) or not isinstance(geo_column, str):
        return pd.Series([None, None])  # Return None for missing values
    match = re.search(r"\(([-\d.]+), ([-\d.]+)\)", geo_column)
    return pd.Series(match.groups()) if match else pd.Series([None, None])

# Apply extraction to each location column
for col in ["ResidenceCityGeo", "InjuryCityGeo", "DeathCityGeo"]:
    df[[f"{col}_Latitude", f"{col}_Longitude"]] = df[col].apply(extract_lat_lon)

# Fill missing values using mode for each column
for col in df.columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop original columns
df.drop(columns=["ResidenceCityGeo", "InjuryCityGeo", "DeathCityGeo"], inplace=True)

# Save cleaned dataset
df.to_excel("Cleaned_v3_Accidental_Drug_deaths.xlsx", index=False)

# Display sample output
print(df.head())
ii) STANDARDIZE "Age" COLUMN
# Convert Age column to numeric
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")

# Fill missing Age values with median
age_median = df["Age"].median()
df["Age"].fillna(age_median, inplace=True)

# Initialize the scaler
scaler = StandardScaler()

# Standardize the Age column and store it in a new column
df["Standardized_Age"] = scaler.fit_transform(df[["Age"]])

# Save the updated dataset
df.to_excel("Final_Accidental_Drug_deaths.xlsx", index=False)

## **FINAL CLEANED DATASET**
# Load the data
uploaded = files.upload()
cleandata = list(uploaded.keys())[0]

# Read the uploaded file into a DataFrame
data = pd.read_excel(cleandata)

#Convert into a dataframe
df = pd.DataFrame(data)
df
## **FINAL OVERVIEW OF MISSING VALUES AFTER PRE-PROCESSING**


# Load the dataset
df = pd.read_excel("Final_Accidental_Drug_deaths.xlsx")
# Confirm missing values are handled
print("Missing Values after handling:\n", df.isnull().sum())
###DATA VISUALIZATION
### **BEFORE PRE-PROCESSING**
#### 1. Identifying the Youngest and Oldest Victims in Overdose Cases (BOXPLOT)
# Load dataset
file_path = "Accidental_Drug_Related_Deaths.csv"
data = pd.read_csv(file_path)

# Boxplot to visualize outliers in the Age column
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Age'])

# Title and display
plt.title('Boxplot for Outlier Detection (Before Preprocessing)')
plt.show()
#### 2. Identifying the Most Common Substances in Drug-Related Deaths (Bar Chart)
# Load dataset
file_path = 'Accidental_Drug_Related_Deaths.csv'
data = pd.read_csv(file_path)

# List of substance-related columns
substances = ["Heroin", "Heroin death certificate (DC)", "Cocaine", "Fentanyl", "Fentanyl Analogue",
              "Oxycodone", "Oxymorphone", "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone",
              "Meth/Amphetamine", "Amphet", "Tramad", "Hydromorphone", "Morphine (Not Heroin)",
              "Xylazine", "Gabapentin", "Opiate NOS", "Heroin/Morph/Codeine", "Other Opioid", "Any Opioid"]

# Check which substance columns exist in the dataset
existing_substances = [col for col in substances if col in data.columns]

# Convert all values to lowercase and strip spaces
for col in existing_substances:
    data[col] = data[col].astype(str).str.strip().str.lower()

# Count occurrences of "Y" in each substance column
substance_counts = data[existing_substances].apply(lambda x: (x == 'y').sum())

# Bar chart visualization
plt.figure(figsize=(12, 6))
sns.barplot(x=substance_counts.index, y=substance_counts.values, color='darkblue')

# Titles and labels
plt.title("Most Common Substances in Drug-Related Deaths")
plt.xlabel("Substance Type")
plt.ylabel("Frequency")
plt.xticks(rotation=90)  # Rotate labels for readability
plt.show()
#### 3. Fentanyl, Heroin & Cocaine Consumption by Age Group (Stacked Bar chart)
# Load dataset
file_path = "Accidental_Drug_Related_Deaths.csv"
data = pd.read_csv(file_path)

# Define substances
substances = ["Heroin", "Fentanyl","Cocaine"]

# Create Age Groups
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61+"]
data["Age Group"] = pd.cut(data["Age"], bins=bins, labels=labels, right=False)

# Convert "Y" values to 1, else 0
for col in substances:
    data[col] = data[col].astype(str).str.strip().str.upper().map({'Y': 1}).fillna(0)

# Count Fentanyl & Heroin usage per age group
age_substance_counts = data.groupby("Age Group")[substances].sum()

# Plot stacked bar chart
ax = age_substance_counts.plot(kind="bar", stacked=True, figsize=(10, 6), color=["#FF9999", "#66B3FF","#FFA500"])

# Titles and labels
plt.title("Fentanyl, Heroin & Cocaine Consumption by Age Group", fontsize=14)
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Number of Cases", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title="Substance", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show chart
plt.show()
####4. Examining Repeated Overdose Cases from Residence Data (Barchart)
# Load dataset
file_path = "Accidental_Drug_Related_Deaths.csv"
data = pd.read_csv(file_path)

# Check if "ResidenceCityGeo" exists in dataset
if "ResidenceCityGeo" in data.columns:
    # Extract city names from "ResidenceCityGeo" column (everything before the coordinates)
    data["Residence City"] = data["ResidenceCityGeo"].apply(lambda x: re.split(r'\s*\(', str(x))[0])

    # Count the number of overdose cases per city
    city_counts = data["Residence City"].value_counts().head(20)  # Top 20 cities

    # Plot bar chart
    plt.figure(figsize=(12, 6))
    sns.barplot(x=city_counts.index, y=city_counts.values, color='darkblue')

    # Titles & Labels
    plt.title("Top 20 Cities with Most Overdose Cases")
    plt.xlabel("Residence City")
    plt.ylabel("Number of Overdose Cases")
    plt.xticks(rotation=90)  # Rotate labels for better visibility
    plt.show()
### **AFTER PRE-PROCESSING**
####1. Identifying the Youngest and Oldest Victims in Overdose Cases (Histogram)
# Load cleaned dataset
file_path = "Final_Accidental_Drug_deaths.xlsx"
data = pd.read_excel(file_path)

# Convert 'Age' column to numeric, handling errors
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Drop NaN values in 'Age' column
data = data.dropna(subset=['Age'])

# Remove the outlier (Age = 87)
data = data[data['Age'] != 87]

# Set figure size
plt.figure(figsize=(10, 6))

# Create histogram with KDE curve
sns.histplot(data['Age'], bins=10, kde=True, color='blue', edgecolor='black')

# Set x-axis labels
plt.xticks(range(10, 100, 10))

# Set y-axis labels manually
plt.yticks([100, 500, 1000, 1500, 2000, 2500])

# Labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Youngest and Oldest Victims in Overdose Cases')

# Show plot
plt.show()
####2. Identifying the Most Common Substances in Drug-Related Deaths (Dual-Bar Chart)
# Load cleaned dataset
file_path = "Final_Accidental_Drug_deaths.xlsx"
data = pd.read_excel(file_path)

# List of substance-related columns
substances = ["Heroin", "Fentanyl", "Cocaine", "Oxycodone", "Methadone", "Benzodiazepine",
              "Ethanol", "Meth/Amphetamine", "Tramad", "Hydromorphone", "Morphine (Not Heroin)"]

# Count the number of 1s (presence) and 0s (absence) for each substance
substance_counts = data[substances].apply(pd.Series.value_counts).fillna(0)

# Rename index for clarity
substance_counts.index = ["Absent (0)", "Present (1)"]

# Plot the dual bar chart
plt.figure(figsize=(12, 6))
substance_counts.T.plot(kind="bar", stacked=False, figsize=(12, 6), color=["orange", "darkblue"])

# Titles and labels
plt.title(" Identifying the Most Common Substances in Drug-Related Deaths (Dual-Bar Chart)")
plt.xlabel("Substance Type")
plt.ylabel("Frequency")
plt.xticks(rotation=90)  # Rotate labels for readability
plt.legend(title="Legend", labels=["Absent", "Present"])

# Show the plot
plt.show()
#### 3. Fentanyl, Heroin & Cocaine Consumption by Age Group (Line chart)
# Load the dataset
file_path = "Final_Accidental_Drug_deaths.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Convert Age column to numeric, dropping non-numeric values
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df = df.dropna(subset=['Age'])

drug_columns = ['Fentanyl', 'Heroin/Morph/Codeine', 'Cocaine']

df_grouped = df.groupby('Age')[drug_columns].sum()

# Plot the line chart
plt.figure(figsize=(10, 6))
for drug in drug_columns:
    plt.plot(df_grouped.index, df_grouped[drug], label=drug)

plt.xlabel("Age Group")
plt.ylabel("Consumption Count")
plt.title("Fentanyl, Heroin & Cocaine Consumption by Age Group")
plt.legend()
plt.grid()
plt.show()
####4. Examining Repeated Overdose Cases from Residence Data (Piechart)
# Load the dataset
file_path = "Final_Accidental_Drug_deaths.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Count repeated overdose cases by Residence City
df_grouped = df['Residence City'].value_counts()

# Plot the pie chart
plt.figure(figsize=(10, 10))
df_grouped.head(10).plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='Paired')

plt.ylabel("")
plt.title("Repeated Overdose Cases by Residence City")
plt.show()
