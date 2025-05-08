#import libraries
import pandas as pd
import pickle

# Add this near the top of your file after imports
pd.set_option('future.no_silent_downcasting', True)

#Manually encode the categorical & binary columns
hashmap = {
"Never": 0,
"Sometimes": 1,
"Frequently": 2,
"Always": 3,

"None":0,

"No Activity": 0,
"up to 2": 1,
"up to 5": 2,
"more than 5": 3,

"less than 1": 1,
"1 to 2": 2,
"more than 2": 3,
"3 to 4": 4,
"5 or more": 5,

"Bicycle": 1,
"Car": 4,
"Motorbike": 3,
"Public": 2,
"Walk": 0,

"no": 0,
"yes": 1,

"Male": 0,
"Female": 1
}

def classify_bmi(df):
    """Calculate BMI classification based on age, height and weight"""
    # Create a copy
    df = df.copy()
    
    # Calculate BMI for all rows
    bmi = df['weight'] / (df['height'] ** 2)
    
    # Initialize results array
    classifications = pd.Series(index=df.index, dtype=int)
    
    # Create children (2-19 years) and adult (20-64 years) masks 
    children_mask = (df['age'] >= 2) & (df['age'] < 20)
    adult_mask = (df['age'] >= 20) & (df['age'] < 65)
    
    # Classify children
    children_class = pd.Series(index=df.index, dtype=int)
    children_class[bmi < 14] = 0
    children_class[(bmi >= 14) & (bmi < 18)] = 1
    children_class[(bmi >= 18) & (bmi < 21)] = 2
    children_class[bmi >= 21] = 3
    
    # Classify adults
    adult_class = pd.Series(index=df.index, dtype=int)
    adult_class[bmi < 18.5] = 0
    adult_class[(bmi >= 18.5) & (bmi < 25)] = 1
    adult_class[(bmi >= 25) & (bmi < 30)] = 2
    adult_class[(bmi >= 30) & (bmi < 35)] = 3
    adult_class[(bmi >= 35) & (bmi < 40)] = 4
    adult_class[bmi >= 40] = 5
    
    # Combine classifications
    classifications[children_mask] = children_class[children_mask]
    classifications[adult_mask] = adult_class[adult_mask]
    return classifications

def add_life_score(df):
    """Calculate lifestyle score based on multiple factors"""
    life_columns = [
        'alcohol_freq',
        'caloric_freq',
        'devices_perday',
        'eat_between_meals',
        'monitor_calories',
        'physical_activity_perweek',
        'smoke',
        'transportation',
        'veggies_freq',
        'water_daily'
    ]

    df["life"] = 0
    for column in life_columns:
        if column in df.columns:
            df["life"] += df[column]
    return df

def process_and_predict(input_df: pd.DataFrame) -> str:

    # Make a copy to avoid modifying the original
    X = input_df.copy()
    
    # Encode categorical columns
    categorical_columns = ['alcohol_freq', 'caloric_freq', 'devices_perday', 'eat_between_meals',
       'gender', 'monitor_calories', 'parent_overweight',
       'physical_activity_perweek', 'smoke', 'transportation', 'veggies_freq',
       'water_daily']

    # Then use replace without infer_objects
    for column in categorical_columns:
        if column in X.columns:
            X[column] = X[column].replace(hashmap)
    
    # Calculate BMI class
    X['bmi_class'] = classify_bmi(X)
    
    # Add life score
    add_life_score(X)

    # Load feature scaler
    try:
        with open(r'objects/feature_scaler.pkl', 'rb') as f_scaler:
            scaler = pickle.load(f_scaler)
    except FileNotFoundError as e:
        raise Exception(f"Feature scaler file not found: {str(e)}")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Load the model
    try:
        with open(r'objects/model.pkl', 'rb') as f_model:
            model = pickle.load(f_model)
    except FileNotFoundError as e:
        raise Exception(f"Model file not found: {str(e)}")
    
    # Make prediction
    prediction = model.predict(X_scaled)
    
    # Convert prediction to category
    hash_obesity_inverted = {
        1: 'Normal_Weight',
        2: 'Overweight_Level_I',
        3: 'Overweight_Level_II',
        4: 'Obesity_Type_I',
        5: 'Insufficient_Weight',
        6: 'Obesity_Type_II',
        7: 'Obesity_Type_III'
    }
    
    # Return prediction as string (for a single input, we get the first element)
    return hash_obesity_inverted[prediction[0]]

# Usage example:
# user_data = pd.DataFrame({
#     'age': [21], 
#     'alcohol_freq': ['Never'],
#     'caloric_freq': ['no'],
#     'devices_perday': ['up to 5'],
#     'eat_between_meals': ['Sometimes'],
#     'gender': ['Female'],
#     'height': [1.62], 
#     'meals_perday': [3],
#     'monitor_calories': ['no'],
#     'parent_overweight': ['yes'],
#     'physical_activity_perweek': ['None'], 
#     'siblings': [3],
#     'smoke': ['no'],
#     'transportation': ['Public'],
#     'veggies_freq': ['Sometimes'],
#     'water_daily': ['1 to 2'],
#     'weight': [80]
# })

# prediction = process_and_predict(user_data)
# print(f"Predicted obesity level: {prediction}")