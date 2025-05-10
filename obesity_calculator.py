import streamlit as st
import pandas as pd
from model import process_and_predict

def main():
    st.set_page_config(layout="wide")

    # Custom CSS to style selectbox dropdowns
    st.markdown("""
    <style>
        /* Target the popover content of selectboxes - the dropdown list itself */
        div[data-baseweb="popover"][aria-label="Select an option"] > div {
            background-color: #f8f9fa !important; /* Very light grey background */
        }

        /* Target individual options on hover */
        div[data-baseweb="popover"][aria-label="Select an option"] ul li:hover {
            background-color: #e9ecef !important; /* Slightly darker grey for hover */
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Welcome to the Obesity Calculator")

    st.header("Please fill in your details:")
    st.markdown("---") # Separator

    classification_details = {
        "Insufficient_Weight": {
            "name": "Insufficient Weight",
            "description": "Your overall body score suggests you are below a healthy weight range. It's important to ensure you are getting adequate nutrition and maintain a healthy lifestyle. Consider consulting a healthcare provider or a nutritionist for guidance.",
            "emoji": "😟"
        },
        "Normal_Weight": {
            "name": "Normal Weight",
            "description": "Congratulations! Your overall body score falls within a healthy weight range. Maintaining a balanced diet, regular physical activity, and healthy habits will help you stay in this positive zone.",
            "emoji": "👍"
        },
        "Overweight_Level_I": {
            "name": "Overweight (Level I)",
            "description": "Your overall body score indicates you are slightly above a healthy weight range. This is a good opportunity to focus on positive lifestyle changes, such as incorporating more physical activity and making mindful food choices to improve your overall health.",
            "emoji": "🙁"
        },
        "Overweight_Level_II": {
            "name": "Overweight (Level II)",
            "description": "Your overall body score suggests you are moderately above a healthy weight range. It's advisable to consider lifestyle modifications, including dietary adjustments and increased physical activity. Consulting a healthcare professional can provide personalized guidance for better health.",
            "emoji": "😥"
        },
        "Obesity_Type_I": {
            "name": "Obesity (Class I)",
            "description": "Your overall body score falls into Obesity Class I. This indicates a significant health risk. It is highly recommended to consult with a healthcare provider to discuss a comprehensive plan for health management, which may include dietary changes, exercise programs, and other interventions.",
            "emoji": "🚨"
        },
        "Obesity_Type_II": {
            "name": "Obesity (Class II)",
            "description": "Your overall body score is classified as Obesity Class II, which is associated with serious health risks. Professional medical advice is crucial. A healthcare provider can help you develop a safe and effective health management plan tailored to your needs.",
            "emoji": "❗"
        },
        "Obesity_Type_III": {
            "name": "Obesity (Class III - Severe Obesity)",
            "description": "Your overall body score indicates Obesity Class III, also known as severe obesity. This condition carries very high health risks. Immediate consultation with a healthcare professional is essential to explore intensive health management strategies and support.",
            "emoji": "🆘"
        }
    }

    # Mappings for clearer selectbox options
    eat_between_meals_map = {
       "Never (0 times per day)": "Never",
       "Occasionally (1-2 times per day)": "Sometimes",
       "Frequently (3-4 times per day)": "Frequently",
       "Almost Always (5+ times per day)": "Always"
    }

    veggies_freq_map = {
       "Rarely or Never in main meals": "Never",
       "Sometimes (in about half of main meals)": "Sometimes",
       "Usually (in most or all main meals)": "Always"
    }

    alcohol_freq_map = {
       "Never or Almost Never": "Never",
       "Occasionally (e.g., a few times a month)": "Sometimes",
       "Regularly (e.g., 2-4 times a week)": "Frequently",
       "Often (most days)": "Always"
    }

    water_daily_map = {
        "Less than 1 liter": "less than 1",
        "1 to 2 liters": "1 to 2",
        "More than 2 liters": "more than 2"
    }

    devices_perday_map = {
        "Up to 2 hours per day": "up to 2",
        "2 to 5 hours per day": "up to 5", # Adjusted to be MECE if original was just "up to 5"
        "More than 5 hours per day": "more than 5"
    }

    transportation_map = {
        "Public Transport (Bus/Train/Subway)": "Public",
        "Private Car": "Car",
        "Bicycle": "Bicycle",
        "Motorbike/Scooter": "Motorbike",
        "Walking": "Walk"
    }

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Input your age (between 15-65):", min_value=15, max_value=65, step=1, key="age")
        gender = st.selectbox(
            "Select your gender:",
            ('Female', 'Male'),
            key="gender"
        )
        height = st.number_input("Enter your height (in meters, e.g., 1.75):", min_value=0.5, max_value=2.5, value=1.62, step=0.01, key="height")
        weight = st.number_input("Enter your weight (in kg, e.g., 70.5):", min_value=10.0, max_value=300.0, value=80.0, step=0.5, key="weight")
        siblings = st.number_input("Number of siblings:", min_value=0, max_value=20, value=1, step=1, key="siblings")
        parent_overweight = st.selectbox(
            "Does one or both of your parents have a history of being overweight?",
            ('yes', 'no'),
            key="parent_overweight"
        )

    with col2:
        st.subheader("Lifestyle & Habits")
        meals_perday = st.selectbox(
            "How many main meals do you typically eat per day?",
            (1, 2, 3, 4, 5),
            index=2,  # Default to 3 meals
            key="meals_perday"
        )
        
        # Eat Between Meals - updated
        eat_between_meals_display = st.selectbox(
            "How often do you eat or snack between meals?",
            options=list(eat_between_meals_map.keys()),
            index=1, # Corresponds to original default "Sometimes"
            key="eat_between_meals_selection" # Changed key to avoid state confusion if needed
        )
        eat_between_meals = eat_between_meals_map[eat_between_meals_display]

        # Veggies Frequency - updated
        veggies_freq_display = st.selectbox(
            "How often do you include vegetables in your main meals?",
            options=list(veggies_freq_map.keys()),
            index=1, # Corresponds to original default "Sometimes"
            key="veggies_freq_selection"
        )
        veggies_freq = veggies_freq_map[veggies_freq_display]

        # Water Daily - updated
        water_daily_display = st.selectbox(
            "How much water do you drink daily?",
            options=list(water_daily_map.keys()),
            index=1, # Corresponds to original default "1 to 2"
            key="water_daily_selection"
        )
        water_daily = water_daily_map[water_daily_display]

        monitor_calories = st.selectbox(
            "Do you monitor the calories you consume daily?",
            ('no', 'yes'),
            key="monitor_calories"
        )
        caloric_freq = st.selectbox( 
            "Do you frequently eats sweets/chocolate?",
            ('no', 'yes'), 
            index=0, # Default to 'no'
            key="caloric_freq"
        )

        # Alcohol Frequency - updated
        alcohol_freq_display = st.selectbox(
            "How often do you consume alcohol?",
            options=list(alcohol_freq_map.keys()),
            index=0, # Corresponds to original default "Never" (implicit index 0)
            key="alcohol_freq_selection"
        )
        alcohol_freq = alcohol_freq_map[alcohol_freq_display]

        smoke = st.selectbox(
            "Do you smoke regularly?",
            ('no', 'yes'),
            key="smoke"
        )
        physical_activity_perweek = st.selectbox(
            "How many days per week do you engage in physical activity (at least 30 mins)?",
            ('None', '1 to 2', '3 to 4', '5 or more'),
            key="physical_activity_perweek"
        )
        
        # Devices Per Day - updated
        devices_perday_display = st.selectbox( 
            "How much time do you spend using technological devices (phone, computer, TV) per day?",
            options=list(devices_perday_map.keys()),
            index=1, # Original default was index 2 for ('up to 2', 'up to 5', 'more than 5'), 
                     # so for the new map, "2 to 5 hours per day" (index 1) should match "up to 5"
            key="devices_perday_selection"
        )
        devices_perday = devices_perday_map[devices_perday_display]
        
        # Transportation - updated
        transportation_display = st.selectbox(
            "What is your usual mode of transportation?",
            options=list(transportation_map.keys()),
            index=0, # Assuming original default was the first item, e.g., "Public"
            key="transportation_selection"
        )
        transportation = transportation_map[transportation_display]

    st.markdown("---") # Separator

    # Button to collect and display data
    if st.button("📝 Find out your obesity level", type="primary"):
        user_data = pd.DataFrame({
            'age': [age],
            'alcohol_freq': [alcohol_freq],
            'caloric_freq': [caloric_freq],
            'devices_perday': [devices_perday],
            'eat_between_meals': [eat_between_meals],
            'gender': [gender],
            'height': [height],
            'meals_perday': [meals_perday],
            'monitor_calories': [monitor_calories],
            'parent_overweight': [parent_overweight],
            'physical_activity_perweek': [physical_activity_perweek],
            'siblings': [siblings],
            'smoke': [smoke],
            'transportation': [transportation], # Ensure consistency
            'veggies_freq': [veggies_freq],
            'water_daily': [water_daily],
            'weight': [weight]
        })

        classification_key = process_and_predict(user_data)


        if classification_key in classification_details:
            details = classification_details[classification_key]
            
            st.header(f"Your Result: {details['name']} {details['emoji']}")
            
            st.info(f"**What this means:** {details['description']}")

        else:
            st.error("Could not determine your classification based on the output from the model.")
            st.write(f"Raw classification output: {classification_key}")
            # BMI reporting in error message removed
            st.warning("Please ensure the model output matches one of the expected classifications: " + ", ".join(classification_details.keys()))

if __name__ == '__main__':
    main()
