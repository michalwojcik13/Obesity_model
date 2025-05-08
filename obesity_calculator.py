import streamlit as st
import pandas as pd
from model import process_and_predict

def main():
    st.set_page_config(layout="wide")
    st.title("📊 User Data Collection Form")

    st.header("Please fill in your details:")

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
        eat_between_meals = st.selectbox(
            "How often do you eat between meals (snacking)?",
            ('Never', 'Sometimes', 'Frequently', 'Always'),
            index=1, # Default to Sometimes
            key="eat_between_meals"
        )
        veggies_freq = st.selectbox(
            "How often do you include vegetables in your main meals?",
            ('Never', 'Sometimes', 'Always'),
            index=1, # Default to Sometimes
            key="veggies_freq"
        )
        water_daily = st.selectbox(
            "How much water do you drink daily?",
            ('less than 1', '1 to 2', 'more than 2'),
            index=1, # Default to 1 to 2 liters
            key="water_daily"
        )
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
        alcohol_freq = st.selectbox(
            "How often do you consume alcohol?",
            ('Never', 'Sometimes', 'Frequently', 'Always'),
            key="alcohol_freq"
        )
        smoke = st.selectbox(
            "Do you smoke?",
            ('no', 'yes'),
            key="smoke"
        )
        physical_activity_perweek = st.selectbox(
            "How many days per week do you engage in physical activity (at least 30 mins)?",
            ('None', '1 to 2', '3 to 4', '5 or more'),
            key="physical_activity_perweek"
        )
        devices_perday = st.selectbox( # Assuming this means time spent on devices
            "How much time do you spend using technological devices (phone, computer, TV) per day?",
            ('up to 2', 'up to 5', 'more than 5'), # Added more granular options
            index=2, # Default to 'up to 5 hours' as per original
            key="devices_perday"
        )
        transportation = st.selectbox(
            "What is your usual mode of transportation?",
            ('Public', 'Car', 'Bicycle', 'Motorbike', 'Walk'),
            key="transportation"
        )

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
            'transportation': [transportation.replace(" ", "_")], # Ensure consistency
            'veggies_freq': [veggies_freq],
            'water_daily': [water_daily],
            'weight': [weight]
        })

        st.subheader("Collected User Data:")
        st.dataframe(user_data)

        prediction = process_and_predict(user_data)

        st.write("You have:")
        st.write(prediction)

if __name__ == '__main__':
    main()
