import streamlit as st
import pandas as pd
import joblib

model = joblib.load("restaurant_rating_model.pkl")

def predict_rating(has_table_booking, has_online_delivery, is_delivering_now, price_range, votes, average_cost_for_two, cuisine_count):
    table_booking = 1 if has_table_booking.lower() == "yes" else 0
    online_delivery = 1 if has_online_delivery.lower() == "yes" else 0
    delivering_now = 1 if is_delivering_now.lower() == "yes" else 0

    data = {
        'Has Table booking': [table_booking],
        'Has Online delivery': [online_delivery],
        'Is delivering now': [delivering_now],
        'Price range': [price_range],
        'Votes': [votes],
        'Average Cost for two': [average_cost_for_two],
        'Cuisine_Count': [cuisine_count]
    }

    df = pd.DataFrame(data)
    predicted_rating = model.predict(df)[0]
    threshold = 3.8
    performance = "Perform" if predicted_rating >= threshold else "Not Perform"

    return predicted_rating, performance

# Streamlit UI
st.title("Restaurant Rating Prediction App")

st.write("### Enter restaurant details:")

has_table_booking = st.selectbox("Has Table Booking?", ["yes", "no"])
has_online_delivery = st.selectbox("Has Online Delivery?", ["yes", "no"])
is_delivering_now = st.selectbox("Is Delivering Now?", ["yes", "no"])
price_range = st.slider("Price Range (1 to 5)", 1, 5, 3)
votes = st.number_input("Number of Votes", min_value=0, step=1)
average_cost_for_two = st.number_input("Average Cost for Two", min_value=0, step=50)
cuisine_count = st.number_input("Cuisine Count", min_value=1, step=1)

if st.button("Predict Rating"):
    rating, performance = predict_rating(
        has_table_booking,
        has_online_delivery,
        is_delivering_now,
        price_range,
        votes,
        average_cost_for_two,
        cuisine_count
    )
    st.success(f"✅ Predicted Rating: {rating:.2f}")
    st.info(f"📊 Restaurant is likely to: **{performance}**")
