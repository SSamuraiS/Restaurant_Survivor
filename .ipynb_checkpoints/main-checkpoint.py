import pandas as pd
import joblib

# Load your pre-trained model
model_filename = "restaurant_rating_model.pkl"
model = joblib.load(model_filename)
# print("Model loaded from", model_filename)

def predict(has_table_booking, has_online_delivery, is_delivering_now, price_range, votes, average_cost_for_two, cuisine_count):
    """
    Predicts the restaurant rating based on input parameters and returns whether the restaurant will perform or not.
    
    Parameters:
        has_table_booking (str): "Yes" or "No"
        has_online_delivery (str): "Yes" or "No"
        is_delivering_now (str): "Yes" or "No"
        price_range (int or float): Price range of the restaurant (numeric)
        votes (int): Number of votes received
        average_cost_for_two (int or float): Average cost for two
        cuisine_count (int): Number of cuisines (extracted from a 'Cuisines' column)
        
    Returns:
        tuple: (predicted_rating, performance)
               predicted_rating: The numeric rating predicted by the model.
               performance: "Perform" if rating >= threshold, otherwise "Not Perform"
    """
    
    # Convert Yes/No inputs to 1/0
    table_booking = 1 if has_table_booking.lower() == "yes" else 0
    online_delivery = 1 if has_online_delivery.lower() == "yes" else 0
    delivering_now = 1 if is_delivering_now.lower() == "yes" else 0
    
    # Construct a DataFrame for a single sample with the expected feature names
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
    
    # Use the loaded model to predict rating
    predicted_rating = model.predict(df)[0]
    
    # Define a threshold rating for good performance
    threshold = 3.5
    performance = "Perform" if predicted_rating >= threshold else "Not Perform"
    
    return predicted_rating, performance

# ---------------------------
# Test Case
# ---------------------------
if __name__ == '__main__':
    # Sample input parameters:
    sample_params = {
        "has_table_booking": "Yes",
        "has_online_delivery": "No",
        "is_delivering_now": "yes",
        "price_range": 2,
        "votes": 10,
        "average_cost_for_two": 100,
        "cuisine_count": 5
    }
    
    rating, result = predict(**sample_params)
    print("Predicted Rating:", rating)
    print("Restaurant Performance:", result)
