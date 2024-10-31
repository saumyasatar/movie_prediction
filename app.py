import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('model.joblib')

# Define function to collect user input and ensure it matches the model’s expected input
def user_input_features():
    # Collect each feature needed for prediction
    color = st.selectbox("Color", ['Color', 'Black and White'])
    num_critic_for_reviews = st.number_input("Number of Critic Reviews", min_value=0, step=1)
    duration = st.number_input("Duration (minutes)", min_value=0, step=1)
    director_facebook_likes = st.number_input("Director Facebook Likes", min_value=0, step=1)
    actor_1_facebook_likes = st.number_input("Actor 1 Facebook Likes", min_value=0, step=1)
    actor_2_facebook_likes = st.number_input("Actor 2 Facebook Likes", min_value=0, step=1)
    actor_3_facebook_likes = st.number_input("Actor 3 Facebook Likes", min_value=0, step=1)
    gross = st.number_input("Gross Revenue", min_value=0.0, step=100000.0)
    genres = st.selectbox("Genre", ['Action', 'Drama', 'Comedy', 'Horror', 'Romance'])
    country = st.selectbox("Country", ['USA', 'UK', 'France', 'Canada'])
    content_rating = st.selectbox("Content Rating", ['G', 'PG', 'PG-13', 'R', 'Unrated'])
    budget = st.number_input("Budget", min_value=0.0, step=100000.0)
    title_year = st.number_input("Title Year", min_value=1900, max_value=2024, step=1)
    imdb_score = st.slider("IMDB Score", 0.0, 10.0, 5.0)

    # Organize the inputs into a DataFrame (drop non-numeric columns)
    data = {
        'color': color,
        'num_critic_for_reviews': num_critic_for_reviews,
        'duration': duration,
        'director_facebook_likes': director_facebook_likes,
        'actor_1_facebook_likes': actor_1_facebook_likes,
        'actor_2_facebook_likes': actor_2_facebook_likes,
        'actor_3_facebook_likes': actor_3_facebook_likes,
        'gross': gross,
        'genres': genres,
        'country': country,
        'content_rating': content_rating,
        'budget': budget,
        'title_year': title_year,
        'imdb_score': imdb_score,
    }
    features = pd.DataFrame(data, index=[0])

    # Apply the same encoding and transformations as used in model training
    features['color'] = features['color'].map({'Color': 1, 'Black and White': 0})
    features = pd.get_dummies(features, columns=['genres', 'country', 'content_rating'], drop_first=True)
    
    return features

# Streamlit app layout
st.title("Movie Classification Prediction")
st.write("This app predicts whether a movie is classified as a 'Hit,' 'Average,' or 'Flop' based on input features.")

# Collect user input
input_df = user_input_features()

# Display input for confirmation
st.write("### User Input Features")
st.write(input_df)

# Ensure input columns match those expected by the model
model_columns = model.feature_names_in_
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Make predictions when the user clicks the Predict button
if st.button('Predict'):
    # Apply the model to make predictions
    prediction = model.predict(input_df)

    # Display the prediction result
    st.write("### Prediction Result")
    st.write(f"The movie is predicted to be: **{prediction[0]}**")
