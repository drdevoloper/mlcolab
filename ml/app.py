from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Load and preprocess the IPL dataset
df = pd.read_csv('ipl.csv')

# Data Preprocessing: Drop columns that are not needed
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

# Keep only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Remove the first 5 overs as predictions are more stable after 5 overs
df = df[df['overs'] >= 5.0]

# Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# One-Hot Encoding for categorical features
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])

# Rearrange columns for consistency
encoded_df = encoded_df[['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 
                         'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders', 
                         'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
                         'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
                         'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 
                         'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders', 
                         'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
                         'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
                         'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Split data into training and testing sets
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Remove 'date' column for training
X_train.drop(labels='date', axis=1, inplace=True)
X_test.drop(labels='date', axis=1, inplace=True)

# Initialize and fit the scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the Ridge regression model
ridge_regressor = Ridge(alpha=1.0)
ridge_regressor.fit(X_train_scaled, y_train)

# Home route to render HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the predict route for the front-end
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # List of teams for encoding
    teams = ['Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab', 
             'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals', 
             'Royal Challengers Bangalore', 'Sunrisers Hyderabad']

    # Prepare the input data for the model
    input_data = {
        'bat_team': data['batting_team'],
        'bowl_team': data['bowling_team'],
        'overs': data['overs'],
        'runs': data['runs'],
        'wickets': data['wickets'],
        'runs_last_5': data['runs_in_prev_5'],
        'wickets_last_5': data['wickets_in_prev_5']
    }

    # Create DataFrame for encoding
    input_df = pd.DataFrame([input_data])

    # One-Hot Encoding
    input_df = pd.get_dummies(data=input_df, columns=['bat_team', 'bowl_team'], drop_first=True)

    # Ensure all columns are present for the model
    for team in teams:
        if 'bat_team_' + team not in input_df.columns:
            input_df['bat_team_' + team] = 0
        if 'bowl_team_' + team not in input_df.columns:
            input_df['bowl_team_' + team] = 0

    # Reorder columns to match training data
    input_df = input_df.reindex(columns=X_train.columns, fill_value=0)

    # Scale the input data for predictions
    input_df_scaled = scaler.transform(input_df)

    # Predict the score
    predicted_score = round(ridge_regressor.predict(input_df_scaled)[0])
    
    return jsonify(predicted_score=predicted_score)

# Define route for prediction history (graph)
@app.route('/history', methods=['POST'])
def history():
    data = request.get_json()
    past_predictions = data.get('past_predictions', [])

    # Returning past predictions for the frontend
    return jsonify({'predictions': past_predictions})

if __name__ == '__main__':
    app.run(debug=True)
