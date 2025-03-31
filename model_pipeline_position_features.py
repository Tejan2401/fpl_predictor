
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Features for each position
features_gkp = ['is_home',
    'relative_strength_overall', 'relative_strength_defence', 'rolling_mins_3',
    'rolling_xgc_3', 'rolling_xgc_10', 'rolling_defensive_efficiency_3',
    'rolling_defensive_efficiency_10', 'rolling_clean_sheets_3', 'rolling_clean_sheets_10',
    'rolling_saves_3', 'rolling_saves_10', 'season_avg_defensive_efficiency',
    'season_avg_clean_sheets', 'season_avg_saves', 'season_avg_goals_conceded',
    'season_avg_expected_goals_conceded'
]

features_def = [
    'relative_strength_defence', 'rolling_xgc_3', 'rolling_xgc_10', 'rolling_clean_sheets_3',
    'rolling_clean_sheets_10', 'rolling_defensive_efficiency_3', 'rolling_defensive_efficiency_10',
    'opponent_rolling_goals_conceded_3', 'opponent_rolling_goals_conceded_10', 'season_avg_clean_sheets',
    'season_avg_goals_conceded', 'season_avg_expected_goals_conceded', 'season_avg_defensive_efficiency', 'rolling_xg_per_90_3',
    'rolling_xg_per_90_10', 'season_xg_per_90', 'rolling_assists_per_90_10',
    'season_assists_per_90', 'rolling_xa_per_90_3', 'rolling_xa_per_90_10', 'season_xa_per_90', 'assist_chances_10','season_avg_team_goals_scored', 'season_avg_assists',
    'season_avg_expected_goals', 'season_avg_expected_assists','rolling_mins_3'
]

features_mid = [
    'rolling_goals_per_90_3', 'rolling_goals_per_90_10', 'season_goals_per_90',
    'rolling_xg_per_90_3', 'rolling_xg_per_90_10', 'season_xg_per_90',
    'rolling_assists_per_90_3', 'rolling_assists_per_90_10', 'season_assists_per_90',
    'rolling_xa_per_90_3', 'rolling_xa_per_90_10', 'season_xa_per_90', 'scoring_chances_10', 'assist_chances_10',
    'team_goal_involvement_3', 'team_goal_involvement_10', 'rolling_mins_3',
    'relative_strength_attack', 'rolling_team_goals_10', 'opponent_rolling_goals_conceded_10', 'season_avg_team_goals_scored', 'season_avg_goals_scored',
    'season_avg_assists', 'season_avg_expected_goals', 'season_avg_expected_assists', 'opponent_goal_concession_3',
    'opponent_goal_concession_10', 'relative_strength_overall', 'relative_strength_defence', 'rolling_clean_sheets_10'
]

features_fwd = [
    'rolling_goals_per_90_3', 'rolling_goals_per_90_10', 'season_goals_per_90',
    'rolling_xg_per_90_3', 'rolling_xg_per_90_10', 'season_xg_per_90', 'rolling_assists_per_90_10', 'season_assists_per_90',
    'rolling_xa_per_90_3', 'rolling_xa_per_90_10', 'season_xa_per_90',
    'scoring_chances_3', 'scoring_chances_10', 'assist_chances_3', 'assist_chances_10',
    'team_goal_involvement_3', 'team_goal_involvement_10', 'rolling_mins_3',
    'relative_strength_attack', 'rolling_team_goals_10',
    'opponent_rolling_goals_conceded_3', 'opponent_rolling_goals_conceded_10', 'season_avg_team_goals_scored',
    'season_avg_goals_scored', 'season_avg_assists', 'season_avg_expected_goals',
    'season_avg_expected_assists', 'opponent_goal_concession_3', 'opponent_goal_concession_10'
]

#Loading and processing data
def load_and_process_data(file_path, position):
    print(f"DEBUG: file_path = {file_path} ({type(file_path)})")
    fpl_df = pd.read_csv(file_path)
    position_df = fpl_df[fpl_df['position'] == position]
    return position_df

# Model training and predictions
def train_and_predict_model(position_df, features, gw_train, gw_test):
    train_df = position_df[position_df['GW'] <= gw_train]
    test_df = position_df[position_df['GW'] == gw_test]

    # One-hot encode the 'difficulty' column for both training and testing sets
    train_encoded = pd.get_dummies(train_df['difficulty'], prefix='difficulty', dtype='int8')
    test_encoded = pd.get_dummies(test_df['difficulty'], prefix='difficulty', dtype='int8')

    # Concatenate encoded columns with the original features
    X_train = pd.concat([train_df[features], train_encoded], axis=1)
    X_test = pd.concat([test_df[features], test_encoded], axis=1)

    # Handle any mismatches in the encoded columns between train and test
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # Prepare the target variable
    y_train = train_df['total_points']
    y_test = test_df['total_points']

    # Handle missing values by allowing XGBoost to handle NaNs natively
    X_train.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    X_test.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Scale the features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize and train the XGBoost model
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.055,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )

    # Fit the model to the training data
    xgb_model.fit(X_train_scaled, y_train)

    # Predict for the test set (GW_TEST))
    y_pred = xgb_model.predict(X_test_scaled)
    return y_test, y_pred, test_df

# Main function to handle prediction for all positions and return the players
def get_all_predicted_players(file_path,gw_train, gw_test):
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    all_predictions = []

    # Iterate over each position to train and predict
    for position in positions:
        position_df = load_and_process_data(file_path, position)

        # Train the model for the given position
        y_test, y_pred, test_df = train_and_predict_model(position_df, globals()[f"features_{position.lower()}"], gw_train, gw_test)

        # Prepare results for each position
        results = pd.DataFrame({"Player": test_df['player'], "Actual Points": y_test, "Predicted Points": y_pred})

        # Add the position column to the results
        results['Position'] = position

        # Append the results to the all_predictions list
        all_predictions.append(results)

    # Concatenate all predictions into one DataFrame
    all_players_df = pd.concat(all_predictions, axis=0)

    # Replace negative predicted points with 0
    all_players_df['Predicted Points'] = np.maximum(all_players_df['Predicted Points'], 0)

    # Return the DataFrame with all players (no sorting or filtering)
    return all_players_df
# Main execution
if __name__ == "__main__":
    file_path = "final_data.csv"
    gw_train = 29
    gw_test = 30
    all_players_df = get_all_predicted_players(file_path, gw_train, gw_test)

    print(all_players_df.sort_values('Predicted Points', ascending=False).head(30))

    # Save for Streamlit
    all_players_df.to_csv("weekly_predictions.csv", index=False)
    print("Predictions saved to weekly_predictions.csv")
