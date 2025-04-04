import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate per 90 minute metrics for goals, xG, assists, and xA
def generate_per_90_features(df):
    df['rolling_goals_per_90_3'] = (df['rolling_goals_3'] / df['rolling_mins_3']) * 90
    df['rolling_goals_per_90_10'] = (df['rolling_goals_10'] / df['rolling_mins_10']) * 90
    df['season_goals_per_90'] = (df['season_avg_goals_scored'] / df['season_avg_minutes']) * 90

    df['rolling_xg_per_90_3'] = (df['rolling_xg_3'] / df['rolling_mins_3']) * 90
    df['rolling_xg_per_90_10'] = (df['rolling_xg_10'] / df['rolling_mins_10']) * 90
    df['season_xg_per_90'] = (df['season_avg_expected_goals'] / df['season_avg_minutes']) * 90

    df['rolling_assists_per_90_3'] = (df['rolling_assists_3'] / df['rolling_mins_3']) * 90
    df['rolling_assists_per_90_10'] = (df['rolling_assists_10'] / df['rolling_mins_10']) * 90
    df['season_assists_per_90'] = (df['season_avg_assists'] / df['season_avg_minutes']) * 90

    df['rolling_xa_per_90_3'] = (df['rolling_xa_3'] / df['rolling_mins_3']) * 90
    df['rolling_xa_per_90_10'] = (df['rolling_xa_10'] / df['rolling_mins_10']) * 90
    df['season_xa_per_90'] = (df['season_avg_expected_assists'] / df['season_avg_minutes']) * 90

    return df

# Generate interaction and composite features
def generate_interaction_features(df):
    df['scoring_chances_3'] = df['rolling_xg_per_90_3'] * df['rolling_finishing_efficiency_3']
    df['scoring_chances_10'] = df['rolling_xg_per_90_10'] * df['rolling_finishing_efficiency_10']

    df['assist_chances_3'] = df['rolling_xa_per_90_3'] * df['rolling_teammate_clinicalness_3']
    df['assist_chances_10'] = df['rolling_xa_per_90_10'] * df['rolling_teammate_clinicalness_10']

    df['team_goal_involvement_3'] = (df['rolling_goals_3'] + df['rolling_assists_3']) / df['rolling_team_goals_3'] * 100
    df['team_goal_involvement_10'] = (df['rolling_goals_10'] + df['rolling_assists_10']) / df['rolling_team_goals_10'] * 100

    df['opponent_goal_concession_3'] = df['opponent_rolling_goals_conceded_3'] * df['relative_strength_attack']
    df['opponent_goal_concession_10'] = df['opponent_rolling_goals_conceded_10'] * df['relative_strength_attack']

    return df

# Load and process data by position
def load_and_process_data(file_path, position):
    fpl_df = pd.read_csv(file_path)
    position_df = fpl_df[fpl_df['position'] == position]
    position_df = generate_per_90_features(position_df)
    position_df = generate_interaction_features(position_df)
    return position_df

# Feature definitions by position
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

# Model training and prediction
def train_and_predict_model(position_df, features, gw_train, gw_test):
    train_df = position_df[position_df['GW'] <= gw_train]
    test_df = position_df[position_df['GW'] == gw_test]

    # One-hot encode difficulty
    train_encoded = pd.get_dummies(train_df['difficulty'], prefix='difficulty', dtype='int8')
    test_encoded = pd.get_dummies(test_df['difficulty'], prefix='difficulty', dtype='int8')

    X_train = pd.concat([train_df[features], train_encoded], axis=1)
    X_test = pd.concat([test_df[features], test_encoded], axis=1)

    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    y_train = train_df['total_points'] - train_df['bonus']
    y_test = test_df['total_points'] - test_df['bonus']

    X_train.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    X_test.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

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
    xgb_model.fit(X_train_scaled, y_train)
    y_pred = xgb_model.predict(X_test_scaled)

    return y_test, y_pred, test_df

# Main function to get predicted players
def get_all_predicted_players(file_path,gw_train, gw_test):
    positions = ['GKP', 'DEF', 'MID', 'FWD']
    all_predictions = []

    for position in positions:
        position_df = load_and_process_data(file_path, position)

        y_test, y_pred, test_df = train_and_predict_model(
            position_df, globals()[f"features_{position.lower()}"], gw_train, gw_test
        )

        test_df['predicted_base_points'] = y_pred
        test_df['Position'] = position
        all_predictions.append(test_df)

    combined_df = pd.concat(all_predictions, axis=0)
    combined_df['predicted_base_points'] = np.maximum(combined_df['predicted_base_points'], 0)

    combined_df['rank'] = combined_df.groupby('fixture')['predicted_base_points'].rank(
        ascending=False, method='first'
    )

    combined_df['predicted_bonus'] = 0
    combined_df.loc[combined_df['rank'] == 1, 'predicted_bonus'] = 3
    combined_df.loc[combined_df['rank'] == 2, 'predicted_bonus'] = 2
    combined_df.loc[combined_df['rank'] == 3, 'predicted_bonus'] = 1

    combined_df['predicted_points'] = combined_df['predicted_base_points'] + combined_df['predicted_bonus']

    results = combined_df[[
        'player', 'Position', 'total_points', 'predicted_base_points',
        'predicted_bonus', 'predicted_points'
    ]].rename(columns={
        'player': 'Player',
        'predicted_base_points': 'Predicted Base Points',
        'predicted_bonus': 'Predicted Bonus Points',
        'predicted_points': 'Predicted Total Points'
    })

    return results
# Main execution
if __name__ == "__main__":
    file_path = "final_data.csv"
    gw_train = 30
    gw_test = 31
    all_players_df = get_all_predicted_players(file_path, gw_train, gw_test)

    print(all_players_df.sort_values('Predicted Total Points', ascending=False).head(30))

    # Save for Streamlit
    all_players_df.to_csv("weekly_predictions.csv", index=False)
    print("Predictions saved to weekly_predictions.csv")
