import streamlit as st
import pandas as pd
import os

CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor")

# Load prediction data
if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found. Please run the pipeline (main.py) first.")
else:
    @st.cache_data
    def load_predictions():
        return pd.read_csv(CACHE_FILE)

    df = load_predictions()

    st.subheader("🔍 Search and Build Your Fantasy Team")

    # Search bar
    name_search = st.text_input("Search for a player", "").lower()

    if name_search:
        df = df[df["Player"].str.lower().str.contains(name_search)]

    df = df.sort_values("Predicted Points", ascending=False)

    # Create a checkbox for each player row (up to 11)
    st.write("✅ Select up to 11 players:")

    selected_players = []
    for i, row in df.iterrows():
        label = f"{row['Player']} ({row['Position']}) - {row['Predicted Points']:.1f} pts"
        if st.checkbox(label, key=row['Player']):
            selected_players.append(row['Player'])

    team_df = df[df["Player"].isin(selected_players)]

    # FPL rules
    MAX_PLAYERS = 11
    POSITION_LIMITS = {"GKP": 1, "DEF": 5, "MID": 5, "FWD": 3}
    position_counts = team_df["Position"].value_counts().to_dict()

    error_msgs = []
    if len(selected_players) > MAX_PLAYERS:
        error_msgs.append("You can only select 11 players total.")
    for pos, limit in POSITION_LIMITS.items():
        if position_counts.get(pos, 0) > limit:
            error_msgs.append(f"Too many {pos}s selected (max {limit}).")

    if error_msgs:
        st.error("⚠️ Invalid team:\n" + "\n".join(error_msgs))
    elif len(selected_players) == 11:
        st.success("✅ Valid team!")

        # Captain selection
        captain = st.selectbox("🎯 Pick your Captain", selected_players)
        team_df["Captain?"] = team_df["Player"].apply(lambda x: "⭐" if x == captain else "")
        team_df["Adjusted Points"] = team_df.apply(
            lambda row: row["Predicted Points"] * 2 if row["Player"] == captain else row["Predicted Points"],
            axis=1
        )

        st.dataframe(team_df[["Captain?", "Player", "Position", "Predicted Points", "Adjusted Points"]])
        total = team_df["Adjusted Points"].sum()
        st.markdown(f"### 🏆 Total Predicted Points (with captain boost): **{total:.1f}**")
    else:
        st.info("Select 11 players to see your predicted score.")
