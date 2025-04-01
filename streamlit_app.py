import streamlit as st
import pandas as pd
import os

CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor")

# Check if prediction file exists
if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found. Please run the pipeline (main.py) first.")
else:
    @st.cache_data
    def load_predictions():
        return pd.read_csv(CACHE_FILE)

    df = load_predictions()

    st.subheader("Predicted Points - All Players")

    # 🔍 Search bar for player name (full table)
    name_search = st.text_input("Search for a player", "").lower()

    if name_search:
        filtered_df_full = df[df["Player"].str.lower().str.contains(name_search)]
    else:
        filtered_df_full = df

    # Display full (or filtered) table
    st.dataframe(filtered_df_full.sort_values("Predicted Points", ascending=False))

    # Position filter
    st.subheader("🔎 Filter by Position")
    positions = df["Position"].unique()
    selected_position = st.selectbox("Choose Position", sorted(positions))
    filtered_df_position = df[df["Position"] == selected_position]

    st.dataframe(filtered_df_position.sort_values("Predicted Points", ascending=False))

    # ================================
    # TEAM BUILDER SECTION
    # ================================
    st.subheader("🧩 Build Your Fantasy Team")

    MAX_PLAYERS = 11
    POSITION_LIMITS = {
        "GKP": 1,
        "DEF": 5,
        "MID": 5,
        "FWD": 3
    }

    selected_players = st.multiselect(
        "Select up to 11 players (follow FPL rules)",
        options=df["Player"].tolist(),
        default=[],
    )

    team_df = df[df["Player"].isin(selected_players)]

    # Count players by position
    position_counts = team_df["Position"].value_counts().to_dict()

    # Validation
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

        # Let user pick a captain
        st.markdown("### 🎯 Choose your Captain")
        captain = st.selectbox("Pick one of your 11 players", options=selected_players)

        # Add captain column
        team_df["Captain?"] = team_df["Player"].apply(lambda x: "⭐" if x == captain else "")

        # Adjust points
        team_df["Adjusted Points"] = team_df.apply(
            lambda row: row["Predicted Points"] * 2 if row["Player"] == captain else row["Predicted Points"],
            axis=1
        )

        # Display team
        st.dataframe(team_df[["Captain?", "Player", "Position", "Predicted Points", "Adjusted Points"]])

        total = team_df["Adjusted Points"].sum()
        st.markdown(f"### 🏆 Total Predicted Points (with captain boost): **{total:.1f}**")

    else:
        st.info("Select 11 players to see your predicted score.")
