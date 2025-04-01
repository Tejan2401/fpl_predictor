import streamlit as st
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
        options=df["Player"],
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

        # Highlight selected team
        team_df["Captain?"] = team_df["Player"].apply(lambda x: "⭐" if x == captain else "")

        # Adjust predicted points
        team_df["Adjusted Points"] = team_df.apply(
            lambda row: row["Predicted Points"] * 2 if row["Player"] == captain else row["Predicted Points"],
            axis=1
        )

        # Display selected team
        st.dataframe(team_df[["Captain?", "Player", "Position", "Predicted Points", "Adjusted Points"]])

        # Total with captain boost
        total = team_df["Adjusted Points"].sum()
        st.markdown(f"### 🏆 Total Predicted Points (with captain boost): **{total:.1f}**")
    else:
        st.info("Select 11 players to see your predicted score.")
