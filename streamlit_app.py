import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")
CACHE_FILE = "weekly_predictions.csv"

st.title("⚽ FPL Points Predictor - GW Selector")

if not os.path.exists(CACHE_FILE):
    st.warning("Predictions not found")
else:
    @st.cache_data
    def load_predictions():
        df = pd.read_csv(CACHE_FILE)

        # Get latest Gameweek
        latest_gw = df["GW"].max()
        df = df[df["GW"] == latest_gw]

        # Group by player, keeping summed points + combined Opponent and H/A info
        df_grouped = df.groupby(["Player", "Position"], as_index=False).agg({
            "Predicted Total Points": "sum",
            "Opponent": lambda x: ", ".join(x.astype(str)),
            "H/A": lambda x: ", ".join(x.astype(str))
        })

        # Clean player name for merging
        df_grouped["Player_clean"] = df_grouped["Player"].str.strip().str.lower()
        df_grouped = df_grouped.drop_duplicates(subset=["Player_clean"])

        return df_grouped, latest_gw

    df, latest_gw = load_predictions()

    st.subheader(f"Predicted Points for Gameweek {latest_gw}")

    # 🔍 Search
    name_search = st.text_input("Search for a player", "").lower()
    filtered_df = df.copy()
    if name_search:
        filtered_df = df[df["Player"].str.lower().str.contains(name_search)]

    filtered_df = filtered_df.sort_values("Predicted Total Points", ascending=False)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### 📋 All Players")
        st.dataframe(filtered_df[["Player", "Position", "Opponent", "H/A", "Predicted Total Points"]])

    with col2:
        st.markdown("### Select Your Starting 11")
        selected_players = st.multiselect(
            "Choose up to 11 players",
            options=filtered_df["Player"].tolist(),
            max_selections=11,
        )

        team_df = df[df["Player"].isin(selected_players)]

        # Count positions
        nb_GK = len(team_df[team_df["Position"] == "GKP"])
        nb_DEF = len(team_df[team_df["Position"] == "DEF"])
        nb_MID = len(team_df[team_df["Position"] == "MID"])
        nb_FWD = len(team_df[team_df["Position"] == "FWD"])

        if selected_players:
            st.markdown(f"Players selected: **{len(selected_players)}** / 11")
            st.markdown(f"GK selected: **{nb_GK}** / 1")
            st.markdown(f"DEF selected: **{nb_DEF}** / min 3")
            st.markdown(f"MID selected: **{nb_MID}** / min 2")
            st.markdown(f"FWD selected: **{nb_FWD}** / min 1")

        if len(selected_players) == 11:
            # Editable team with captain checkbox
            display_df = team_df[["Player", "Position", "Opponent", "H/A", "Predicted Total Points"]].copy()
            display_df["Captain"] = False

            edited_team = st.data_editor(
                display_df,
                column_config={
                    "Captain": st.column_config.CheckboxColumn(
                        label="Captain",
                        help="Select one player to be captain"
                    )
                },
                use_container_width=True,
                num_rows="fixed"
            )

            # Normalize names for merge
            edited_team["Player_clean"] = edited_team["Player"].str.strip().str.lower()

            # Merge with full data to get position/opponent/H/A
            merged_df = df[["Player_clean", "Position", "Opponent", "H/A"]].rename(
                columns={
                    "Position": "Merged_Position",
                    "Opponent": "Merged_Opponent",
                    "H/A": "Merged_HA"
                }
            )

            edited_team_full = edited_team.merge(merged_df, on="Player_clean", how="left")

            # Assign merged fields back
            edited_team_full["Position"] = edited_team_full["Merged_Position"]
            edited_team_full["Opponent"] = edited_team_full["Merged_Opponent"]
            edited_team_full["H/A"] = edited_team_full["Merged_HA"]

            if "Position" not in edited_team_full.columns or edited_team_full["Position"].isna().any():
                st.error("⚠️ Could not determine player positions after merging.")
                st.write("⚠️ Debug info:", edited_team_full)
            else:
                captains_selected = edited_team_full[edited_team_full["Captain"] == True]

                if len(captains_selected) == 1:
                    captain_name = captains_selected.iloc[0]["Player"]

                    edited_team_full["Adjusted Points"] = edited_team_full.apply(
                        lambda row: row["Predicted Total Points"] * 2 if row["Captain"] else row["Predicted Total Points"],
                        axis=1
                    )

                    st.success("✅ Captain selected!")
                    st.markdown(f"### 🏆 Total Predicted Points: **{edited_team_full['Adjusted Points'].sum():.1f}**")

                    # Render players by position
                    def render_row(players):
                        cols = st.columns(len(players))
                        for col, (_, player) in zip(cols, players.iterrows()):
                            is_captain = "⭐" if player["Player"] == captain_name else ""
                            col.markdown(f"{is_captain} {player['Player']}")
                            col.markdown(f"{player['Predicted Total Points']:.1f} pts")

                    gk = edited_team_full[edited_team_full["Position"] == "GKP"]
                    defs = edited_team_full[edited_team_full["Position"] == "DEF"]
                    mids = edited_team_full[edited_team_full["Position"] == "MID"]
                    fwds = edited_team_full[edited_team_full["Position"] == "FWD"]

                    st.markdown("---")
                    st.markdown("🧤 Goalkeeper")
                    render_row(gk)

                    st.markdown("🛡️ Defenders")
                    render_row(defs)

                    st.markdown("🎨 Midfielders")
                    render_row(mids)

                    st.markdown("🎯 Forwards")
                    render_row(fwds)

                elif len(captains_selected) > 1:
                    st.error("🚫 You can only select **one** captain.")
                else:
                    st.info("☑️ Select a captain to see total points.")
        elif len(selected_players) > 11:
            st.error("🚫 You can only select **11** players.")
