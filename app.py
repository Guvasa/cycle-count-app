#------------- Daily Workload App for Cycle Count ---------------
#------------ Sandro Guzmán Vargas --------------
#---------------- 2025-04-25 --------------------
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv("normalized_data_with_ABC_classification.csv", delimiter=',', encoding='latin1')

# Static dictionary of MaxLocations per SubSite
default_max_locations = {
    "CALIDAD Location": 1,
    "CHIHUAHUA": 1,
    "Distribution Center (DC)": 33,
    "Ingeniería de Servicio MTY": 1,
    "Integración 2.0": 1,
    "MTY Qality Assurance": 1,
    "MTY. BRANCH": 13,
    "MTY. BRANCH - RM": 3,
    "Qro. branch": 3,
    "Raw Material": 30,
    "Ticket Express": 4,
    "TIJ Quality Assurance": 1,
    "Tijuana branch": 4,
    "URUAPAN - APEAM": 1
}

# UI: Date input with default to today
selected_date = st.date_input("Select today's date", value=datetime.today())
today = datetime.combine(selected_date, datetime.min.time())

# UI layout: subsite selection and max location inputs side by side
subsite_options = df['SubSite'].unique()
col1, col2 = st.columns([2, 1])

with col1:
    selected_subsites = st.multiselect("Select SubSite(s)", subsite_options, default=['Distribution Center (DC)'])

with col2:
    subsite_max_location_input = {}
    for subsite in selected_subsites:
        default_value = default_max_locations.get(subsite, 5)
        subsite_max_location_input[subsite] = st.number_input(
            f"Max for {subsite[:15] + ('...' if len(subsite) > 15 else '')}",
            min_value=1,
            max_value=100,
            value=int(default_value),
            step=1,
            key=f"max_input_{subsite}"
        )

# UI: Optional seed location selector
location_options = df['Location'].unique()
selected_location = st.selectbox("Optional: Select a specific location to use as seed", options=["" ] + list(location_options))

# UI: Flexible quota sliders
st.sidebar.header("Adjust ABC Quotas (must total 100%)")
a_pct = st.sidebar.slider("% A", 0, 100, 10)
b_pct = st.sidebar.slider("% B", 0, 100 - a_pct, 15)
c_pct = 100 - a_pct - b_pct
st.sidebar.markdown(f"**% C:** {c_pct}%")

# Convert date and calculate days since last count
#df['LastCount_Date'] = pd.to_datetime(df['LastCount_Date'], dayfirst=True, errors='coerce')
df['LastCount_Date'] = pd.to_datetime(df['LastCount_Date'].astype(str).str.strip(), dayfirst=True, errors='coerce')
df['DaysSinceLastCount'] = (today - df['LastCount_Date']).dt.days

# Enhanced classification rules with frequency limits
def able_to_be_counted(row):
    if (row['Classification'] == 'A') and (row['Times_Counted_CurrentQtr'] < 3):
        return row['DaysSinceLastCount'] >= 26
    elif (row['Classification'] == 'B') and (row['Times_Counted_CurrentQtr'] < 2):
        return row['DaysSinceLastCount'] >= 40
    elif (row['Classification'] == 'C') and (row['Times_Counted_CurrentQtr'] < 1):
        return row['DaysSinceLastCount'] >= 50
    return False

df['AbleToBeCounted'] = df.apply(able_to_be_counted, axis=1)

# Run button to control execution
if st.button("Run"):
    filtered_df = df[df['SubSite'].isin(selected_subsites)]
    eligible_df = filtered_df[filtered_df['AbleToBeCounted']].copy()

    def get_clustered_locations(site_df, full_group_df, subsite, max_locations):
        site_df = site_df.sort_values(by=['Times_Counted_CurrentQtr', 'Z'])

        quota = {
            'A': int(np.ceil(max_locations * (a_pct / 100))),
            'B': int(np.ceil(max_locations * (b_pct / 100)))
        }
        quota['C'] = max_locations - quota['A'] - quota['B']

        selected = []
        for cls in ['A', 'B', 'C']:
            class_df = site_df[site_df['Classification'] == cls].sort_values(by=['Times_Counted_CurrentQtr', 'Z', 'X', 'Y'])
            selected.append(class_df.head(quota[cls]))
        selected_df = pd.concat(selected)

        # Use selected location if provided, otherwise most recent
        if selected_location and selected_location in full_group_df['Location'].values:
            seed_row = full_group_df[full_group_df['Location'] == selected_location].iloc[0]
        else:
            seed_row = full_group_df.sort_values(by='LastCount_Date', ascending=False).iloc[0]

        seed_coords = [[seed_row['X'], seed_row['Y'], seed_row['Z']]]
        st.write(f"📍 Chosen seed for {seed_row['Location']}: {[int(seed_row['X']), int(seed_row['Y']), int(seed_row['Z'])]}")

        coords = selected_df[['X', 'Y', 'Z']]
        n_clusters = max(1, len(coords) // max_locations)
        kmeans = KMeans(n_clusters=n_clusters, init=seed_coords, n_init=1, random_state=42)
        selected_df['Cluster'] = kmeans.fit_predict(coords)

        return selected_df

    daily_workload = []
    for subsite, site_group in filtered_df.groupby('SubSite'):
        eligible_group = eligible_df[eligible_df['SubSite'] == subsite]
        if eligible_group.empty:
            continue
        max_locations = subsite_max_location_input.get(subsite, 5)
        clustered_df = get_clustered_locations(eligible_group, site_group, subsite, max_locations)
        daily_workload.append(clustered_df)

    if daily_workload:
        final_selection = pd.concat(daily_workload)
        st.subheader("Selected Locations")
        st.dataframe(final_selection[['Sitio', 'SubSite', 'Location', 'X', 'Y', 'Z', 'Classification', 'Times_Counted_CurrentQtr', 'LastCount_Date', 'Cluster']])

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            final_selection['Z'], final_selection['X'], final_selection['Y'],
            c=final_selection['Cluster'], cmap='tab10')
        ax.set_title("3D Clustered Daily Workload")
        ax.set_xlabel("Z (Aisle)")
        ax.set_ylabel("X (Module)")
        ax.set_zlabel("Y (Vertical Height)")

        for _, row in final_selection.iterrows():
            ax.text(row['Z'], row['X'], row['Y'], str(row['Location']), fontsize=8)

        st.pyplot(fig)

        # Generate export-ready CSV
        export_df = final_selection.copy()
        export_df.insert(0, 'Index', range(1, len(export_df) + 1))
        export_df.insert(3, 'Comment', '')
        export_df.insert(5, 'Comment2', '')
        export_columns = ['Index', 'Location', 'LastCount_Date', 'Comment', 'SubSite', 'Comment2', 'Sitio', 'X', 'Y', 'Z', 'Classification', 'Times_Counted_CurrentQtr', 'Cluster']
        export_df = export_df[export_columns]

        st.download_button("📥 Download Daily Workload CSV", export_df.to_csv(index=False), file_name="daily_workload.csv", mime="text/csv")

        # Summary of final selection
        st.subheader("📊 Workload Summary")

        total_count = len(final_selection)
        class_counts = final_selection['Classification'].value_counts().to_dict()
        cluster_counts = final_selection['Cluster'].value_counts().sort_index().to_dict()

        st.markdown(f"- **Total Locations Selected:** {total_count}")
        st.markdown("- **By Classification:**")
        for cls in ['A', 'B', 'C']:
            count = class_counts.get(cls, 0)
            st.markdown(f"  - {cls}: {count} ({(count/total_count)*100:.1f}%)")

        st.markdown("- **By Cluster:**")
        for cluster_id, count in cluster_counts.items():
            st.markdown(f"  - Cluster {cluster_id}: {count} locations")
    else:
        st.warning("No eligible locations for selected SubSite(s) on this date.")
