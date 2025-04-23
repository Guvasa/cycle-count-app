import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import os
os.environ["OMP_NUM_THREADS"] = "1"

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("normalized_data_with_ABC_classification.csv", delimiter=',')

df = load_data()

# --- Streamlit UI ---
st.title("Cycle Count Cluster Planner")

# Select date
selected_date = st.date_input("Select Date", datetime.today())
today = datetime.combine(selected_date, datetime.min.time())

# Select Site/SubSite
site_options = df[['Sitio', 'SubSite']].drop_duplicates().sort_values(by=['Sitio', 'SubSite'])
site_selection = st.selectbox("Select Site/SubSite", site_options.apply(lambda x: f"{x['Sitio']} / {x['SubSite']}", axis=1))
selected_site, selected_subsite = site_selection.split(" / ")

# --- Preprocessing ---
df['LastCount_Date'] = pd.to_datetime(df['LastCount_Date'], dayfirst=True, errors='coerce')
df['DaysSinceLastCount'] = (today - df['LastCount_Date']).dt.days

def able_to_be_counted(row):
    if row['Classification'] == 'A':
        return row['DaysSinceLastCount'] >= 30
    elif row['Classification'] == 'B':
        return row['DaysSinceLastCount'] >= 45
    elif row['Classification'] == 'C':
        return row['DaysSinceLastCount'] >= 90
    return False

df['AbleToBeCounted'] = df.apply(able_to_be_counted, axis=1)

# Cluster and sample workload
def get_clustered_locations(site_df, full_group_df, max_locations):
    site_df = site_df.sort_values(by='Times_Counted_CurrentQtr')

    quota = {
        'A': int(np.ceil(max_locations * 0.10)),
        'B': int(np.ceil(max_locations * 0.15)),
    }
    quota['C'] = max_locations - quota['A'] - quota['B']

    selected = []
    for cls in ['A', 'B', 'C']:
        class_df = site_df[site_df['Classification'] == cls]
        class_df = class_df.sort_values(by='Times_Counted_CurrentQtr')
        selected.append(class_df.head(quota[cls]))

    selected_df = pd.concat(selected)

    seed_coords = full_group_df.sort_values(by='LastCount_Date', ascending=False)[['X', 'Y', 'Z']].head(1).values
    coords = selected_df[['X', 'Y', 'Z']]
    n_clusters = max(1, len(coords) // max_locations)
    kmeans = KMeans(n_clusters=n_clusters, init=seed_coords, n_init=1, random_state=42)
    selected_df['Cluster'] = kmeans.fit_predict(coords)

    return selected_df

# Apply clustering
site_group = df[(df['Sitio'] == selected_site) & (df['SubSite'] == selected_subsite)]
eligible_group = site_group[site_group['AbleToBeCounted']]

if eligible_group.empty:
    st.warning("No eligible locations for selected Site/SubSite on this date.")
else:
    max_locations = site_group['MaxLocations'].iloc[0]
    clustered_df = get_clustered_locations(eligible_group, site_group, max_locations)

    st.subheader("Selected Locations")
    st.dataframe(clustered_df[['Location', 'X', 'Y', 'Z', 'Classification', 'Times_Counted_CurrentQtr', 'LastCount_Date', 'Cluster']])

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        clustered_df['X'], clustered_df['Y'], clustered_df['Z'],
        c=clustered_df['Cluster'], cmap='tab10'
    )
    ax.set_title("3D Clustered Daily Workload")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    st.pyplot(fig)