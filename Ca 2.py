import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Load dataset
file_path = r"C:\Users\rahil\OneDrive\Documents\Exel CA 2\Border_Crossing_Entry_Data[1].csv"
df = pd.read_csv(file_path, low_memory=False)

# Data Preprocessing
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y', errors='coerce')
df = df.dropna(subset=['Date']) 
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')

# 1. Monthly and Seasonal Trends
monthly_trend = df.groupby(['Year', 'Month_Name'])['Value'].sum().unstack()
print("Monthly Trend Data:\n", monthly_trend)
plt.figure(figsize=(12, 6))
sns.heatmap(monthly_trend, cmap='coolwarm', annot=True, fmt='.0f')
plt.title("Monthly Border Crossing Trends")
plt.ylabel("Year")
plt.xlabel("Month")
plt.show()

# 2. Border and State-Wise Traffic Comparison
border_traffic = df.groupby('Border')['Value'].sum()
print("Total Traffic by Border:\n", border_traffic)
plt.figure(figsize=(8, 5))
sns.barplot(x=border_traffic.index, y=border_traffic.values, palette="viridis")
plt.title("Traffic Comparison: US-Canada vs. US-Mexico Borders")
plt.ylabel("Total Crossings")
plt.show()

# Changed: Horizontal Bar Chart for Top 10 States
state_traffic = df.groupby('State')['Value'].sum().sort_values(ascending=False)
print("Top 10 States by Border Traffic:\n", state_traffic.head(10))
plt.figure(figsize=(12, 6))
sns.barplot(y=state_traffic.head(10).index, x=state_traffic.head(10).values, palette="magma")
plt.title("Top 10 States by Border Traffic")
plt.xlabel("Total Crossings")
plt.ylabel("State")
plt.show()

# 3. Top 5 Busiest and Least Busy Ports
port_traffic = df.groupby('Port Name')['Value'].sum().sort_values(ascending=False)
print("Top 5 Busiest Ports:\n", port_traffic.head(5))
plt.figure(figsize=(12, 6))
sns.barplot(x=port_traffic.head(5).index, y=port_traffic.head(5).values, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Top 5 Busiest Ports")
plt.ylabel("Total Crossings")
plt.show()

print("Top 5 Least Busy Ports:\n", port_traffic.tail(5))
plt.figure(figsize=(12, 6))
sns.barplot(x=port_traffic.tail(5).index, y=port_traffic.tail(5).values, palette="Blues")
plt.xticks(rotation=45)
plt.title("Least Busy Ports")
plt.ylabel("Total Crossings")
plt.show()

# Changed: Donut Chart for Mode of Transport Analysis
transport_mode = df.groupby('Measure')['Value'].sum().sort_values(ascending=False)
print("Traffic by Mode of Transport:\n", transport_mode)

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(transport_mode.values, labels=transport_mode.index,
                                   autopct='%1.1f%%', startangle=140, colors=sns.color_palette("crest"))
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Border Crossings by Mode of Transport (Donut Chart)")
plt.show()

# 5. Geospatial Visualization
world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

if 'Longitude' in df.columns and 'Latitude' in df.columns:
    df = df.dropna(subset=['Longitude', 'Latitude'])  # Drop missing values
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    print("Border Crossing Geospatial Data (Sample):\n", gdf.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    world.boundary.plot(ax=ax, color='black')
    gdf.plot(ax=ax, markersize=10, alpha=0.5, color='red')
    plt.title("Border Crossing Locations")
    plt.show()
else:
    print("Longitude and Latitude columns missing! Skipping geospatial visualization.")

# 6. Year-over-Year Growth Analysis
yearly_growth = df.groupby('Year')['Value'].sum()
print("Yearly Growth Data:\n", yearly_growth)
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_growth.index, y=yearly_growth.values, marker='o', color='brown')
plt.title("Yearly Border Traffic Growth")
plt.ylabel("Total Crossings")
plt.xlabel("Year")
plt.grid()
plt.show()
