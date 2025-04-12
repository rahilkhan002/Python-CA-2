import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Load dataset
file_path = r"C:\Users\rahil\OneDrive\Documents\Exel CA 2\Border_Crossing_Entry_Data[1].csv"
df = pd.read_csv(file_path, low_memory=False)

# --- EDA: Exploratory Data Analysis ---
print("=== Dataset Overview ===")
print(f"Shape of dataset: {df.shape}")
print("\nColumns:\n", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

print("\n=== Data Types and Missing Values ===")
print(df.info())
print("\nMissing values per column:\n", df.isnull().sum())

print("\n=== Unique Values per Column ===")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

print("\n=== Descriptive Statistics ===")
print(df.describe(include='all'))

# Optional: Histogram of Value column
plt.figure(figsize=(8, 5))
sns.histplot(df['Value'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Crossing Values")
plt.xlabel("Crossings")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Data Preprocessing ---
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'], format='%b-%y', errors='coerce')
df = df.dropna(subset=['Date']) 
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.strftime('%B')

# Month order for plots
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']

# --- 1. Monthly and Seasonal Trends: Line Plot ---
recent_years = sorted(df['Year'].unique())[-5:]
monthly_line_df = df[df['Year'].isin(recent_years)].groupby(['Year', 'Month'])['Value'].sum().reset_index()

plt.figure(figsize=(14, 6))
for year in recent_years:
    year_data = monthly_line_df[monthly_line_df['Year'] == year]
    plt.plot(year_data['Month'], year_data['Value'], marker='o', label=year)

plt.xticks(ticks=range(1, 13), labels=month_order, rotation=45)
plt.title("Monthly Border Crossing Trends by Year (Line Plot)")
plt.xlabel("Month")
plt.ylabel("Total Crossings")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Optional: Grouped Bar Chart by Month-Year ---
monthly_bar_df = df[df['Year'].isin(recent_years)].groupby(['Month_Name', 'Year'])['Value'].sum().reset_index()
monthly_bar_df['Month_Name'] = pd.Categorical(monthly_bar_df['Month_Name'], categories=month_order, ordered=True)
monthly_bar_df = monthly_bar_df.sort_values(['Month_Name', 'Year'])

plt.figure(figsize=(16, 6))
sns.barplot(data=monthly_bar_df, x='Month_Name', y='Value', hue='Year', palette='Set2')
plt.title("Monthly Border Crossing Comparison Across Years (Bar Chart)")
plt.xlabel("Month")
plt.ylabel("Total Crossings")
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# --- 2. Border-Wise Comparison ---
border_traffic = df.groupby('Border')['Value'].sum()
print("Total Traffic by Border:\n", border_traffic)
plt.figure(figsize=(8, 5))
sns.barplot(x=border_traffic.index, y=border_traffic.values, palette="viridis")
plt.title("Traffic Comparison: US-Canada vs. US-Mexico Borders")
plt.ylabel("Total Crossings")
plt.tight_layout()
plt.show()

# --- 3. Top 10 States ---
state_traffic = df.groupby('State')['Value'].sum().sort_values(ascending=False)
print("Top 10 States by Border Traffic:\n", state_traffic.head(10))
plt.figure(figsize=(12, 6))
sns.barplot(y=state_traffic.head(10).index, x=state_traffic.head(10).values, palette="magma")
plt.title("Top 10 States by Border Traffic")
plt.xlabel("Total Crossings")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# --- 4. Top & Least Busy Ports ---
port_traffic = df.groupby('Port Name')['Value'].sum().sort_values(ascending=False)
print("Top 5 Busiest Ports:\n", port_traffic.head(5))
plt.figure(figsize=(12, 6))
sns.barplot(x=port_traffic.head(5).index, y=port_traffic.head(5).values, palette="coolwarm")
plt.xticks(rotation=45)
plt.title("Top 5 Busiest Ports")
plt.ylabel("Total Crossings")
plt.tight_layout()
plt.show()

print("Top 5 Least Busy Ports:\n", port_traffic.tail(5))
plt.figure(figsize=(12, 6))
sns.barplot(x=port_traffic.tail(5).index, y=port_traffic.tail(5).values, palette="Blues")
plt.xticks(rotation=45)
plt.title("Least Busy Ports")
plt.ylabel("Total Crossings")
plt.tight_layout()
plt.show()

# --- 5. Donut Chart: Transport Mode ---
transport_mode = df.groupby('Measure')['Value'].sum().sort_values(ascending=False)
print("Traffic by Mode of Transport:\n", transport_mode)

plt.figure(figsize=(8, 8))
wedges, texts, autotexts = plt.pie(transport_mode.values, labels=transport_mode.index,
                                   autopct='%1.1f%%', startangle=140, colors=sns.color_palette("crest"))
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title("Border Crossings by Mode of Transport (Donut Chart)")
plt.tight_layout()
plt.show()

# --- 6. Geospatial Visualization ---
world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")

if 'Longitude' in df.columns and 'Latitude' in df.columns:
    df = df.dropna(subset=['Longitude', 'Latitude'])
    df['geometry'] = df.apply(lambda row: Point(row['Longitude'], row['Latitude']), axis=1)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')

    print("Border Crossing Geospatial Data (Sample):\n", gdf.head())

    fig, ax = plt.subplots(figsize=(10, 6))
    world.boundary.plot(ax=ax, color='black')
    gdf.plot(ax=ax, markersize=10, alpha=0.5, color='red')
    plt.title("Border Crossing Locations")
    plt.tight_layout()
    plt.show()
else:
    print("Longitude and Latitude columns missing! Skipping geospatial visualization.")

# --- 7. Yearly Growth ---
yearly_growth = df.groupby('Year')['Value'].sum()
print("Yearly Growth Data:\n", yearly_growth)
plt.figure(figsize=(10, 5))
sns.lineplot(x=yearly_growth.index, y=yearly_growth.values, marker='o', color='brown')
plt.title("Yearly Border Traffic Growth")
plt.ylabel("Total Crossings")
plt.xlabel("Year")
plt.grid()
plt.tight_layout()
plt.show()
