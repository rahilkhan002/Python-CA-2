import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# Load dataset
file_path = r"C:\Users\rahil\OneDrive\Documents\Exel CA 2\Border_Crossing_Entry_Data[1].csv"
df = pd.read_csv(file_path, low_memory=False)
