import geopandas as gpd
import matplotlib.pyplot as plt

# 1. Define the file path (point directly to the .shp file)
# Ensure the .dbf, .prj, and .shx files are in the exact same directory
boundary_file = "samo_tracts/samo_boundary.shp"  # Update this path to the actual location of your shapefile

# 2. Load the Shapefile
print("Loading shapefile data...")
park_boundary = gpd.read_file(boundary_file)

# 3. Setup the Plot
fig, ax = plt.subplots(figsize=(10, 8))

# 4. Plot the Boundary
print("Rendering map...")
# Using a light fill color with a distinct border for visibility
park_boundary.plot(ax=ax, facecolor="lightblue", edgecolor='black', linewidth=1.5)

# 5. Map Formatting
ax.set_title("Santa Monica Mountains National Recreation Area Boundary", fontsize=14)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# Add a grid for easier spatial reference
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()