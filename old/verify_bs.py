import pandas as pd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import Point
import pyproj
from pyproj import Transformer
import argparse
import os

def create_preview_map(ntm_file, lat, lon, radius_m, output_name):
    print(f"Generating preview map for {output_name}...")
    
    # 1. Load and Filter BTS
    df_ntm = pd.read_csv(ntm_file, sep=';', header=None, 
                         names=['Tech', 'MCC', 'MNC', 'CID', 'v1', 'eNB', 'v2', 'Lat', 'Lon', 'Desc', 'v3'],
                         on_bad_lines='skip')
    
    deg_radius = radius_m / 111000.0
    filtered_bts = df_ntm[
        (df_ntm['Lat'] > lat - deg_radius) & (df_ntm['Lat'] < lat + deg_radius) &
        (df_ntm['Lon'] > lon - deg_radius) & (df_ntm['Lon'] < lon + deg_radius)
    ].copy()

    # 2. Download Buildings for the map
    center_point = (lat, lon)
    try:
        buildings = ox.features.features_from_point(center_point, tags={'building': True}, dist=radius_m)
    except:
        print("No buildings found for preview.")
        buildings = None

    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    
    if buildings is not None:
        buildings.plot(ax=ax, facecolor='lightgray', edgecolor='gray', alpha=0.7)
    
    # Plot BTS
    if not filtered_bts.empty:
        ax.scatter(filtered_bts['Lon'], filtered_bts['Lat'], c='red', s=50, marker='^', label='BTS (from Database)')
        for idx, row in filtered_bts.iterrows():
            ax.annotate(f"CID:{row['CID']}", (row['Lon'], row['Lat']), fontsize=9, xytext=(5,5), textcoords='offset points')
    
    ax.scatter([lon], [lat], c='blue', s=100, marker='x', label='Center')
    
    ax.set_title(f"BTS Placement Verification - {output_name}\nLat:{lat}, Lon:{lon}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(f"{output_name}_preview.png", dpi=300, bbox_inches='tight')
    print(f"Preview map saved as: {output_name}_preview.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lat', type=float, default=41.2205)
    parser.add_argument('--lon', type=float, default=13.5698)
    parser.add_argument('--radius', type=float, default=500.0)
    parser.add_argument('--ntm', type=str, default='tim_20250716_lteitaly.ntm')
    parser.add_argument('--name', type=str, default='GAETA_VERIFICATION')
    
    args = parser.parse_args()
    create_preview_map(args.ntm, args.lat, args.lon, args.radius, args.name)
