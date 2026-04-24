import pandas as pd
import numpy as np
import os
import sys
import argparse
import osmnx as ox
import pyproj
import shapely
from shapely.geometry import shape, Polygon, Point, LineString
from shapely.ops import transform
import math
import pyvista as pv
from pyproj import Transformer
import open3d as o3d
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def points_2d_to_poly(points, z):
    faces = [len(points), *range(len(points))]
    poly = pv.PolyData([p + (z,) for p in points], faces=faces)
    return poly

def generate_sionna_scene(ntm_file, lat, lon, radius_m, location_str, target_epsg_code):
    target_epsg = f"EPSG:{target_epsg_code}"
    print(f"--- Starting Scene Generation: {location_str} ---")
    
    # 1. Load NTM Data
    print(f"Loading BTS data from {ntm_file}...")
    df_ntm = pd.read_csv(ntm_file, sep=';', header=None, 
                         names=['Tech', 'MCC', 'MNC', 'CID', 'v1', 'eNB', 'v2', 'Lat', 'Lon', 'Desc', 'v3'],
                         on_bad_lines='skip')
    
    # Filter BTS
    deg_radius = radius_m / 111000.0
    filtered_bts = df_ntm[
        (df_ntm['Lat'] > lat - deg_radius) & (df_ntm['Lat'] < lat + deg_radius) &
        (df_ntm['Lon'] > lon - deg_radius) & (df_ntm['Lon'] < lon + deg_radius)
    ].copy()
    print(f"Found {len(filtered_bts)} BTS in range.")

    # 2. Setup Directories
    wsg84 = pyproj.CRS("epsg:4326")
    target_crs = pyproj.CRS(target_epsg)
    transformer = Transformer.from_crs(wsg84, target_crs, always_xy=True)
    
    center_target = transformer.transform(lon, lat)
    center_x, center_y = center_target
    
    location_dir = f"{location_str}_{center_x}_{center_y}"
    output_base_dir = f"simple_scene/{location_dir}"
    os.makedirs(f"{output_base_dir}/mesh", exist_ok=True)

    # 3. Download OSM Data
    print(f"Downloading OSM buildings for radius {radius_m}m...")
    aoi_poly_target = Point(center_x, center_y).buffer(radius_m)
    back_transformer = Transformer.from_crs(target_crs, wsg84, always_xy=True)
    aoi_poly_wsg84 = transform(lambda x, y: back_transformer.transform(x, y), aoi_poly_target)
    
    try:
        buildings = ox.features.features_from_polygon(aoi_poly_wsg84, tags={'building': True})
        filtered_buildings = buildings[buildings.intersects(aoi_poly_wsg84)]
        print(f"Processing {len(filtered_buildings)} buildings...")
    except Exception as e:
        print(f"No buildings found or error: {e}")
        filtered_buildings = []

    # 4. Initialize XML
    scene = ET.Element("scene", version="2.1.0")
    ET.SubElement(scene, "default", name="spp", value="1024")
    ET.SubElement(scene, "default", name="resx", value="1024")
    ET.SubElement(scene, "default", name="resy", value="768")
    integrator = ET.SubElement(scene, "integrator", type="path")
    ET.SubElement(integrator, "integer", name="max_depth", value="12")

    # Materials
    material_colors = {
        "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
        "mat-itu_marble": (0.701101, 0.644479, 0.485150),
        "mat-itu_wet_ground": (0.91, 0.569, 0.055),
    }
    for mat_id, rgb in material_colors.items():
        bsdf_twosided = ET.SubElement(scene, "bsdf", type="twosided", id=mat_id)
        bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
        ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

    # 5. Ground
    print("Generating ground...")
    ground_coords = list(aoi_poly_target.exterior.coords)
    ground_points = [(c[0]-center_x, c[1]-center_y) for c in ground_coords]
    ground_polydata = points_2d_to_poly(ground_points, 0)
    ground_mesh = ground_polydata.delaunay_2d()
    ground_file = f"{output_base_dir}/mesh/ground.ply"
    pv.save_meshio(ground_file, ground_mesh)
    
    s_shape = ET.SubElement(scene, "shape", type="ply", id="mesh-ground")
    ET.SubElement(s_shape, "string", name="filename", value=f"mesh/ground.ply") # Relative path for Sionna
    ET.SubElement(s_shape, "ref", id="mat-itu_wet_ground", name="bsdf")

    # 6. Buildings
    b_transformer = Transformer.from_crs(wsg84, target_crs, always_xy=True).transform
    for idx, (b_idx, building) in enumerate(filtered_buildings.iterrows()):
        b_poly = building['geometry']
        if b_poly.geom_type != 'Polygon': continue
        b_poly = transform(b_transformer, b_poly)
        
        levels = building.get('building:levels', 1)
        try: height = float(levels) * 3.5
        except: height = 10.0
        
        oriented_coords = list(b_poly.exterior.coords)
        if b_poly.exterior.is_ccw: oriented_coords.reverse()
        pts = [(c[0]-center_x, c[1]-center_y) for c in oriented_coords]
        
        b_polydata = points_2d_to_poly(pts, 0)
        b_mesh = b_polydata.delaunay_2d().triangulate().extrude((0, 0, height), capping=True)
        b_file = f"{output_base_dir}/mesh/building_{idx}.ply"
        b_mesh.save(b_file)
        
        # Open3D cleanup (headless)
        o3d_mesh = o3d.io.read_triangle_mesh(b_file)
        o3d.io.write_triangle_mesh(b_file, o3d_mesh)
        
        s_shape = ET.SubElement(scene, "shape", type="ply", id=f"mesh-building_{idx}")
        ET.SubElement(s_shape, "string", name="filename", value=f"mesh/building_{idx}.ply")
        ET.SubElement(s_shape, "ref", id="mat-itu_marble", name="bsdf")

    # 7. Roads
    print("Downloading roads...")
    try:
        G = ox.graph_from_polygon(polygon=aoi_poly_wsg84, simplify=False, retain_all=True, truncate_by_edge=True, network_type='drive')
        graph = ox.project_graph(G, to_crs=target_epsg)
        mesh_collection = pv.PolyData()
        for u, v, key, data in graph.edges(keys=True, data=True):
            if 'geometry' not in data:
                u_data, v_data = graph.nodes[u], graph.nodes[v]
                data['geometry'] = LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])
            lanes = data.get('lanes', 1)
            if isinstance(lanes, list): lanes = lanes[0]
            try: road_width = float(lanes) * 3.5
            except: road_width = 3.5
            line_buffer = data['geometry'].buffer(road_width)
            if not line_buffer.is_empty and line_buffer.geom_type == 'Polygon':
                oriented_coords = list(line_buffer.exterior.coords)
                if line_buffer.exterior.is_ccw: oriented_coords.reverse()
                pts = [(c[0]-center_x, c[1]-center_y) for c in oriented_coords]
                r_mesh = points_2d_to_poly(pts, 0.1).delaunay_2d()
                mesh_collection = mesh_collection + r_mesh
        
        road_file = f"{output_base_dir}/mesh/roads_combined.ply"
        pv.save_meshio(road_file, mesh_collection)
        s_shape = ET.SubElement(scene, "shape", type="ply", id="mesh-roads")
        ET.SubElement(s_shape, "string", name="filename", value="mesh/roads_combined.ply")
        ET.SubElement(s_shape, "ref", id="mat-itu_concrete", name="bsdf")
    except:
        print("Road generation skipped.")

    # 8. Transmitters
    print("Adding BTS transmitters...")
    for idx, row in filtered_bts.iterrows():
        tx_target = transformer.transform(row['Lon'], row['Lat'])
        tx_x, tx_y = tx_target[0] - center_x, tx_target[1] - center_y
        transmitter = ET.SubElement(scene, "transmitter", name=f"BTS_{row['CID']}")
        pos = ET.SubElement(transmitter, "point", name="position")
        pos.set("x", str(tx_x)); pos.set("y", str(tx_y)); pos.set("z", "30")

    # 9. Save XML
    xml_string = ET.tostring(scene, encoding="utf-8")
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="    ")
    xml_path = f"{output_base_dir}/simple_OSM_scene.xml"
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_pretty)
    
    print(f"--- Done! Files saved in: {output_base_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sionna Scene Generator CLI')
    parser.add_argument('--ntm', type=str, default='tim_20250716_lteitaly.ntm', help='Path to NTM file')
    parser.add_argument('--lat', type=float, default=41.2205, help='Latitude')
    parser.add_argument('--lon', type=float, default=13.5698, help='Longitude')
    parser.add_argument('--radius', type=float, default=500.0, help='Radius in meters')
    parser.add_argument('--name', type=str, default='GAETA_SCENE', help='Location name')
    parser.add_argument('--epsg', type=int, default=32633, help='Target EPSG (e.g. 32633 for Italy UTM)')
    
    args = parser.parse_args()
    generate_sionna_scene(args.ntm, args.lat, args.lon, args.radius, args.name, args.epsg)
