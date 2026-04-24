import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import tkintermapview
import pandas as pd
import os
import json
import math
import numpy as np
import threading
import re
import pyproj
from geopy.geocoders import Nominatim

# Imports for Sionna scene generation
try:
    import osmnx as ox
    import pyvista as pv
    import shapely
    from shapely.geometry import shape, Polygon, LineString
    from shapely.ops import transform
    from pyproj import Transformer
    import open3d as o3d
    import xml.etree.ElementTree as ET
    import xml.dom.minidom as minidom
    HAS_SIONNA_LIBS = True
except ImportError:
    HAS_SIONNA_LIBS = False

class LTEItalySelectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sionna Project Designer - LTE Selection")
        self.root.geometry("1300x850")

        # --- Localization (English Only) ---
        self.texts = {
            "title": "Sionna Project Designer - LTE Selection",
            "db_label": "1. LTE Italy Database",
            "db_load": "Load .ntm file",
            "db_none": "No database loaded",
            "db_ready": "Database ready: {} BTS",
            "search_label": "2. Search Location",
            "search_btn": "Go",
            "live_label": "2b. Live View (Database)",
            "live_check": "Show BTS while browsing",
            "live_update": "Refresh View Now",
            "live_note": "Note: BTS appear only with zoom > 12",
            "cluster_check": "Cluster BTS with same Lat/Lon",
            "sel_label": "3. Selection Area",
            "sel_instr": "INSTRUCTIONS:\n1. Right click on map -> 'Add Polygon Vertex'\n2. Add at least 3 vertices\n3. Right click -> 'Close Selection Polygon'\n4. Export project.",
            "sel_reset": "Reset Area",
            "exp_label": "4. Export",
            "exp_name": "Project Name:",
            "exp_btn": "EXPORT FULL PROJECT (JSON + XML)",
            "log_label": "Console Log:",
            "app_start": "Application started. Load an .ntm database to begin.",
            "db_loaded": "Loaded database with {} antennas.",
            "map_moved": "Map moved to: {}",
            "loc_not_found": "Location '{}' not found.",
            "angle": "Corner",
            "bts_found": "Found {} BTS inside the selection polygon.",
            "reset_msg": "Selection reset.",
            "err_title": "Error",
            "err_load": "Loading failed: {}",
            "warn_title": "Warning",
            "warn_sel": "Select an area and close the polygon before exporting!",
            "success_title": "Success",
            "success_msg": "Project exported successfully!\nCreated JSON, XML and meshes.",
            "export_ok": "Project saved in: {}",
            "export_start": "Starting Sionna Scene export (Buildings & Roads)...",
            "export_mesh": "Generating mesh: {}...",
            "lib_error": "Sionna libraries (osmnx, pyvista, etc.) not found. Exporting JSON only."
        }

        self.df_ntm = None
        self.selection_points = []
        self.selection_polygon = None
        self.selection_closed = False
        self.bts_markers = []
        self.browsing_markers = []
        self.vertex_markers = []
        self.geolocator = Nominatim(user_agent="sionna_selector_v2")

        self.auto_refresh_var = tk.BooleanVar(value=True)
        self.cluster_same_coords_var = tk.BooleanVar(value=False)

        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)
        self.sidebar = ttk.Frame(self.paned, width=350)
        self.paned.add(self.sidebar)
        self.map_frame = ttk.Frame(self.paned)
        self.paned.add(self.map_frame)

        self.setup_sidebar()
        self.setup_map()
        self.log(self.t("app_start"))

    def t(self, key, *args):
        text = self.texts.get(key, key)
        if args: return text.format(*args)
        return text

    def setup_sidebar(self):
        p = {'padx': 10, 'pady': 5}
        db_f = ttk.LabelFrame(self.sidebar, text=self.t("db_label"))
        db_f.pack(fill=tk.X, **p)
        ttk.Button(db_f, text=self.t("db_load"), command=self.load_ntm).pack(fill=tk.X, padx=5, pady=5)
        self.db_status = ttk.Label(db_f, text=self.t("db_none"), foreground="red")
        self.db_status.pack(padx=5)

        search_f = ttk.LabelFrame(self.sidebar, text=self.t("search_label"))
        search_f.pack(fill=tk.X, **p)
        self.search_entry = ttk.Entry(search_f)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        self.search_entry.bind("<Return>", lambda e: self.search())
        ttk.Button(search_f, text=self.t("search_btn"), command=self.search).pack(side=tk.RIGHT, padx=5)

        live_f = ttk.LabelFrame(self.sidebar, text=self.t("live_label"))
        live_f.pack(fill=tk.X, **p)
        ttk.Checkbutton(live_f, text=self.t("live_check"), variable=self.auto_refresh_var).pack(anchor=tk.W, padx=5)
        ttk.Checkbutton(live_f, text=self.t("cluster_check"), variable=self.cluster_same_coords_var, command=self.refresh_all_bts_views).pack(anchor=tk.W, padx=5)
        ttk.Button(live_f, text=self.t("live_update"), command=self.update_browsing_bts).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(live_f, text=self.t("live_note"), font=("Arial", 7), foreground="gray").pack(padx=5)

        sel_f = ttk.LabelFrame(self.sidebar, text=self.t("sel_label"))
        sel_f.pack(fill=tk.X, **p)
        ttk.Label(sel_f, text=self.t("sel_instr"), font=("Arial", 8), justify=tk.LEFT).pack(padx=5, pady=5)
        ttk.Button(sel_f, text=self.t("sel_reset"), command=self.reset_selection).pack(fill=tk.X, padx=5, pady=5)

        exp_f = ttk.LabelFrame(self.sidebar, text=self.t("exp_label"))
        exp_f.pack(fill=tk.X, **p)
        ttk.Label(exp_f, text=self.t("exp_name")).pack(anchor=tk.W, padx=5)
        self.name_var = tk.StringVar(value="Sionna_Project_1")
        ttk.Entry(exp_f, textvariable=self.name_var).pack(fill=tk.X, padx=5)
        self.export_btn = ttk.Button(exp_f, text=self.t("exp_btn"), command=self.export_project)
        self.export_btn.pack(fill=tk.X, padx=5, pady=10)

        ttk.Label(self.sidebar, text=self.t("log_label")).pack(anchor=tk.W, padx=10)
        self.log_box = tk.Text(self.sidebar, height=15, font=("Consolas", 8), bg="#f0f0f0")
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def setup_map(self):
        self.map_widget = tkintermapview.TkinterMapView(self.map_frame, corner_radius=0)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        self.map_widget.set_position(45.4642, 9.1900) 
        self.map_widget.set_zoom(14)
        self.map_widget.add_right_click_menu_command(label="Add Polygon Vertex", command=self.add_selection_point, pass_coords=True)
        self.map_widget.add_right_click_menu_command(label="Close Selection Polygon", command=self.close_selection_polygon, pass_coords=False)
        self.map_widget.canvas.bind("<ButtonRelease-1>", lambda e: self.update_browsing_bts())

    def load_ntm(self):
        f = filedialog.askopenfilename(filetypes=[("NTM files", "*.ntm")])
        if f:
            try:
                self.df_ntm = pd.read_csv(f, sep=';', header=None, names=['Tech', 'MCC', 'MNC', 'CID', 'v1', 'eNB', 'v2', 'Lat', 'Lon', 'Desc', 'v3'], on_bad_lines='skip')
                self.db_status.config(text=self.t("db_ready", len(self.df_ntm)), foreground="green")
                self.log(self.t("db_loaded", len(self.df_ntm)))
                self.update_browsing_bts()
            except Exception as e:
                messagebox.showerror(self.t("err_title"), self.t("err_load", str(e)))

    def search(self):
        addr = self.search_entry.get()
        if addr:
            loc = self.geolocator.geocode(addr)
            if loc:
                self.map_widget.set_position(loc.latitude, loc.longitude)
                self.log(self.t("map_moved", addr))
                self.root.after(500, self.update_browsing_bts)
            else:
                self.log(self.t("loc_not_found", addr))

    def update_browsing_bts(self, event=None):
        if self.df_ntm is None or not self.auto_refresh_var.get(): return
        zoom = self.map_widget.zoom
        if zoom < 12:
            if self.browsing_markers:
                for m in self.browsing_markers: m.delete()
                self.browsing_markers = []
            return
        pos = self.map_widget.get_position()
        delta = 0.5 / (2 ** (zoom - 10))
        nearby = self.df_ntm[(self.df_ntm['Lat'].between(pos[0] - delta, pos[0] + delta)) & (self.df_ntm['Lon'].between(pos[1] - delta, pos[1] + delta))].copy()
        if self.cluster_same_coords_var.get(): nearby = self.cluster_colocated_bts(nearby)
        nearby = nearby.head(100)
        for m in self.browsing_markers: m.delete()
        self.browsing_markers = []
        for _, bts in nearby.iterrows():
            m = self.map_widget.set_marker(bts['Lat'], bts['Lon'], text=f"{bts['Tech']}", marker_color_circle="gray", marker_color_outside="white", font="Arial 7")
            self.browsing_markers.append(m)

    def refresh_all_bts_views(self):
        self.update_browsing_bts()
        if self.selection_closed: self.filter_bts()

    def cluster_colocated_bts(self, df):
        if df is None or df.empty: return pd.DataFrame(columns=df.columns if df is not None else None)
        rows = []
        grouped = df.groupby(['Lat', 'Lon'], dropna=False, sort=False)
        for (_, _), group in grouped:
            first = group.iloc[0].copy()
            cluster_size = int(len(group))
            first['cluster_size'] = cluster_size
            first['cluster_cids'] = ','.join(group['CID'].astype(str).tolist())
            if cluster_size > 1:
                tech_values = sorted({str(Tech) for Tech in group['Tech'].dropna().astype(str)})
                first['Tech'] = tech_values[0] if len(tech_values) == 1 else f"MIXED({len(tech_values)})"
                first['Desc'] = f"Cluster of {cluster_size} BTS at same coordinates"
            rows.append(first)
        return pd.DataFrame(rows)

    def add_selection_point(self, coords):
        if self.selection_closed: self.reset_selection()
        self.selection_points.append(coords)
        m = self.map_widget.set_marker(coords[0], coords[1], text=f"{self.t('angle')} {len(self.selection_points)}", marker_color_circle="red")
        self.vertex_markers.append(m)
        if len(self.selection_points) >= 2:
            if self.selection_polygon: self.selection_polygon.delete()
            self.selection_polygon = self.map_widget.set_path(self.selection_points, color="red", width=2)

    def close_selection_polygon(self):
        if len(self.selection_points) < 3: return
        if self.selection_polygon: self.selection_polygon.delete()
        closed_path = self.selection_points + [self.selection_points[0]]
        self.selection_polygon = self.map_widget.set_path(closed_path, color="red", width=2)
        self.selection_closed = True
        self.filter_bts()

    def get_selection_polygon_lonlat(self):
        if len(self.selection_points) < 3 or not self.selection_closed: return None
        poly = Polygon([(pt[1], pt[0]) for pt in self.selection_points])
        return poly if poly.is_valid else poly.buffer(0)

    def filter_bts(self):
        if self.df_ntm is None: return
        nearby = self.selected_bts()
        for m in self.bts_markers: m.delete()
        self.bts_markers = []
        for _, bts in nearby.iterrows():
            marker_text = f"{bts['Tech']} {bts['CID']}"
            m = self.map_widget.set_marker(bts['Lat'], bts['Lon'], text=marker_text, marker_color_circle="blue", font="Arial 8 bold")
            self.bts_markers.append(m)
        self.log(self.t("bts_found", len(nearby)))

    def selected_bts(self):
        if self.df_ntm is None: return pd.DataFrame()
        poly = self.get_selection_polygon_lonlat()
        if poly is None: return pd.DataFrame()
        mask = [poly.contains(shapely.geometry.Point(lon, lat)) for lat, lon in zip(self.df_ntm['Lat'], self.df_ntm['Lon'])]
        selected = self.df_ntm[mask].copy()
        if self.cluster_same_coords_var.get(): selected = self.cluster_colocated_bts(selected)
        return selected

    def _safe_numeric(self, value):
        if pd.isna(value): return np.nan
        if isinstance(value, (int, float, np.number)): return float(value)
        text = str(value).strip()
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group(0)) if match else np.nan

    def resolve_building_height(self, row):
        h = self._safe_numeric(row.get('height', np.nan))
        if np.isfinite(h) and h > 0: return h, "height"
        levels = self._safe_numeric(row.get('building:levels', np.nan))
        if np.isfinite(levels) and levels > 0: return levels * 3.5, "levels"
        return 3.5, "default"

    def _to_json_scalar(self, value):
        if pd.isna(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value

    def convert_lane_to_numeric(self, lane):
        if lane is None:
            return None
        if isinstance(lane, (int, float, np.number)):
            return float(lane)
        text = str(lane).strip()
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text)
        return float(match.group(0)) if match else None

    def calculate_edge_geometry(self, graph, u, v):
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        return LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])

    def build_project_json(self, poly_lonlat, utm_epsg=None):
        if poly_lonlat is None:
            return None

        if utm_epsg is None:
            utm_epsg = self.get_utm_epsg(poly_lonlat.centroid.x)

        min_lon, min_lat, max_lon, max_lat = poly_lonlat.bounds
        center_lon = poly_lonlat.centroid.x
        center_lat = poly_lonlat.centroid.y
        nearby = self.selected_bts() if self.df_ntm is not None else pd.DataFrame()

        transmitters = []
        if nearby is not None and not nearby.empty:
            for _, row in nearby.iterrows():
                tx = {col: self._to_json_scalar(row[col]) for col in nearby.columns}
                transmitters.append(tx)

        return {
            "project_name": self.name_var.get(),
            "utm_epsg": utm_epsg,
            "area": {
                "min_lat": min_lat,
                "max_lat": max_lat,
                "min_lon": min_lon,
                "max_lon": max_lon,
                "polygon": [{"lat": lat, "lon": lon} for lon, lat in poly_lonlat.exterior.coords],
            },
            "center": {"lat": center_lat, "lon": center_lon},
            "transmitters": transmitters,
        }

    def reset_selection(self):
        self.selection_points, self.selection_closed = [], False
        if self.selection_polygon: self.selection_polygon.delete()
        for m in self.vertex_markers + self.bts_markers: m.delete()
        self.vertex_markers, self.bts_markers, self.selection_polygon = [], [], None
        self.log(self.t("reset_msg"))

    def get_utm_epsg(self, lon):
        zone = int(math.floor((lon + 180) / 6) + 1)
        return f"EPSG:{32600 + zone}"

    def points_2d_to_poly(self, points, z):
        valid = [p for p in points if not any(math.isnan(coord) for coord in p)]
        if len(valid) < 3: return None
        return pv.PolyData([p + (z,) for p in valid], faces=[len(valid), *range(len(valid))])

    def export_project(self):
        if len(self.selection_points) < 3 or not self.selection_closed:
            messagebox.showwarning(self.t("warn_title"), self.t("warn_sel")); return
        if not HAS_SIONNA_LIBS:
            self.log(self.t("lib_error")); self.export_json_only(); return
        self.export_btn.config(state=tk.DISABLED)
        threading.Thread(target=self._run_export_task, daemon=True).start()

    def _run_export_task(self):
        try:
            ox.settings.use_cache = True
            ox.settings.timeout = 180
            proj_name = self.name_var.get()
            poly_lonlat = self.get_selection_polygon_lonlat()
            if poly_lonlat.geom_type == 'MultiPolygon': poly_lonlat = max(poly_lonlat.geoms, key=lambda p: p.area)
            
            utm_epsg = self.get_utm_epsg(poly_lonlat.centroid.x)
            transformer = Transformer.from_crs("EPSG:4326", utm_epsg, always_xy=True)
            
            # Project area boundary and calculate center in meters
            projected_coords = [transformer.transform(lon, lat) for lon, lat in poly_lonlat.exterior.coords]
            projected_poly = Polygon(projected_coords)
            center_x, center_y = projected_poly.centroid.x, projected_poly.centroid.y
            
            location_dir = f"{proj_name}_{center_x}_{center_y}"
            base_path = os.path.join("simple_scene", location_dir)
            mesh_path = os.path.join(base_path, "mesh")
            os.makedirs(mesh_path, exist_ok=True)

            self.log(self.t("export_start"))
            scene_xml = ET.Element("scene", version="2.1.0")
            
            # --- XML Defaults & Integrator (Notebook Cell 3) ---
            ET.SubElement(scene_xml, "default", name="spp", value="4096")
            ET.SubElement(scene_xml, "default", name="resx", value="1024")
            ET.SubElement(scene_xml, "default", name="resy", value="768")
            
            integrator = ET.SubElement(scene_xml, "integrator", type="path")
            ET.SubElement(integrator, "integer", name="max_depth", value="12")

            # --- Materials (Notebook Cell 3 Colors) ---
            mats = {
                "mat-itu_concrete": (0.539479, 0.539479, 0.539480),
                "mat-itu_marble": (0.701101, 0.644479, 0.485150),
                "mat-itu_metal": (0.219526, 0.219526, 0.254152),
                "mat-itu_wood": (0.043, 0.58, 0.184),
                "mat-itu_wet_ground": (0.91, 0.569, 0.055),
            }
            for mid, rgb in mats.items():
                bsdf_twosided = ET.SubElement(scene_xml, "bsdf", type="twosided", id=mid)
                bsdf_diffuse = ET.SubElement(bsdf_twosided, "bsdf", type="diffuse")
                ET.SubElement(bsdf_diffuse, "rgb", value=f"{rgb[0]} {rgb[1]} {rgb[2]}", name="reflectance")

            # --- Emitter & Camera ---
            emitter = ET.SubElement(scene_xml, "emitter", type="constant", id="World")
            ET.SubElement(emitter, "rgb", value="1.000000 1.000000 1.000000", name="radiance")
            
            sensor = ET.SubElement(scene_xml, "sensor", type="perspective", id="Camera")
            ET.SubElement(sensor, "string", name="fov_axis", value="x")
            ET.SubElement(sensor, "float", name="fov", value="42.854885")
            ET.SubElement(sensor, "float", name="near_clip", value="0.1")
            ET.SubElement(sensor, "float", name="far_clip", value="10000.0")
            
            trans = ET.SubElement(sensor, "transform", name="to_world")
            ET.SubElement(trans, "lookat", origin="0, 0, 500", target="0, 0, 0", up="0, 1, 0")
            
            sampler = ET.SubElement(sensor, "sampler", type="independent")
            ET.SubElement(sampler, "integer", name="sample_count", value="$spp")
            film = ET.SubElement(sensor, "film", type="hdrfilm")
            ET.SubElement(film, "integer", name="width", value="$resx")
            ET.SubElement(film, "integer", name="height", value="$resy")

            # --- Ground (Notebook Cell 11 Logic) ---
            self.log(self.t("export_mesh", "Ground"))
            oriented_ground = list(projected_poly.exterior.coords)
            if projected_poly.exterior.is_ccw: oriented_ground.reverse()
            ground_pts = [(c[0]-center_x, c[1]-center_y) for c in oriented_ground]
            g_mesh = self.points_2d_to_poly(ground_pts, 0).delaunay_2d()
            # Scaling relative to center
            g_mesh.points[:] = (g_mesh.points - g_mesh.center)*1.5 + g_mesh.center
            pv.save_meshio(os.path.join(mesh_path, "ground.ply"), g_mesh)
            
            shp = ET.SubElement(scene_xml, "shape", type="ply", id="mesh-ground")
            ET.SubElement(shp, "string", name="filename", value="mesh/ground.ply")
            ET.SubElement(shp, "ref", id="mat-itu_wet_ground", name="bsdf")
            ET.SubElement(shp, "boolean", name="face_normals", value="true")

            # --- Buildings (Notebook Cell 14 Logic) ---
            self.log("Querying OpenStreetMap for buildings...")
            buildings = ox.features.features_from_polygon(poly_lonlat, tags={'building': True})
            if not buildings.empty:
                buildings = buildings[buildings.intersects(poly_lonlat)]

            buildings_list = buildings.to_dict('records')
            self.log(f"Buildings candidates: {len(buildings_list)}")

            source_crs = pyproj.CRS(buildings.crs) if buildings.crs else pyproj.CRS("EPSG:4326")
            target_crs = pyproj.CRS(utm_epsg)
            building_transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True).transform

            mesh_count = 0
            for i, building in enumerate(buildings_list):
                try:
                    building_polygon = shape(building['geometry'])
                    if building_polygon.geom_type != 'Polygon':
                        continue

                    building_polygon = transform(building_transformer, building_polygon)

                    building_height, _ = self.resolve_building_height(building)

                    z_coordinates = np.full(len(building_polygon.exterior.coords), 0)
                    exterior_coords = building_polygon.exterior.coords
                    oriented_coords = list(exterior_coords)
                    if building_polygon.exterior.is_ccw:
                        oriented_coords.reverse()

                    points = [(coord[0] - center_x, coord[1] - center_y) for coord in oriented_coords]
                    boundary_points_polydata = self.points_2d_to_poly(points, z_coordinates[0])
                    if boundary_points_polydata is None:
                        continue

                    footprint_plane = boundary_points_polydata.delaunay_2d()
                    footprint_plane = footprint_plane.triangulate()
                    footprint_3d = footprint_plane.extrude((0, 0, building_height), capping=True)

                    full_mesh_path = os.path.join(mesh_path, f"building_{mesh_count}.ply")
                    footprint_3d.save(full_mesh_path)

                    local_mesh = o3d.io.read_triangle_mesh(full_mesh_path)
                    o3d.io.write_triangle_mesh(full_mesh_path, local_mesh)

                    ss = ET.SubElement(scene_xml, "shape", type="ply", id=f"mesh-building_{mesh_count}")
                    ET.SubElement(ss, "string", name="filename", value=f"mesh/building_{mesh_count}.ply")
                    ET.SubElement(ss, "ref", id="mat-itu_marble", name="bsdf")
                    ET.SubElement(ss, "boolean", name="face_normals", value="true")
                    mesh_count += 1
                except Exception:
                    continue

                if (i + 1) % 50 == 0:
                    self.log(f"Buildings processed: {i + 1}/{len(buildings_list)} (exported {mesh_count})")

            self.log(f"Buildings exported: {mesh_count}")

            # --- Roads (Notebook Cell 17 & 18 Logic) ---
            self.log("Querying OpenStreetMap for roads...")
            try:
                # Use simplify=False as in notebook cell 17
                graph = ox.graph_from_polygon(poly_lonlat, simplify=False, retain_all=True, truncate_by_edge=True, network_type='drive')
                graph = ox.project_graph(graph, to_crs=utm_epsg)
                mesh_collection = pv.PolyData()
                for u, v, key, data in graph.edges(keys=True, data=True):
                    if 'geometry' not in data:
                        data['geometry'] = self.calculate_edge_geometry(graph, u, v)
                    
                    if not data['geometry'].intersects(projected_poly): continue
                    
                    lanes = data.get('lanes', 1)
                    if not isinstance(lanes, list):
                        lanes = [lanes]

                    num_lanes = [self.convert_lane_to_numeric(lane) for lane in lanes]
                    num_lanes = [lane for lane in num_lanes if lane is not None]
                    if not num_lanes:
                        continue
                    rw = num_lanes[0] * 3.5
                    
                    buf = data['geometry'].buffer(rw)
                    polys = [buf] if buf.geom_type == 'Polygon' else list(buf.geoms)
                    for rp in polys:
                        if rp.is_empty or not hasattr(rp, 'exterior'): continue
                        oriented_r = list(rp.exterior.coords)
                        if rp.exterior.is_ccw: oriented_r.reverse()
                        pts = [(c[0]-center_x, c[1]-center_y) for c in oriented_r]
                        raw = self.points_2d_to_poly(pts, 0.25)
                        if raw:
                            mesh_collection = mesh_collection + raw.delaunay_2d()
                
                if mesh_collection.n_points > 0:
                    pv.save_meshio(os.path.join(mesh_path, "road_mesh_combined.ply"), mesh_collection)
                    rs = ET.SubElement(scene_xml, "shape", type="ply", id="mesh-roads_combined")
                    ET.SubElement(rs, "string", name="filename", value="mesh/road_mesh_combined.ply")
                    ET.SubElement(rs, "ref", id="mat-itu_concrete", name="bsdf")
                    ET.SubElement(rs, "boolean", name="face_normals", value="true")
            except Exception as road_err:
                self.log(f"Road export failed: {road_err}")

            # Save XML
            xml_str = minidom.parseString(ET.tostring(scene_xml)).toprettyxml(indent="    ")
            with open(os.path.join(base_path, "simple_OSM_scene.xml"), "w", encoding="utf-8") as f: f.write(xml_str)
            
            # Export project JSON in the same style used by the notebook flow
            self.export_json_only(os.path.join(base_path, f"{proj_name}.json"))
            
            self.root.after(0, lambda: self._on_export_finished(True, base_path))
        except Exception as ex:
            self.root.after(0, lambda: self._on_export_finished(False, str(ex)))

    def _on_export_finished(self, success, result):
        self.export_btn.config(state=tk.NORMAL)
        if success: self.log(self.t("export_ok", result)); messagebox.showinfo(self.t("success_title"), self.t("success_msg"))
        else: self.log(self.t("err_load", result)); messagebox.showerror(self.t("err_title"), self.t("err_load", result))

    def export_json_only(self, path=None):
        poly = self.get_selection_polygon_lonlat()
        if poly is None: return
        project = self.build_project_json(poly)
        if not path: path = filedialog.asksaveasfilename(defaultextension=".json", initialfile=f"{project['project_name']}.json")
        if path:
            with open(path, "w", encoding="utf-8") as f: json.dump(project, f, indent=4, ensure_ascii=False)

    def log(self, msg):
        if hasattr(self, 'log_box'):
            def append(): self.log_box.insert(tk.END, f"> {msg}\n"); self.log_box.see(tk.END)
            if threading.current_thread() is threading.main_thread(): append()
            else: self.root.after(0, append)

if __name__ == "__main__":
    root = tk.Tk(); app = LTEItalySelectionGUI(root); root.mainloop()
