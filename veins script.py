# ==================== Auto-install dependencies ====================
import subprocess
import sys

required_modules = ['pandas', 'numpy']
for module in required_modules:
    try:
        __import__(module)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", module])

# ==================== Imports ====================
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, DoubleVar, IntVar
import math
import json

# ==================== Vein Processing ====================
def process_veins():
    file_path = filedialog.askopenfilename(
        title="Select coal blocks locations file",
        filetypes=[("TSV files", "*.tsv"), ("All files", "*.*")]
    )
    if not file_path:
        messagebox.showwarning("No file", "No file was selected. Exiting.")
        return []

    data = pd.read_csv(file_path, sep="\t")
    coords = data["Coordinates"].str.split("/", expand=True).astype(int)
    coords.columns = ["x", "y", "z"]

    points = coords.to_numpy()
    point_to_index = {tuple(p): i for i, p in enumerate(points)}

    def get_neighbors(idx):
        x, y, z = points[idx]
        neighbor_positions = [
            (x+1, y, z), (x-1, y, z),
            (x, y+1, z), (x, y-1, z),
            (x, y, z+1), (x, y, z-1),
        ]
        return [point_to_index[n] for n in neighbor_positions if n in point_to_index]

    visited = set()
    groups = []
    for i in range(len(points)):
        if i not in visited:
            stack = [i]
            group = []
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    group.append(node)
                    stack.extend(get_neighbors(node))
            groups.append(group)

    vein_centers = []
    for group in groups:
        xs = [points[idx][0] for idx in group]
        ys = [points[idx][1] for idx in group]
        zs = [points[idx][2] for idx in group]
        center = {
            "x": int(sum(xs)/len(xs)),
            "y": int(sum(ys)/len(ys)),
            "z": int(sum(zs)/len(zs)),
            "r": 0,
            "g": 1,
            "b": 0,
            "options": {"name": ""}  # leave name blank
        }
        vein_centers.append(center)

    messagebox.showinfo(
        "Done",
        f"✅ Process complete!\n\nFound {len(vein_centers)} vein centers."
    )

    return vein_centers

# ==================== Path Ordering ====================
def order_points_by_path(points, start_target=None,
                         vertical_weight=0.5, loopback_weight=0.5,
                         max_y_movement=None, desired_length=None,
                         max_distance=None):
    if not points:
        return []

    def distance(p1, p2):
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        dz = p1['z'] - p2['z']
        return math.sqrt(dx*dx + dz*dz) + abs(dy) * vertical_weight

    if start_target:
        start_point = min(points, key=lambda p: distance(p, start_target))
    else:
        avg_x = sum(p['x'] for p in points) / len(points)
        avg_y = sum(p['y'] for p in points) / len(points)
        avg_z = sum(p['z'] for p in points) / len(points)
        start_point = min(points, key=lambda p: math.sqrt((p['x']-avg_x)**2 + (p['y']-avg_y)**2 + (p['z']-avg_z)**2))

    remaining = [p for p in points if p != start_point]
    path = [start_point]

    while remaining and (desired_length is None or len(path) < desired_length):
        last_point = path[-1]

        # Max Y filter
        if max_y_movement is not None:
            candidates = [p for p in remaining if abs(p['y'] - last_point['y']) <= max_y_movement]
        else:
            candidates = remaining[:]

        # Max distance filter
        if max_distance is not None:
            candidates = [p for p in candidates if math.sqrt((p['x']-last_point['x'])**2 +
                                                             (p['y']-last_point['y'])**2 +
                                                             (p['z']-last_point['z'])**2) <= max_distance]

        if not candidates:
            break

        def sort_key(p):
            dist = distance(last_point, p)
            direction_bonus = 0
            if len(path) > 1:
                prev = path[-2]
                dx1 = last_point['x'] - prev['x']
                dy1 = last_point['y'] - prev['y']
                dz1 = last_point['z'] - prev['z']
                dx2 = p['x'] - last_point['x']
                dy2 = p['y'] - last_point['y']
                dz2 = p['z'] - last_point['z']
                dot = dx1*dx2 + dy1*dy2 + dz1*dz2
                direction_bonus = -dot * loopback_weight
            return dist + direction_bonus

        candidates.sort(key=sort_key)
        chosen = candidates[0]
        path.append(chosen)
        remaining.remove(chosen)

    if desired_length is not None and len(path) > desired_length:
        path = path[:desired_length]

    for point in path:
        point['options']['name'] = ""  # leave name blank

    return path

def calculate_total_distance(path):
    total_distance = 0
    for i in range(1, len(path)):
        dx = path[i]['x'] - path[i-1]['x']
        dy = path[i]['y'] - path[i-1]['y']
        dz = path[i]['z'] - path[i-1]['z']
        total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)
    return total_distance

# ==================== GUI ====================
def run_gui():
    START_TARGET = {"x": 513, "y": 106, "z": 526}
    vein_centers = []

    root = tk.Tk()
    root.title("Vein Centers Path Tool")
    root.geometry("1000x850")
    root.minsize(900, 600)

    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)
    main_frame.grid_rowconfigure(6, weight=1)
    main_frame.grid_rowconfigure(8, weight=1)
    main_frame.grid_columnconfigure(1, weight=1)

    # --- Controls ---
    tk.Label(main_frame, text="Vertical Travel Weight (0-1)").grid(row=0, column=0, sticky="w", padx=5)
    vertical_weight_var = DoubleVar(value=0.5)
    tk.Scale(main_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=vertical_weight_var).grid(row=0, column=1, sticky="ew", padx=5)

    tk.Label(main_frame, text="Avoid Loop Backs Weight (0-1)").grid(row=1, column=0, sticky="w", padx=5)
    loopback_weight_var = DoubleVar(value=0.5)
    tk.Scale(main_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL, variable=loopback_weight_var).grid(row=1, column=1, sticky="ew", padx=5)

    tk.Label(main_frame, text="Max Vertical (Y) Movement Between Nodes (blocks)").grid(row=2, column=0, sticky="w", padx=5)
    max_y_var = IntVar(value=0)
    tk.Scale(main_frame, from_=0, to=50, resolution=1, orient=tk.HORIZONTAL, variable=max_y_var).grid(row=2, column=1, sticky="ew", padx=5)

    tk.Label(main_frame, text="Max Distance Between Nodes (units)").grid(row=3, column=0, sticky="w", padx=5)
    max_distance_var = IntVar(value=0)
    tk.Scale(main_frame, from_=0, to=100, resolution=1, orient=tk.HORIZONTAL, variable=max_distance_var).grid(row=3, column=1, sticky="ew", padx=5)

    tk.Label(main_frame, text="Desired Number of Nodes in Path (0 = no limit)").grid(row=4, column=0, sticky="w", padx=5)
    path_length_var = IntVar(value=0)
    tk.Entry(main_frame, textvariable=path_length_var).grid(row=4, column=1, sticky="ew", padx=5)

    # --- Text areas ---
    tk.Label(main_frame, text="Vein Centers:").grid(row=6, column=0, sticky="w", padx=5)
    input_textarea = scrolledtext.ScrolledText(main_frame, height=10)
    input_textarea.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=5, pady=2)

    tk.Label(main_frame, text="Processed waypoints:").grid(row=7, column=0, sticky="w", padx=5)
    output_textarea = scrolledtext.ScrolledText(main_frame, height=10)
    output_textarea.grid(row=8, column=0, columnspan=2, sticky="nsew", padx=5, pady=2)

    total_distance_label = tk.Label(main_frame, text="Total Distance: 0.00 units", fg="blue")
    total_distance_label.grid(row=9, column=0, columnspan=2, pady=5)

    node_count_label = tk.Label(main_frame, text="Nodes in Path: 0", fg="red")
    node_count_label.grid(row=10, column=0, columnspan=2, pady=5)

    # --- Load File Button ---
    def load_file():
        nonlocal vein_centers
        vein_centers = process_veins()
        input_textarea.delete("1.0", tk.END)
        input_textarea.insert(tk.END, json.dumps(vein_centers, indent=2))
        update_path()

    # --- Update Function ---
    def update_path(*args):
        if not vein_centers:
            return
        max_y = max_y_var.get() if max_y_var.get() > 0 else None
        max_dist = max_distance_var.get() if max_distance_var.get() > 0 else None
        desired_length = path_length_var.get() or None
        ordered_points = order_points_by_path(
            vein_centers,
            start_target=START_TARGET,
            vertical_weight=vertical_weight_var.get(),
            loopback_weight=loopback_weight_var.get(),
            max_y_movement=max_y,
            max_distance=max_dist,
            desired_length=desired_length
        )

        total_distance = calculate_total_distance(ordered_points)
        output_textarea.delete("1.0", tk.END)
        output_textarea.insert(tk.END, json.dumps(ordered_points, separators=(',', ':')))
        total_distance_label.config(text=f"Total Distance: {total_distance:.2f} units")

        # --- Node count live update ---
        current_count = len(ordered_points)
        if desired_length is not None and current_count < desired_length:
            node_count_label.config(text=f"Nodes in Path: {current_count} (below desired!)", fg="red")
        else:
            node_count_label.config(text=f"Nodes in Path: {current_count}", fg="green")

    # --- Bindings ---
    vertical_weight_var.trace("w", update_path)
    loopback_weight_var.trace("w", update_path)
    max_y_var.trace("w", update_path)
    max_distance_var.trace("w", update_path)
    path_length_var.trace("w", update_path)

    tk.Button(main_frame, text="Load Coal Blocks File", command=load_file).grid(row=11, column=0, columnspan=2, pady=5)

    root.mainloop()

if __name__ == "__main__":
    run_gui()
