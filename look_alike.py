"""
Cluster Management Tool for Image Libraries

This script helps in organizing photo libraries by detecting faces and clustering them.
It uses face_recognition and sklearn for processing and clustering faces. The GUI is
built using Tkinter to interact with users for selecting photo directories, viewing 
clusters, and merging similar faces.

Steps Involved:
1. Load/Create face embeddings.
2. Cluster faces using DBSCAN.
3. Allow users to manually review and adjust clusters.
"""

import os
import numpy as np
import face_recognition
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from face_utils import show_similar_faces, process_and_cache_images
# For the GUI
import tkinter as tk
import hashlib
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import json

# Step 0: Prompt User to Select Photo Library Directory

# Initialize Tkinter root for dialogs
root = tk.Tk()
root.withdraw()  # Hide the root window

# Prompt the user to select the photo library directory
photo_dir = filedialog.askdirectory(title="Select Your Photo Library Directory")

# Check if a directory was selected
if not photo_dir:
    messagebox.showerror("Error", "No directory selected. Exiting the application.")
    exit()

# Step 1: Face Detection and Embedding

# Files to store face data
embeddings_file = 'face_embeddings.npy'
file_paths_file = 'face_file_paths.npy'
face_locations_file = 'face_locations.npy'

# Check if embeddings already exist
if os.path.exists(embeddings_file) and os.path.exists(file_paths_file) and os.path.exists(face_locations_file):
    print("Loading existing face embeddings...")
    face_embeddings = np.load(embeddings_file)
    file_paths = np.load(file_paths_file)
    face_locations_list = np.load(face_locations_file)
else:
    print("No existing embeddings found. Processing images to detect faces and compute embeddings...")
    # Store face embeddings, file paths, and face locations
    face_embeddings = []
    file_paths = []
    face_locations_list = []
    total_files = sum(len(files) for _, _, files in os.walk(photo_dir))
    processed_files = 0

    # Supported image extensions
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

    for root_dir, dirs, files in os.walk(photo_dir):
        for file in files:
            file_path = os.path.join(root_dir, file)
            processed_files += 1  # Increment processed files at the start
            # Check if the file has a supported image extension
            if file.lower().endswith(supported_extensions):
                try:
                    image = face_recognition.load_image_file(file_path)
                    face_locations = face_recognition.face_locations(image)
                    # Proceed if at least one face is found
                    if face_locations:
                        embeddings = face_recognition.face_encodings(image, face_locations)
                        for face_location, embedding in zip(face_locations, embeddings):
                            face_embeddings.append(embedding)
                            file_paths.append(file_path)
                            face_locations_list.append(face_location)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
            # Optional: Print progress every 100 files
            if processed_files % 100 == 0 or processed_files == total_files:
                print(f"Processed {processed_files}/{total_files} files.", end='\r')

    print(f"\nTotal faces detected: {len(face_embeddings)}")

    # Convert lists to numpy arrays
    face_embeddings = np.array(face_embeddings)
    file_paths = np.array(file_paths)
    face_locations_list = np.array(face_locations_list)

    # Save embeddings to files
    np.save(embeddings_file, face_embeddings)
    np.save(file_paths_file, file_paths)
    np.save(face_locations_file, face_locations_list)
    print("Face embeddings saved for future use.")

# Step 2: Face Clustering
print("Step 2: Clustering faces...")

# Adjusted clustering parameters (you may need to tweak these)
clustering = DBSCAN(eps=0.3, min_samples=30, metric='euclidean').fit(face_embeddings)
labels = clustering.labels_

# Debugging statements
print(f"Embeddings array shape: {face_embeddings.shape}")
print(f"Unique labels from clustering: {set(labels)}")
num_noise_points = list(labels).count(-1)
print(f"Number of noise points (label -1): {num_noise_points}")

# Organize images by their cluster labels
clustered_images = defaultdict(list)
unclustered_images = []  # Initialize unclustered images list

for label, file_path, face_location, embedding in zip(labels, file_paths, face_locations_list, face_embeddings):
    data = {
        'file_path': file_path,
        'embedding': embedding,
        'face_location': face_location
    }
    if label != -1:  # Exclude noise points
        clustered_images[label].append(data)
    else:
        # Add noise points to unclustered images
        unclustered_images.append(data)

# Get a sorted list of cluster labels
cluster_labels = sorted(clustered_images.keys())

print(f"Number of clusters formed: {len(cluster_labels)}")

# Create a cluster for unclustered images with label -2
if unclustered_images:
    unclustered_label = -2  # Use -2 to represent unclustered images
    clustered_images[unclustered_label] = unclustered_images
    cluster_labels.append(unclustered_label)
    cluster_labels.sort()
    print(f"Cluster for unclustered images created with label {unclustered_label}.")

# Print cluster sizes
for label in cluster_labels:
    print(f"Cluster {label} has {len(clustered_images[label])} images")

# Step 3: Run the caching script (create_image_cache.py) to cache images
print("Step 3: Running the caching script to cache images...")
process_and_cache_images()  # Call the function to process and cache images

# Step 4: Launching the GUI
print("Step 4: Launching the GUI...")

# Show the main application window
root.deiconify()
root.title("Cluster Management Tool")
root.geometry("1000x700")

current_cluster_index = 0
cluster_names = {}
current_images = []
selected_images = []  # To keep track of selected images for reassignment

# Load existing cluster names if available
if os.path.exists('cluster_names.json'):
    with open('cluster_names.json', 'r') as f:
        cluster_names = json.load(f)

# Scrollable frame setup
def on_frame_configure(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

canvas = tk.Canvas(root)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas.configure(yscrollcommand=scrollbar.set)

images_frame = ttk.Frame(canvas)
canvas.create_window((0, 0), window=images_frame, anchor='nw')

images_frame.bind("<Configure>", on_frame_configure)

# Bind the canvas to mouse wheel events for scrolling
def on_mouse_wheel(event):
    if event.num == 4 or event.delta > 0:  # Scroll up
        canvas.yview_scroll(-1, "units")
    elif event.num == 5 or event.delta < 0:  # Scroll down
        canvas.yview_scroll(1, "units")
# Bind scrolling to trackpad and mouse wheel
canvas.bind_all("<MouseWheel>", on_mouse_wheel)
# macOS (bind mouse button 4 and 5 for scrolling)
canvas.bind_all("<Button-4>", on_mouse_wheel)
canvas.bind_all("<Button-5>", on_mouse_wheel)

# Number of photos to show per page
PHOTOS_PER_PAGE = 80

# Current page index
current_page = 0

# Directory to store cached images
CACHE_DIR = 'image_cache'

# Function to generate cache filename based on file path and face location
def get_cache_filename(file_path, face_location):
    """Generate a unique cache filename based on file path and face location."""
    hash_input = f"{file_path}{face_location}".encode('utf-8')
    file_hash = hashlib.md5(hash_input).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.png")

# Function to load cached image if available, otherwise process and cache it
def load_cached_image(file_path, face_location):
    """Load cached image if available, otherwise process and cache it."""
    cache_filename = get_cache_filename(file_path, face_location)

    # Check if the image is already cached
    if os.path.exists(cache_filename):
        try:
            # Load the cached image
            img = Image.open(cache_filename)
            return ImageTk.PhotoImage(img)  # Return the cached image
        except Exception as e:
            print(f"Error loading cached image {cache_filename}: {e}")
    
    # If not cached, process the image and cache it
    try:
        img = Image.open(file_path)
        top, right, bottom, left = face_location
        img = img.crop((left, top, right, bottom))
        img = img.resize((150, 150), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        
        # Save the processed image to the cache
        img.save(cache_filename, format='PNG')
        
        return photo
    except Exception as e:
        print(f"Error processing image {file_path}: {e}")
        return None

def show_file_path(event, file_path):
    """Display the original file path in a message box when the user right-clicks on an image."""
    messagebox.showinfo("Image File Path", f"Original file path: {file_path}")

# Function to load and display images for the current cluster
def load_images(cluster_index):
    global selected_images, current_page
    selected_images = []
    # Clear previous images
    for widget in images_frame.winfo_children():
        widget.destroy()
    # Get image data for the current cluster
    cluster_label = cluster_labels[cluster_index]
    image_data = clustered_images[cluster_label]
    
    # Pagination logic: slice the image_data based on the current page
    start_idx = current_page * PHOTOS_PER_PAGE
    end_idx = start_idx + PHOTOS_PER_PAGE
    current_page_data = image_data[start_idx:end_idx]

    imgs = []
    for idx, data in enumerate(current_page_data):
        img_path = data['file_path']
        face_location = data['face_location']
        
        # Load the cached image or process and cache it
        photo = load_cached_image(img_path, face_location)
        if not photo:
            continue
        
        # Create a checkbox for image selection
        var = tk.IntVar()
        checkbox = tk.Checkbutton(images_frame, image=photo, variable=var)
        checkbox.image = photo  # Keep a reference to avoid garbage collection
        checkbox.var = var
        checkbox.grid(row=idx//4, column=idx%4, padx=5, pady=5)
        selected_images.append({'checkbox': checkbox, 'data': data})

    # Dynamically calculate the row for the navigation buttons
    num_rows = (len(current_page_data) + 3) // 4  # Calculate how many rows we need (4 images per row)

    # Add navigation buttons (Next/Previous)
    nav_frame = tk.Frame(images_frame)
    nav_frame.grid(row=num_rows + 1, columnspan=4, pady=10)
    
    # Add Previous button
    if current_page > 0:
        prev_button = tk.Button(nav_frame, text="Previous", command=lambda: change_page(-1))
        prev_button.pack(side=tk.LEFT, padx=5)

    # Add Next button
    if end_idx < len(image_data):
        next_button = tk.Button(nav_frame, text="Next", command=lambda: change_page(1))
        next_button.pack(side=tk.LEFT, padx=5)
    
    # Display current page information
    page_label = tk.Label(nav_frame, text=f"Page {current_page + 1} of {((len(image_data) - 1) // PHOTOS_PER_PAGE) + 1}")
    page_label.pack(side=tk.LEFT, padx=5)
    
    # Create a label to display the image
    img_label = tk.Label(images_frame, image=photo)
    img_label.image = photo  # Keep a reference to avoid garbage collection
    img_label.grid(row=idx // 4, column=idx % 4, padx=5, pady=5)
    
    # Bind right-click event to show the file path
    img_label.bind('<Button-3>', lambda event, path=img_path: show_file_path(event, path))
    return imgs

def change_page(direction):
    global current_page
    # Update the current page index
    current_page += direction

    # Ensure the page index stays within bounds
    if current_page < 0:
        current_page = 0
    elif current_page * PHOTOS_PER_PAGE >= len(clustered_images[cluster_labels[current_cluster_index]]):
        current_page -= 1  # Stay on the current page if out of bounds
    
    # Reload images with the updated page index
    load_images(current_cluster_index)

def update_ui():
    global current_images, current_page
    current_page = 0 # Reset to first page when changing clusters
    if cluster_labels:
        current_images = load_images(current_cluster_index)
        # Update cluster label display
        cluster_label = cluster_labels[current_cluster_index]
        if cluster_label == -2:
            cluster_label_var.set("Unclustered Images")
        else:
            cluster_label_var.set(f"Cluster {cluster_label}")

        # Load existing name if available
        name = cluster_names.get(str(cluster_label), "")
        name_var.set(name)
    else:
        # No clusters left
        current_images = []
        cluster_label_var.set("No clusters available")
        name_var.set("")
        # Clear images frame
        for widget in images_frame.winfo_children():
            widget.destroy()

def save_name():
    name = name_entry.get()
    if cluster_labels:
        cluster_label = cluster_labels[current_cluster_index]
        if name:
            cluster_names[str(cluster_label)] = name
        else:
            cluster_names.pop(str(cluster_label), None)

def next_cluster():
    save_name()
    global current_cluster_index
    if cluster_labels:
        if current_cluster_index < len(cluster_labels) - 1:
            current_cluster_index += 1
            update_ui()
        else:
            messagebox.showinfo("Info", "This is the last cluster.")
    else:
        messagebox.showinfo("Info", "No clusters available.")

def prev_cluster():
    save_name()
    global current_cluster_index
    if cluster_labels:
        if current_cluster_index > 0:
            current_cluster_index -= 1
            update_ui()
        else:
            messagebox.showinfo("Info", "This is the first cluster.")
    else:
        messagebox.showinfo("Info", "No clusters available.")

def remove_selected_images():
    global current_cluster_index
    # Remove selected images from current cluster
    if not cluster_labels:
        messagebox.showinfo("Info", "No clusters available.")
        return
    cluster_label = cluster_labels[current_cluster_index]
    to_remove = []
    to_remove_file_paths = set()
    for item in selected_images:
        if item['checkbox'].var.get() == 1:
            to_remove.append(item['data'])
            to_remove_file_paths.add(item['data']['file_path'])
    # Remove selected items by reconstructing the list based on file paths
    clustered_images[cluster_label] = [
        data for data in clustered_images[cluster_label]
        if data['file_path'] not in to_remove_file_paths
    ]
    # Add the removed images to unclustered_images
    unclustered_images.extend(to_remove)
    # If the cluster is empty, remove it
    if not clustered_images[cluster_label]:
        del clustered_images[cluster_label]
        cluster_labels.remove(cluster_label)
        if current_cluster_index >= len(cluster_labels):
            current_cluster_index = len(cluster_labels) - 1
    update_ui()

def reassign_selected_images():
    def assign_to_cluster():
        global current_cluster_index
        try:
            target_label = int(cluster_entry.get())
            if target_label not in cluster_labels:
                cluster_labels.append(target_label)
                cluster_labels.sort()
                clustered_images[target_label] = []
            # Move selected images to the target cluster
            cluster_label = cluster_labels[current_cluster_index]
            to_move = []
            to_move_file_paths = set()
            for item in selected_images:
                if item['checkbox'].var.get() == 1:
                    to_move.append(item['data'])
                    to_move_file_paths.add(item['data']['file_path'])
            # Remove selected items from current cluster
            clustered_images[cluster_label] = [
                data for data in clustered_images[cluster_label]
                if data['file_path'] not in to_move_file_paths
            ]
            # Add selected items to target cluster
            clustered_images[target_label].extend(to_move)
            # If the current cluster is empty, remove it
            if not clustered_images[cluster_label]:
                del clustered_images[cluster_label]
                cluster_labels.remove(cluster_label)
                if current_cluster_index >= len(cluster_labels):
                    current_cluster_index = len(cluster_labels) - 1
            assign_window.destroy()
            update_ui()
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid cluster number.")
    # Popup window to enter target cluster label
    assign_window = tk.Toplevel(root)
    assign_window.title("Reassign to Cluster")
    assign_window.geometry("300x100")
    tk.Label(assign_window, text="Enter Cluster Number:").pack(pady=5)
    cluster_entry = tk.Entry(assign_window)
    cluster_entry.pack(pady=5)
    tk.Button(assign_window, text="Assign", command=assign_to_cluster).pack(pady=5)

def save_all():
    save_name()
    # Save cluster names
    with open('cluster_names.json', 'w') as f:
        json.dump(cluster_names, f)
    # Save updated clusters
    # Map embeddings and file paths to new labels
    new_labels = []
    new_file_paths = []
    new_embeddings = []
    new_face_locations = []
    # Save clustered images
    for label in cluster_labels:
        for data in clustered_images[label]:
            new_labels.append(label)
            new_file_paths.append(data['file_path'])
            new_embeddings.append(data['embedding'])
            new_face_locations.append(data['face_location'])
    # Save unclustered images with label -1
    for data in unclustered_images:
        new_labels.append(-1)
        new_file_paths.append(data['file_path'])
        new_embeddings.append(data['embedding'])
        new_face_locations.append(data['face_location'])
    # Save to files
    np.save('updated_labels.npy', np.array(new_labels))
    np.save('updated_file_paths.npy', np.array(new_file_paths))
    np.save('updated_embeddings.npy', np.array(new_embeddings))
    np.save('updated_face_locations.npy', np.array(new_face_locations))
    messagebox.showinfo("Success", "All data saved successfully.")

def save_and_exit():
    save_all()
    root.destroy()

# UI Components
top_frame = ttk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

cluster_label_var = tk.StringVar()
cluster_label_display = ttk.Label(top_frame, textvariable=cluster_label_var, font=("Helvetica", 16))
cluster_label_display.pack(pady=10)

name_var = tk.StringVar()
name_frame = ttk.Frame(root)
name_frame.pack(pady=10)

name_label = ttk.Label(name_frame, text="Assign Name:")
name_label.pack(side=tk.LEFT)

name_entry = ttk.Entry(name_frame, textvariable=name_var)
name_entry.pack(side=tk.LEFT)

# Action Buttons
action_frame = ttk.Frame(root)
action_frame.pack(pady=10)

remove_button = ttk.Button(action_frame, text="Remove Selected Images", command=remove_selected_images)
remove_button.pack(padx=5)

reassign_button = ttk.Button(action_frame, text="Reassign Selected Images", command=reassign_selected_images)
reassign_button.pack(padx=5)

similar_button = ttk.Button(action_frame, text="Show Similar Faces", command=lambda: show_similar_faces(clustered_images, cluster_labels, current_cluster_index, root))
similar_button.pack(padx=5)

def reapply_clustering():
    global face_embeddings, clustered_images, unclustered_images, cluster_labels

    # Get new parameters from user input
    eps_value = eps_var.get()
    min_samples_value = min_samples_var.get()

    if not face_embeddings.size:
        messagebox.showerror("Error", "No face embeddings found to cluster.")
        return

    # Step 2: Re-cluster the faces with new parameters
    print(f"Reapplying clustering with EPS={eps_value} and Min Samples={min_samples_value}")
    
    clustering = DBSCAN(eps=eps_value, min_samples=min_samples_value, metric='euclidean').fit(face_embeddings)
    labels = clustering.labels_

    # Clear previous clusters
    clustered_images.clear()
    unclustered_images.clear()

    # Organize images by their cluster labels
    for label, file_path, face_location, embedding in zip(labels, file_paths, face_locations_list, face_embeddings):
        data = {
            'file_path': file_path,
            'embedding': embedding,
            'face_location': face_location
        }
        if label != -1:  # Exclude noise points
            clustered_images[label].append(data)
        else:
            # Add noise points to unclustered images
            unclustered_images.append(data)

    # Update the cluster labels list
    cluster_labels = sorted(clustered_images.keys())
    print(f"Unique labels from clustering: {set(labels)}") 

    # Add the unclustered cluster (with label -2) if there are unclustered images
    if unclustered_images:
        unclustered_label = -2
        clustered_images[unclustered_label] = unclustered_images
        cluster_labels.append(unclustered_label)

    # Refresh the UI to show the updated clusters
    update_ui()

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="black", relief="solid", borderwidth=1, padx=2, pady=2)
        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None

# Frame for DBSCAN parameter input
param_frame = ttk.Frame(root)
param_frame.pack(pady=10)

# EPS parameter
eps_label = ttk.Label(param_frame, text="EPS (Neighborhood Radius):")
eps_label.pack(side=tk.LEFT, padx=5)
eps_var = tk.DoubleVar(value=0.3)  # Default value of 0.3
eps_entry = ttk.Entry(param_frame, textvariable=eps_var, width=5)
eps_entry.pack(side=tk.LEFT, padx=5)

# Min Samples parameter
min_samples_label = ttk.Label(param_frame, text="Min Samples (Density Threshold):")
min_samples_label.pack(side=tk.LEFT, padx=5)
min_samples_var = tk.IntVar(value=30)  # Default value of 30
min_samples_entry = ttk.Entry(param_frame, textvariable=min_samples_var, width=5)
min_samples_entry.pack(side=tk.LEFT, padx=5)

# Button to apply new parameters and re-cluster
recluster_button = ttk.Button(param_frame, text="Reapply Clustering", command=lambda: reapply_clustering())
recluster_button.pack(side=tk.LEFT, padx=10)

# Apply Tooltips
ToolTip(eps_entry, "The maximum distance between two samples for them to be considered in the same neighborhood.")
ToolTip(min_samples_entry, "The minimum number of samples required to form a cluster.")

# Navigation Buttons
nav_frame = ttk.Frame(root)
nav_frame.pack(pady=10)

prev_button = ttk.Button(nav_frame, text="<< Previous", command=prev_cluster)
prev_button.pack(side=tk.LEFT, padx=5)

next_button = ttk.Button(nav_frame, text="Next >>", command=next_cluster)
next_button.pack(side=tk.LEFT, padx=5)

save_button = ttk.Button(root, text="Save All", command=save_all)
save_button.pack(pady=10)

exit_button = ttk.Button(root, text="Save and Exit", command=save_and_exit)
exit_button.pack(pady=10)

update_ui()
root.mainloop()
