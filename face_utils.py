"""
Face Utilities for Clustering and Image Management

This script contains utility functions that assist in face detection, clustering, 
and manual review tasks. The primary functions include image processing for displaying 
in the GUI and functions to assist with showing similar faces and clustering.

Functions:
1. process_and_cache_images: Caches processed images for efficient GUI rendering.
2. show_similar_faces: Displays similar faces to allow users to add them to clusters.
"""

import os
import numpy as np
import hashlib
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import cosine_similarity

# Directory to store cached images
CACHE_DIR = 'image_cache'

# Create the cache directory if it doesn't exist
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(file_path, face_location):
    """Generate a unique cache filename based on file path and face location."""
    hash_input = f"{file_path}{face_location}".encode('utf-8')
    file_hash = hashlib.md5(hash_input).hexdigest()
    return os.path.join(CACHE_DIR, f"{file_hash}.png")

def save_to_cache(img, cache_filename):
    """Save the processed image to disk cache."""
    img.save(cache_filename, format='PNG')  # Save the processed image as PNG

def process_and_cache_images():
    """Read face embeddings, file paths, and locations, then cache the images."""
    
    # Load face data from .npy files
    face_embeddings = np.load('face_embeddings.npy')
    file_paths = np.load('face_file_paths.npy')
    face_locations = np.load('face_locations.npy')

    total_images = len(face_embeddings)  # Total number of images to be cached
    cached_files = 0  # Track how many files have been cached so far
    
    for i in range(total_images):
        file_path = file_paths[i]
        face_location = face_locations[i]

        # Generate cache filename
        cache_filename = get_cache_filename(file_path, face_location)
        
        # Add one to the count of processed files
        cached_files += 1

        # Check if already cached
        if not os.path.exists(cache_filename):
            try:
                # Load the image and process it
                img = Image.open(file_path)
                top, right, bottom, left = face_location
                img = img.crop((left, top, right, bottom))
                img = img.resize((150, 150), Image.LANCZOS)

                # Save to cache
                save_to_cache(img, cache_filename)
                print(f"Cached: {cache_filename} ({cached_files}/{total_images} files processed)")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    print(f"Caching complete: {cached_files}/{total_images} files cached.")

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
    
# Function to show similar faces and allow adding them to a cluster
def show_similar_faces(clustered_images, cluster_labels, current_cluster_index, root):
    def refresh_similar_faces():
        # Re-run the similarity calculation and update the UI with the new most similar images
        similar_window.destroy()
        show_similar_faces(clustered_images, cluster_labels, current_cluster_index, root)

    # Target cluster
    cluster_label = cluster_labels[current_cluster_index]
    target_embeddings = np.array([data['embedding'] for data in clustered_images[cluster_label]])
    target_name = f"Cluster {cluster_label}"

    # Collect data from other clusters
    other_data = []
    for label, images in clustered_images.items():
        if label != cluster_label:
            other_data.extend(images)

    if not other_data:
        messagebox.showinfo("Info", "No other faces to compare.")
        return

    # Compute embeddings of other faces
    other_embeddings = np.array([data['embedding'] for data in other_data])

    print("Step 1: Calculating cosine similarity...")

    # Perform similarity calculation and print progress
    similarity_matrix = cosine_similarity(target_embeddings, other_embeddings)
    
    print(f"Step 2: Calculating similarity matrix for {len(other_embeddings)} faces...")
    max_similarity_scores = np.max(similarity_matrix, axis=0)
    sorted_indices = np.argsort(-max_similarity_scores)

    sorted_data = [other_data[i] for i in sorted_indices]
    sorted_similarity_scores = max_similarity_scores[sorted_indices]

    top_N = 120

    print(f"Step 3: Sorting and displaying top {top_N} most similar faces...")

    # Create a new window to display similar faces
    similar_window = tk.Toplevel(root)
    similar_window.title(f"Faces Similar to {target_name}")
    similar_window.geometry("1000x700")

    # Scrollable frame setup
    def on_similar_frame_configure(event):
        similar_canvas.configure(scrollregion=similar_canvas.bbox("all"))

    similar_canvas = tk.Canvas(similar_window)
    similar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    similar_scrollbar = tk.Scrollbar(similar_window, orient="vertical", command=similar_canvas.yview)
    similar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    similar_canvas.configure(yscrollcommand=similar_scrollbar.set)

    similar_images_frame = ttk.Frame(similar_canvas)
    similar_canvas.create_window((0, 0), window=similar_images_frame, anchor='nw')
    similar_images_frame.bind("<Configure>", on_similar_frame_configure)

    similar_selected_images = []

    # Print the progress of loading and displaying images
    displayed_count = 0
    for idx, data in enumerate(sorted_data):
        if displayed_count >= top_N:
            break
        img_path = data['file_path']
        face_location = data['face_location']
        similarity_score = sorted_similarity_scores[idx]

        # New check to skip faces already in the target cluster
        if any(img_path == existing_data['file_path'] for existing_data in clustered_images[cluster_label]):
            continue
        photo = load_cached_image(img_path, face_location)

        # Create checkbox for each image
        var = tk.IntVar()
        checkbox = tk.Checkbutton(similar_images_frame, image=photo, variable=var)
        checkbox.image = photo  # Keep reference to avoid garbage collection
        checkbox.var = var
        checkbox.grid(row=idx//4, column=idx%4, padx=5, pady=5)
        similar_selected_images.append({'checkbox': checkbox, 'data': data})

        displayed_count += 1

    print(f"Step 4: Displaying top {top_N} images.")
    
    # Add selected images to the target cluster and refresh
    def add_selected_to_cluster_and_refresh():
        # Move selected images to the target cluster
        cluster_label = cluster_labels[current_cluster_index]
        to_move = []
        to_move_file_paths = set()
        for item in similar_selected_images:
            if item['checkbox'].var.get() == 1:
                to_move.append(item['data'])
                to_move_file_paths.add(item['data']['file_path'])
        # Remove selected items from current cluster
        clustered_images[cluster_label] = [
            data for data in clustered_images[cluster_label]
            if data['file_path'] not in to_move_file_paths
        ]

        # Remove selected items from their original clusters
        for data in to_move:
            for label, images in clustered_images.items():
                if label != cluster_label:
                    clustered_images[label] = [img for img in images if img['file_path'] != data['file_path']]
        # Add selected items to target cluster
        clustered_images[cluster_label].extend(to_move)

        # Ensure the unclustered label (-2) exists
        if -2 not in clustered_images:
            clustered_images[-2] = []  # Create the unclustered cluster if it doesn't exist

        # Add selected items to the unclustered cluster (-2)
        for data in to_move:
            data['label'] = -2  # Assign the unclustered label (-2) to the image
            clustered_images[-2].append(data)
        # Refresh the similar faces display
        print("Step 5: Images added to the cluster. Refreshing the similar faces window...")
        refresh_similar_faces()

    # Button to add selected images to the cluster and refresh
    add_button = ttk.Button(similar_window, text="Add Selected to Cluster", command=add_selected_to_cluster_and_refresh)
    add_button.pack(pady=10)

# Example function to call show_similar_faces
def reassign_selected_images_to_cluster(selected_images):
    # Add functionality to reassign selected images to a cluster
    pass
