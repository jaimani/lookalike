Editor's Note: This project and README was created entirely via prompts to ChatGPT and Claude. I have edited at most 5 lines of code.

# Face Clustering and Management Tool

Find out who someone looks like

## Overview

This project provides a toolkit for clustering and managing faces in a photo library. It utilizes face detection and clustering techniques to help users organize their photos by identifying similar faces. The graphical user interface (GUI) enables manual review, merging of clusters, and enhanced organization. This toolkit can be used by photographers, archivists, or anyone looking to manage large collections of photos with identifiable faces.

### Key Features

- Face detection and embedding generation using `face_recognition`.
- Clustering of face embeddings using DBSCAN.
- GUI-based manual review and management of clusters, built with Tkinter.
- Efficient caching and reloading of previously computed face embeddings to save time.

## Project Structure

- `cluster_management_tool.py`: The main script that allows users to select a photo directory, detect faces, cluster them, and manage clusters through a GUI.
- `face_utils.py`: Contains utility functions for image processing, caching, and displaying similar faces for clustering.
- `requirements.txt`: Lists all dependencies required to run the project.

## Installation

### Prerequisites

- Python 3.7 or higher
- `pip` for managing Python packages

### Dependencies

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

Dependencies include:

- `numpy`
- `face_recognition`
- `scikit-learn`
- `Pillow`
- `tkinter` (pre-installed with Python)

## Usage

1. **Run the Tool**:

   ```sh
   python cluster_management_tool.py
   ```

2. **Select Photo Library Directory**: When prompted, select the directory containing your images.

3. **Face Detection and Clustering**: The tool will automatically detect faces, create embeddings, and cluster similar faces.

4. **Manual Review**: Use the GUI to manually review clusters and merge similar faces as needed.

### GUI Overview

- The GUI allows you to browse different face clusters, view individual faces, and manage clusters by merging similar ones.

## Contribution

Contributions are welcome! If you'd like to contribute, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- `face_recognition` library by Adam Geitgey for providing the core face detection and embedding capabilities.
- `scikit-learn` for clustering utilities.
