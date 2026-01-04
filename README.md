# K-Means Implementation from Scratch

This project implements the **K-Means clustering algorithm** in Python without relying on external machine learning libraries. The notebook allows the user to choose the number of clusters (**k**) and provides flexible options for centroid initialization and distance measurement.

## Overview

The algorithm is designed to cluster data points by:
- **Initializing centroids** using one of two methods:
  - **Random Initialization:** Randomly select k data points as the starting centroids.
  - **Deterministic Initialization:** Divide the dataset into k segments and choose a "critical" point (such as the median or a central value) from each segment as the initial centroid.
- **Calculating distances** between data points and centroids using:
  - **Euclidean Distance**
  - **Manhattan Distance**

This approach enables a clear understanding of how the K-Means algorithm works under the hood and allows experimentation with different initialization methods and distance metrics.

## Features

- **Customizable k-value:** The user can set the number of clusters.
- **Flexible centroid initialization:**
  - Random selection.
  - Deterministic selection based on data segmentation.
- **Distance metrics:** Option to compute both Euclidean and Manhattan distances.
- **Educational tool:** Ideal for learning and understanding the mechanics behind K-Means clustering without abstracting away the details through libraries.

## Requirements

- **Python 3.x**
- **Jupyter Notebook** (or any other environment that can run `.ipynb` files)
- Python packages (if used within the notebook):
  - `numpy`
  - `pandas`
  - `matplotlib` (optional, for visualization purposes)

## Project Structure
```text
.
├── notebooks/
│   └── 01_kmeans_from_scratch.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

## Getting Started

### 1. Clone the Repository

Clone the repository to your local machine using:

```bash
git clone https://github.com/Amr-Belal-77/KMeans-from-scratch.git
