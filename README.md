Face Swap

This repository contains a Python script that utilizes OpenCV and dlib to perform face swapping between two images. By detecting facial landmarks and triangulating regions of interest, it warps and transposes facial features from one face onto another.

Prerequisites
Python 3.x
OpenCV (pip install opencv-python)
dlib (pip install dlib)

Usage
Place your source and target images in the project directory.
Ensure you have shape_predictor_68_face_landmarks.dat in the root of your project directory. You can obtain this file from dlib's official repository or other trusted sources.
Run the face_swap.py script: python face_swap.py

How It Works
The script performs the following steps:

Landmark Detection: Detect 68 facial landmarks on both source and target images using dlib.

Triangulation: Create triangles using these landmarks.

Warping: For each triangle in the source face, warp it to match the corresponding triangle in the target face.

Face Merging: Merge the warped source face onto the target image to achieve the face-swapped effect.

Disclaimer
The results may vary based on the input images' alignment, lighting, and quality. This is a basic implementation, and there are more advanced methods available for better and more realistic results.
