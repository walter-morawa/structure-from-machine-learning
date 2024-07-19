# SfML
Machine Learning for Structure from Motion

# Goal
The purpose of this repository is to create an open source library for using machine learning for feature detection, structure from motion, and dense point cloud creation.

- We want to create an open source Quick Stitch library for aerial data. Aerial data is typically sequential, so this vastly simplifies the process of image stitching. Quick stitch is for 2 dimensions, as opposed to a point cloud or mesh which takes much longer to process. Existing options from OpenCV rarely work on more than a few images.

- We want to apply supervised and unsupervised learning to feature detection. There are many great feature detectors such as SIFT, ORB, KAZE and others. We want to apply ML techniques to see if we can create a faster and/or higher quality feature detector using existing detectors for training.

- We want to apply machine learning to the entire structure from motion process, from feature detection, bundle adjustment, alignement, point cloud creation and even meshing.
