import cv2
import opensfm
import numpy as np
import os
import time  # Import time for time tracking


def stitch_drone_images(image_paths, max_width=800):
    start_time = time.time()
    print("Starting image stitching process...")
    # 1. Feature Detection and Matching (OpenCV)
    print("Detecting and matching features...")  # Add print statement
    keypoints, descriptors = [], []
    for i, path in enumerate(image_paths):
        image_start_time = time.time()
        print(f"Processing image {i+1}/{len(image_paths)}: {path}")  
        img = cv2.imread(path)
        height = int(img.shape[0] * max_width / img.shape[1]) if max_width else img.shape[0]
        img = cv2.resize(img, (max_width, height)) if max_width else img

        print("  Feature Detection...")
        kp, des = cv2.SIFT_create().detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

        image_end_time = time.time()
        print(f"  Feature detection for image {i+1} completed in {:.2f} seconds.".format(image_end_time - image_start_time))

    matches = find_matches(descriptors)  # Use OpenCV's BFMatcher or FLANN

    # 2. Pairwise Homography Estimation (OpenCV)
    print("Estimating pairwise homographies...")  # Add print statement
    homographies = estimate_homographies(keypoints, matches)
    end_time = time.time()
    print("Pairwise homographies completed in {:.2f} seconds.".format(end_time - start_time)) #add print statement


    # 3. OpenSfM Integration
    print("Performing bundle adjustment with OpenSfM...")  # Add print statement
    reconstruction = create_opensfm_reconstruction(image_paths, keypoints, matches, homographies)
    opensfm.reconstruction.bundle(reconstruction)  # Perform bundle adjustment
    end_time = time.time()
    print("bundle adjustment completed in {:.2f} seconds.".format(end_time - start_time)) #add print statement

    # 4. Extract Optimized Homographies and Warp (OpenCV)
    print("Warping and blending images...")  # Add print statement
    optimized_homographies = extract_homographies_from_opensfm(reconstruction)
    stitched_image = warp_and_blend(image_paths, optimized_homographies)
    end_time = time.time()
    print("Image stitching completed in {:.2f} seconds.".format(end_time - start_time)) #add print statement

    return stitched_image

# ... (Helper functions for feature matching, homography estimation, 
#      OpenSfM interaction, warping, and blending)
