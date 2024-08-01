import cv2
import sys
print(sys.path)
import opensfm
import numpy as np
import os
import time  # Import time for time tracking

def find_matches(descriptors):
    """
    Finds matches between descriptors using OpenCV's BFMatcher.

    Args:
        descriptors: A list of descriptor arrays for each image.

    Returns:
        A list of lists of DMatch objects representing the matches between images.
    """

    bf = cv2.BFMatcher()
    matches = []
    for i in range(len(descriptors) - 1):
        matches_i = bf.knnMatch(descriptors[i], descriptors[i + 1], k=2)
        good_matches_i = []
        for m, n in matches_i:
            if m.distance < 0.75 * n.distance:  # Apply ratio test for better matches
                good_matches_i.append(m)
        matches.append(good_matches_i)
    return matches

def estimate_homographies(keypoints, matches):
    """
    Estimates pairwise homographies between images based on keypoint matches.

    Args:
        keypoints: A list of keypoint lists for each image.
        matches: A list of lists of DMatch objects representing the matches between images.

    Returns:
        A list of homography matrices (3x3 numpy arrays).
    """

    homographies = []
    for i in range(len(matches)):
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches[i]]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i + 1][m.trainIdx].pt for m in matches[i]]).reshape(-1, 1, 2)

        # Use RANSAC for robust homography estimation
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append(M)

    return homographies

def create_opensfm_reconstruction(image_paths, keypoints, matches, homographies):
    """
    Creates an OpenSfM reconstruction object and populates it with image data,
    features, matches, and initial homographies.

    Args:
        image_paths: A list of paths to the images.
        keypoints: A list of keypoint lists for each image.
        matches: A list of lists of DMatch objects representing the matches between images.
        homographies: A list of homography matrices (3x3 numpy arrays).

    Returns:
        An OpenSfM reconstruction object ready for bundle adjustment.
    """

    # Create a new reconstruction object
    reconstruction = opensfm.types.Reconstruction()

    # Add cameras (assuming a simple perspective camera model for now)
    for i in range(len(image_paths)):
        camera = opensfm.types.PerspectiveCamera()
        camera.id = f"camera_{i}"
        camera.width = keypoints[i][0].pt[0] * 2 # Assuming keypoints are centered
        camera.height = keypoints[i][0].pt[1] * 2
        reconstruction.add_camera(camera)

    # Add shots (images)
    for i, path in enumerate(image_paths):
        shot = opensfm.types.Shot()
        shot.id = f"shot_{i}"
        shot.camera = reconstruction.cameras[f"camera_{i}"]
        shot.pose = opensfm.types.Pose()  # Initialize with identity pose
        if i > 0:  # Set initial pose based on homographies (except for the first image)
            shot.pose.set_from_matrix(homographies[i - 1])
        reconstruction.add_shot(shot)

    # Add points and observations (based on matches)
    next_point_id = 0
    for i in range(len(matches)):
        for match in matches[i]:
            # Create a new 3D point if it doesn't exist yet
            if match.queryIdx not in reconstruction.points:
                point = opensfm.types.Point()
                point.id = next_point_id
                point.coordinates = np.zeros(3)  # Initialize with dummy coordinates
                reconstruction.add_point(point)
                next_point_id += 1

            # Add observations (2D projections) of the point in both images
            observation1 = opensfm.types.Observation()
            observation1.id = f"shot_{i}"
            observation1.point = match.queryIdx
            observation1.projection = keypoints[i][match.queryIdx].pt

            observation2 = opensfm.types.Observation()
            observation2.id = f"shot_{i + 1}"
            observation2.point = match.queryIdx
            observation2.projection = keypoints[i + 1][match.trainIdx].pt

            reconstruction.add_observation(observation1)
            reconstruction.add_observation(observation2)

    return reconstruction


def extract_homographies_from_opensfm(reconstruction):
    """
    Extracts the optimized homography matrices from an OpenSfM reconstruction.

    Args:
        reconstruction: An OpenSfM reconstruction object after bundle adjustment.

    Returns:
        A list of homography matrices (3x3 numpy arrays).
    """

    # Reference (anchor) image/shot
    reference_shot_id = reconstruction.reference.shot_id  # Assuming you have a reference shot set

    homographies = []
    for shot in reconstruction.shots.values():
        if shot.id == reference_shot_id:
            homographies.append(np.identity(3))  # Identity homography for the reference image
        else:
            # Compute homography that warps this shot into the reference shot's coordinate system
            relative_pose = shot.pose.relative_to(reconstruction.shots[reference_shot_id].pose)
            H = relative_pose.get_homography_matrix() 
            homographies.append(H)

    return homographies

def warp_and_blend(image_paths, homographies):
    """
    Warps and blends images based on the provided homographies.

    Args:
        image_paths: A list of paths to the original (full-resolution) images.
        homographies: A list of homography matrices (3x3 numpy arrays).

    Returns:
        The stitched panorama image.
    """

    # Load original (full-resolution) images
    images = [cv2.imread(path) for path in image_paths]

    # Find the size of the output panorama
    corners = find_corners(images, homographies)
    output_shape = calculate_output_shape(corners)

    # Warp images to the output panorama
    warped_images = []
    for img, H in zip(images, homographies):
        warped = cv2.warpPerspective(img, H, output_shape)
        warped_images.append(warped)

    # Blend warped images (you might need to experiment with different blending techniques)
    stitched_image = blend_images(warped_images)

    return stitched_image

def find_corners(images, homographies):
    """
    Finds the corners of each warped image in the output panorama coordinate system.
    """
    corners = []
    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        img_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(img_corners, H)
        corners.append(warped_corners)
    return corners

def calculate_output_shape(corners):
    """
    Calculates the size of the output panorama based on the warped image corners.
    """
    all_corners = np.concatenate(corners, axis=0)
    min_x, min_y = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    max_x, max_y = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]]) 

    # Adjust homographies to account for the translation
    adjusted_homographies = [translation @ H for H in homographies]

    return (max_x - min_x, max_y - min_y), adjusted_homographies

def blend_images(warped_images):
    """
    Blends the warped images using a simple weighted averaging technique.
    You might want to explore more advanced blending methods for better results.
    """
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(warped_images)  
    if status != cv2.Stitcher_OK:
        print("Error during image stitching:", status)
        return None
    return stitched


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
        print(f"  Feature detection for image {i+1} completed in {image_end_time - image_start_time:.2f} seconds.") 

    matches = find_matches(descriptors)  # Use OpenCV's BFMatcher or FLANN

    # 2. Pairwise Homography Estimation (OpenCV)
    print("Estimating pairwise homographies...")  # Add print statement
    homographies = estimate_homographies(keypoints, matches)
    end_time = time.time()
    print(f"Pairwise homographies completed in {end_time - start_time:.2f} seconds.")


    # 3. OpenSfM Integration
    print("Performing bundle adjustment with OpenSfM...")  # Add print statement
    reconstruction = create_opensfm_reconstruction(image_paths, keypoints, matches, homographies)
    opensfm.reconstruction.bundle(reconstruction)  # Perform bundle adjustment
    end_time = time.time()
    print(f"Bundle adjustment completed in {end_time - start_time:.2f} seconds.")

    # 4. Extract Optimized Homographies and Warp (OpenCV)
    print("Warping and blending images...")  # Add print statement
    optimized_homographies = extract_homographies_from_opensfm(reconstruction)
    stitched_image = warp_and_blend(image_paths, optimized_homographies)
    end_time = time.time()
    print(f"Image stitching completed in {end_time - start_time:.2f} seconds.") 

    return stitched_image

# ... (Helper functions for feature matching, homography estimation, 
#      OpenSfM interaction, warping, and blending)
