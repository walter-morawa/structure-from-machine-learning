import cv2
from stitch_drone_images import stitch_drone_images  # Import your function
import os

def main():
    image_dir = "/app/sample-dataset/demo"  # test on demo first then fulldataset- dataset
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg'))]

    stitched_image = stitch_drone_images(image_paths)

    if stitched_image is not None:
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image stitching failed.")

if __name__ == "__main__":
    main()
