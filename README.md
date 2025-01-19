Documentation for Face Image Processing and Comparison Project
Description

This project is designed for preprocessing images, including resizing, blurring, converting to grayscale, histogram equalization, and then comparing faces in images using the DeepFace library.

The project includes the following features:

    Image Preprocessing: Resizing, blurring, converting to grayscale, and histogram equalization.
    Face Comparison: Using the Facenet model from the DeepFace library to compare a target image with other images.
    Multiprocessing: Multiprocessing is used to speed up the comparison process.
    Saving Processed Images: Processed images are saved in a specified folder for further use.

Key Functions
1. preprocess_image(image_path, output_folder)

Preprocessing function for an image.
Description:

    Loads the image from the given path.
    Applies the following preprocessing steps:
        Resizes the image to 224x224 pixels.
        Applies Gaussian blur.
        Converts the image to grayscale.
        Applies histogram equalization.
    Saves the processed image to the specified folder.

Arguments:

    image_path (str): The path to the source image.
    output_folder (str): The folder to save the processed images.

Returns:

    The path to the saved processed image if successful.
    None if the image could not be loaded.

2. compare_faces(target_image_path, compare_image_path)

Function to compare two images.
Description:

    Uses the DeepFace model to compare two images.
    Compares the target image with another image using the DeepFace.verify method.
    Returns the results: whether the faces match and the distance between them.

Arguments:

    target_image_path (str): The path to the target image.
    compare_image_path (str): The path to the image to compare.

Returns:

    A tuple containing:
        The path to the compared image.
        A boolean value True if the faces match, or False otherwise.
        The distance between the images (if computed).

3. batch_compare(target_image_path, folder_path)

Function to compare a target image with every image in a specified folder.
Description:

    Loads all images from the specified folder.
    Compares each image in the folder with the target image using compare_faces.
    Uses multiprocessing to speed up the comparison process.

Arguments:

    target_image_path (str): The path to the target image.
    folder_path (str): The path to the folder containing images for comparison.

Returns:

    A list of comparison results for each image. Each list element is a tuple: (image path, match, distance).

4. process_folder_images(input_folder, output_folder)

Function to preprocess all images in a folder.
Description:

    Applies preprocessing to all images in the specified folder.
    Saves the processed images to another folder.

Arguments:

    input_folder (str): The folder containing source images.
    output_folder (str): The folder to save the processed images.

Returns:

    A list of paths to the processed images.

Example Usage

if __name__ == "__main__":
    # Specify the target image path and the folder for comparison
    target_image_path = "face_5.jpg"
    folder_path = "img"  # Folder containing images for comparison
    processed_folder = "processed_images"  # Folder to save processed images

    # Check for the target image and folder for comparison
    if not os.path.exists(target_image_path):
        print("Error: Target image not found.")
        exit()

    if not os.path.isdir(folder_path):
        print("Error: Folder for comparison not found.")
        exit()

    # Process all images in the folder and save them
    processed_images = process_folder_images(folder_path, processed_folder)

    # Compare the target image with the processed images
    comparison_results = batch_compare(target_image_path, processed_folder)

    # Print the comparison results
    print("Comparison results:")
    for img_path, verified, distance in comparison_results:
        print(f"Image: {img_path} | Match: {verified} | Distance: {distance}")

Requirements

    Python 3.x
    Libraries:
        DeepFace
        cv2 (OpenCV)
        concurrent.futures
        os
        datetime
