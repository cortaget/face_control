Face Detection Project using OpenCV

This project implements real-time face detection using the OpenCV library. It captures video from a webcam, recognizes faces, and saves their images to a local folder.
Installation

    Ensure you have Python and pip installed.

    Install the required libraries by running the following command:

    bash

    pip install opencv-python

    Download the Haar Cascade classifier file for face detection (cascade.xml) and place it in the root folder of the project.

Running the Project

To run the project, execute the face detection script:

bash

python face_detection.py

Description

The project consists of two main classes:

    face_detection.py:
        Loads the Haar Cascade classifier and starts video capture from the webcam.
        Processes each frame, converts it to grayscale, and detects faces.
        For each detected face, it draws a rectangle and saves the face image in the img folder.

    manage_images.py:
        Deletes all files and subfolders in the img folder if it exists and creates a new empty folder.
        This allows for clearing old images before a new capture.

Usage

    After running the face_detection.py script, the application will display video from your webcam.
    Each detected face will be highlighted with a rectangle, and its image will be saved in the img folder.
    To exit the application, press the q key.

License

This project is open and free to use. You may use and modify it at your discretion.
