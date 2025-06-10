# 😷 Real-Time Face Mask Detector

A real-time system built with Python to detect whether a person is wearing a face mask. The application uses a deep learning model on a live webcam feed and triggers an audible alert if a person is not wearing a mask.

## Demo
*(Here you can add a GIF or a screenshot of your project in action!)*

![Demo GIF](link-to-your-gif.gif)

## Features
- **Real-Time Detection:** Identifies masks on live video streams from a webcam.
- **Deep Learning Model:** Uses a fine-tuned MobileNetV2 model for high accuracy.
- **Live Sound Alert:** Provides an audible 'beep' when a person without a mask is detected.
- **Visual Feedback:** Draws bounding boxes around faces, colored green for "Mask" and red for "No Mask".

## Technology Stack
- **Language:** Python
- **Libraries:**
    - **TensorFlow / Keras:** For building and training the CNN model.
    - **OpenCV:** For video capture and image processing.
    - **Scikit-learn:** For splitting the dataset.
    - **Playsound:** For the alert system.
    - **Matplotlib:** For plotting training history.

## Installation & Usage

Follow these steps to set up and run the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Face-Mask-Detector.git](https://github.com/YourUsername/Face-Mask-Detector.git)
    cd Face-Mask-Detector
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: We will create the requirements.txt file next)*

4.  **Download the Dataset:**
    Download the face mask dataset from [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) and place the contents into a `dataset` folder in the project root.

5.  **Train the Model (Optional):**
    To train the model from scratch, run the training script. This will generate `face_mask_detector.h5` and `plot.png`.
    ```bash
    python train_model.py
    ```

6.  **Run the Detector:**
    Execute the main script to start the live detection from your webcam.
    ```bash
    python detect_mask_video.py
    ```
    Press 'q' to exit the application.
