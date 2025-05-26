# Face Recognition System with GPIO Control for Raspberry Pi

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.x-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi-red.svg)


![PLAKAT](https://github.com/user-attachments/assets/7e7bbd60-3c99-4b7f-8256-e869e4b43746)



A lightweight face recognition system for Raspberry Pi that automatically detects faces via camera, compares them against a reference database, and controls an LED indicator to signal access permission.

<p align="center">
  <img src="/api/placeholder/640/320" alt="Face Recognition System" />
</p>

## ✨ Features

- 🔍 Real-time face detection using OpenCV
- 🧠 Advanced face comparison using SIFT algorithm and histogram analysis
- 💡 GPIO integration for LED status indicator
- 🔄 Multi-threaded design for responsive performance
- 💾 Automatic face capture and storage
- 🏃‍♂️ Headless operation (no GUI required)

## 📋 Requirements

### Hardware
- Raspberry Pi (any model with GPIO support)
- Camera compatible with Raspberry Pi
- LED
- 220-330Ω Resistor
- Jumper wires

### Software
- Python 3.x
- OpenCV (cv2)
- NumPy
- RPi.GPIO
- Cascade classifier file (`cascade.xml`)

## 📁 File Structure

```
/
├── faceCut.py            # Main executable file
├── cascade.xml           # Classifier file for face detection
├── img/                  # Folder with reference images for comparison
│   ├── person1.jpg
│   ├── person2.jpg
│   └── ...
└── captured/             # Folder for saving detected faces (created automatically)
    ├── face_20250517_123045.jpg
    └── ...
```

## 🔌 Hardware Setup

1. Connect the LED to your Raspberry Pi:
   - Connect the anode (longer leg) of the LED to GPIO16 through a resistor
   - Connect the cathode (shorter leg) to any GND pin

![image](https://github.com/user-attachments/assets/c1502d1c-437f-4197-8731-e8ec732e1b33)


## ⚙️ Installation

### Basic Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/face-recognition-gpio.git
   cd face-recognition-gpio
   ```

2. Install basic dependencies:
   ```bash
   pip install opencv-python numpy RPi.GPIO
   ```

3. Create an `img` folder and add reference face images:
   ```bash
   mkdir -p img
   # Copy your reference images to the img folder
   ```

4. Download a cascade classifier file:
   ```bash
   # OpenCV's default face detector
   wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml -O cascade.xml
   ```

### Detailed Installation Instructions

For optimal performance and to resolve all dependencies correctly, follow these comprehensive installation steps:

**1. System Preparation**
```bash
# Update system packages  
sudo apt update && sudo apt upgrade -y  

# Install system dependencies  
sudo apt install -y \  
    build-essential cmake pkg-config \  
    libopenblas-dev liblapack-dev \  
    python3-opencv python3-numpy python3-scipy \  
    libx11-dev libatlas-base-dev \  
    libgtk-3-dev libboost-python-dev  
```

**2. Create a Virtual Environment**
```bash
# Navigate to the working directory  
cd ~/python3  

# Create and activate a virtual environment  
python3 -m venv myenv  
source myenv/bin/activate  
```

**3. Install Python Libraries**
```bash
# Install dlib (may take a while on Raspberry Pi)  
pip install dlib --no-cache-dir  

# Install face_recognition without automatic dependencies  
pip install face-recognition-models  
pip install face_recognition --no-deps  

# Install OpenCV with NumPy 1.x (required for compatibility)  
pip install opencv-python --index-url=https://www.piwheels.org/simple/  
pip uninstall -y numpy  
pip install numpy==1.26.4  # Last stable NumPy 1.x version  

# Install missing dependencies  
pip install Click>=6.0 Pillow  
```

**4. Verify Installation**
```bash
# Test the facial recognition script  
python3 faceCut.py  
```

> **Note:** Installing dlib on Raspberry Pi can take 1-2 hours depending on your Pi model. For Raspberry Pi 3 or earlier models, you may need to increase swap space temporarily to complete the compilation.

## 🚀 Usage

Run the script with superuser privileges (required for GPIO access):

```bash
sudo python3 faceCut.py
```

The system will start in automatic scanning mode. To stop, press `Ctrl+C`.

## 🧠 How It Works

### Face Detection Process
1. The system captures frames from the camera
2. Each frame is processed to detect faces using a cascade classifier
3. When a face is detected, it's saved to the `captured` folder
4. The face is then compared with reference images

### Face Comparison Algorithm
The system uses a combined approach for face comparison:

1. **SIFT Features (70% weight)**
   - Extracts and matches key points between images
   - Uses Lowe's ratio test to filter quality matches

2. **Histogram Analysis (30% weight)**
   - Computes and compares intensity histograms
   - Uses correlation method for similarity calculation

3. **Access Decision**
   - Combined score above 30% grants access
   - LED indicator provides visual feedback

### LED Signals
- ✅ **Access Granted**: LED turns on for 5 seconds
- ❌ **Access Denied**: LED blinks 3 times

## ⚙️ Configuration

You can adjust the following parameters in the code:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LED_PIN` | 16 | GPIO pin for LED connection |
| `combined_score > 30` | 30% | Threshold for granting access |
| `scaleFactor` | 1.1 | Face detection scaling parameter |
| `minNeighbors` | 5 | Face detection quality parameter |

## 🛠️ Advanced Configuration

### For Better Recognition Accuracy

```python
# Increase minNeighbors for fewer false positives
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1, 
    minNeighbors=6,  # Default: 5
    minSize=(30, 30)
)

# Adjust feature weights
combined_score = 0.8 * match_score1 + 0.2 * hist_score  # Default: 0.7/0.3

# Increase threshold for stricter matching
if combined_score > 40:  # Default: 30
    match_found = True
```

### For Better Performance

```python
# Lower resolution for faster processing
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Default: 640
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Default: 480

# Increase scaleFactor for faster detection
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.2,  # Default: 1.1
    minNeighbors=5,
    minSize=(30, 30)
)
```

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Camera not detected | Check connections and run `vcgencmd get_camera` |
| No faces detected | Improve lighting and check cascade file |
| False recognitions | Add more reference images, adjust threshold |
| LED not working | Check wiring and GPIO pin number |
| Permission errors | Make sure to run with `sudo` |

## 🔐 Security Considerations

This system is intended for educational and demonstration purposes. For production use, consider:

- Implementing data encryption
- Using more robust face recognition algorithms
- Adding liveness detection to prevent photo attacks
- Implementing proper user management
- Adding event logging
- Increasing the security of the physical setup


## 📜 License

MIT License

Copyright (c) [2025] [cortaget]

Permission is hereby granted, free of charge, to any person obtaining a copy


⭐️ If you found this useful, please star the repository!
