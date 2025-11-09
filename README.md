# 🚦 Smart Traffic Light Management System using Deep Learning

### **AI-Powered Adaptive Traffic Control with Ambulance Priority**

This project implements an **intelligent traffic light management system** that dynamically adjusts signal timings based on real-time traffic density and prioritizes **ambulance movement** using **audio-visual deep learning models**. The system runs on a **Flask web server**, integrates with **Raspberry Pi GPIO pins** for physical signal control, and combines both **siren detection** (audio) and **vehicle detection** (video).

---

## 🧠 **Key Features**

✅ **Siren Detection (Audio):**  
A custom **2D CNN model trained from scratch** detects emergency sirens from live microphone input using **TensorFlow Lite** for efficient on-device inference.  

✅ **Ambulance & Vehicle Detection (Video):**  
A **fine-tuned YOLOv8-nano model** detects and classifies vehicles (car, bus, truck, bike, ambulance, etc.) in real time using the connected camera.  

✅ **Dynamic Green Signal Timing:**  
A custom **Green Signal Time (GST)** formula calculates optimal green light durations based on the number and type of vehicles detected per lane.  

✅ **Emergency Response Mode:**  
If both a **siren** and an **ambulance** are detected, the system automatically activates a **yellow light** phase to clear the lane for emergency passage.  

✅ **Web Dashboard:**  
A **Flask-based interface** streams **live video**, **audio**, and **real-time stats** (siren status, vehicle counts, GST, and current light phase).  

✅ **Raspberry Pi GPIO Integration:**  
The red, yellow, and green LEDs (traffic lights) are controlled using GPIO pins **(17, 27, 22)**.

---

## ⚙️ **System Architecture**

```
 ┌────────────────────────┐
 │  Microphone Input      │
 │  (PyAudio)             │
 └──────────┬─────────────┘
            │
            ▼
     [2D CNN Siren Model]
        (TFLite Inference)
            │
            ▼
 ┌────────────────────────┐
 │  YOLOv8-Nano Detection │
 │  (Vehicles & Ambulance)│
 └──────────┬─────────────┘
            │
            ▼
    [GST Calculation Logic]
            │
            ▼
   [Raspberry Pi GPIO Lights]
            │
            ▼
    [Flask Web Dashboard]
```

---

## 🧩 **Technologies Used**

| Category | Technology |
|-----------|-------------|
| **Backend** | Flask, Python |
| **Deep Learning** | TensorFlow Lite, YOLOv8 (Ultralytics) |
| **Audio Processing** | PyAudio, Librosa |
| **Computer Vision** | OpenCV |
| **Hardware Control** | GPIOZero (for Raspberry Pi LEDs) |
| **Web Streaming** | Flask Response + FFmpeg (for MP3 streaming) |

---

## 🧰 **Hardware Requirements**

- Raspberry Pi 4   
- USB Camera / PiCam  
- Microphone  
- 3 LEDs (Red, Yellow, Green)  
- 3 × 220Ω resistors  
- Jumper wires and breadboard  

**GPIO Pin Mapping:**
| LED Color | GPIO Pin |
|------------|-----------|
| Red        | 17 |
| Yellow     | 27 |
| Green      | 22 |

---

## 📦 **Installation Guide**

### 1. Clone the repository
```bash
git clone https://github.com/RonitPG19/Smart-Traffic-System-AIoT.git
cd Smart-Traffic-System-AIoT
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> 💡 *Ensure you have TensorFlow Lite, Ultralytics, OpenCV, and GPIOZero installed.*

### 3. Add model files
Place your trained models in the root directory:
```
siren_roadnoise_model.tflite
yolo_v8n_trained.pt
```

### 4. Run the Flask server
```bash
python app.py
```

### 5. Access the dashboard
Open your browser and visit:
```
http://<raspberry-pi-ip>:5000
```

---

## 📊 **Web Dashboard Preview**

**Dashboard Features:**
- Live camera feed with bounding boxes  
- Real-time audio siren stream  
- Vehicle counts by class  
- Adaptive green signal duration  
- Live traffic light phase and countdown  

---

## 🎯 **Main Objectives**

- Reduce traffic congestion using adaptive signal timing.  
- Ensure immediate lane clearance for emergency vehicles.  
- Integrate multimodal AI (audio + vision) for real-time decision-making.  
- Create a low-cost, scalable solution for **Smart City Infrastructure**.  

---

## 🧪 **Future Enhancements**

- Integration with **cloud-based traffic analytics dashboards**  
- Support for **multi-intersection coordination**  
- Integration with **IoT sensors** for environmental monitoring**  
- Use of **edge AI hardware accelerators** (e.g., Coral TPU, NVIDIA Jetson)  

---

## 📁 **Repository Structure**

```
📦 smart-traffic-light-system
 ┣ 📜 app.py                     # Main Flask server
 ┣ 📜 requirements.txt           # Dependencies
 ┣ 📜 README.md                  # Project documentation
 ┣ 📜 siren_roadnoise_model.tflite  # Audio CNN model
 ┗ 📜 yolo_v8n_trained.pt        # YOLOv8-nano vehicle detection model
```

---

## 👨‍💻 **Contributors**

- Ronit Girglani
- Kenil Patel
- Vedant Patel
- Kirtan Visnagara  
---

## 🧾 **License**
This project is licensed under the **MIT License** — feel free to use, modify, and distribute with attribution.
