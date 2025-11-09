from flask import Flask, Response, stream_with_context, jsonify
import cv2, pyaudio, threading, time, subprocess, os, signal
from gpiozero import LED
from queue import Queue
from collections import deque
import numpy as np, librosa, tensorflow as tf
from ultralytics import YOLO

# -----------------------
# Config
# -----------------------
MODEL_PATH = "siren_roadnoise_model.tflite"
YOLO_MODEL_PATH = "yolo_v8n_trained.pt"
NO_OF_LANES = 2  # 🔹 adjust for your road

RESULT_SR = 48000
MODEL_DURATION = 5
N_MELS = 64
CONF_THRESHOLD = 0.5
FFMPEG_BITRATE = "128k"
MP3_READ_CHUNK = 1024
INFER_INTERVAL = 1.0
BUFFER_SECONDS = 8
light_state = {"phase": "Red", "countdown": 0}
light_lock = threading.Lock()
emergency_state = {"active": False}
emergency_lock = threading.Lock()


# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

# -----------------------
# GPIO Traffic Lights
# -----------------------
red, yellow, green = LED(17), LED(27), LED(22)

def smart_traffic_controller():
    global gst_state, light_state

    while True:
        # Wait if emergency blinking is active
        with emergency_lock:
            if emergency_state["active"]:
                time.sleep(1)
                continue

        # 🔴 RED PHASE
        red.on(); yellow.off(); green.off()
        with emergency_lock:
            if emergency_triggered:
                time.sleep(1)
                continue

        with light_lock:
            light_state.update({"phase": "Red", "countdown": 20 })
        print("🔴 Red light ON — capturing vehicles for GST...")

        for i in range(5, 0, -1):
            with emergency_lock:
                if emergency_state["active"]:
                    break
            with light_lock:
                light_state["countdown"] = i
            time.sleep(1)

        with emergency_lock:
            if emergency_state["active"]:
                continue  # Skip GST & move to next cycle safely

        # 🧮 Calculate GST ONLY after Red
        with vehicle_lock:
            counts = dict(vehicle_state["counts"])
        with gst_lock:
            if counts and sum(counts.values()) > 0:
                gst_state["gst"] = calculate_gst(counts, NO_OF_LANES)
            else:
                gst_state["gst"] = 0.0

        gst_val = gst_state["gst"]
        print(f"🟢 Next Green light duration: {gst_val} seconds")

        # 🟢 GREEN PHASE
        red.off(); yellow.off(); green.on()
        with emergency_lock:
            if emergency_triggered:
                time.sleep(1)
                continue
        with light_lock:
            light_state.update({"phase": "Green", "countdown": int(gst_val if gst_val > 0 else 5)})
        print(f"🟢 Green light ON for {gst_val} seconds")

        for i in range(light_state["countdown"], 0, -1):
            with emergency_lock:
                if emergency_state["active"]:
                    break
            with light_lock:
                light_state["countdown"] = i
            time.sleep(1)

        with emergency_lock:
            if emergency_state["active"]:
                continue

        # 🟡 YELLOW PHASE
        green.off(); yellow.on()
        with emergency_lock:
            if emergency_triggered:
                time.sleep(1)
                continue
        with light_lock:
            light_state.update({"phase": "Yellow", "countdown": 5})
        print("🟡 Yellow light ON for 5 seconds")

        for i in range(5, 0, -1):
            with emergency_lock:
                if emergency_state["active"]:
                    break
            with light_lock:
                light_state["countdown"] = i
            time.sleep(1)


threading.Thread(target=smart_traffic_controller, daemon=True).start()

def detect_ambulance_and_siren(frame):
    """
    Uses YOLO to detect ambulance and checks for bright red/blue flashing lights in the frame.
    Returns True if both are detected.
    """
    global yolo_model
    if not yolo_model:
        print("⚠️ YOLO model not loaded.")
        return False

    results = yolo_model.predict(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    ambulance_detected = False
    for box in boxes:
        cls_name = names[int(box.cls[0])].lower()
        if "ambulance" in cls_name:
            ambulance_detected = True
            break

    if not ambulance_detected:
        return False

    # --- Detect flashing siren-like regions ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 100, 70), (140, 255, 255))

    red_area = cv2.countNonZero(red_mask)
    blue_area = cv2.countNonZero(blue_mask)

    # Siren light if both red and blue occupy significant area (threshold can be tuned)
    siren_light_detected = (red_area > 1000 and blue_area > 1000)

    return ambulance_detected and siren_light_detected


def check_for_ambulance():
    """Run YOLO to check for ambulance when siren is detected."""
    # Prevent re-entry if already blinking
    with emergency_lock:
        if emergency_state["active"]:
            return

    if not yolo_model:
        print("⚠️ YOLO model not loaded, cannot detect ambulance.")
        return

    print("🚑 Siren detected — scanning for ambulance...")

    # Grab one frame from camera
    ret, frame = camera.read()
    if not ret:
        print("❌ Could not read frame for ambulance detection.")
        return

    results = yolo_model.predict(frame, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    found_ambulance = False
    for box in boxes:
        cls_name = names[int(box.cls[0])].lower()
        if "ambulance" in cls_name:
            found_ambulance = True
            break

    if found_ambulance:
        print("🚨 Ambulance detected — entering emergency mode!")
        threading.Thread(target=emergency_blink_lights, args=(10,), daemon=True).start()
    else:
        print("✅ No ambulance detected, continuing normal operation.")

def trigger_emergency_yellow(duration=10):
    global emergency_triggered
    with emergency_lock:
        if emergency_triggered:
            return
        emergency_triggered = True

    print("⚠️ Emergency Yellow Activated for Ambulance")

    red.off(); green.off(); yellow.on()

    for i in range(duration, 0, -1):
        with light_lock:
            light_state.update({"phase": "Emergency Yellow", "countdown": i})
        time.sleep(1)

    yellow.off()
    red.on()
    print("✅ Emergency yellow ended — resuming normal operation.")

    with emergency_lock:
        emergency_triggered = False


# -----------------------
# Camera
# -----------------------
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 25)
if not camera.isOpened():
    raise RuntimeError("❌ Could not open camera")
print("✅ Camera opened")

# -----------------------
# Microphone
# -----------------------
audio = pyaudio.PyAudio()
SUPPORTED_RATES = [48000, 44100, 32000, 22050, 16000]
mic = None
MIC_RATE = None
for rate in SUPPORTED_RATES:
    try:
        mic = audio.open(format=pyaudio.paInt16, channels=1, rate=rate,
                         input=True, frames_per_buffer=1024)
        MIC_RATE = rate
        print(f"✅ Microphone opened at {rate} Hz")
        break
    except Exception:
        continue
if mic is None:
    raise RuntimeError("❌ No working mic found")

audio_queue = Queue(maxsize=200)
buffer_capacity = int(BUFFER_SECONDS * MIC_RATE)
rolling_buffer = deque(maxlen=buffer_capacity)
buffer_lock = threading.Lock()

def audio_capture_thread():
    while True:
        try:
            data = mic.read(1024, exception_on_overflow=False)
        except Exception as e:
            print("Mic read error:", e); continue
        if not audio_queue.full():
            audio_queue.put(data)
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        with buffer_lock:
            rolling_buffer.extend(arr)
threading.Thread(target=audio_capture_thread, daemon=True).start()

# -----------------------
# FFmpeg Audio Stream
# -----------------------
def generate_audio_mp3():
    cmd = ['ffmpeg', '-f', 's16le', '-ar', str(MIC_RATE), '-ac', '1', '-i', 'pipe:0',
           '-acodec', 'libmp3lame', '-b:a', FFMPEG_BITRATE, '-f', 'mp3', 'pipe:1']
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    def feed():
        while True:
            chunk = audio_queue.get()
            if process.poll() is not None: break
            try: process.stdin.write(chunk)
            except BrokenPipeError: break
    threading.Thread(target=feed, daemon=True).start()
    while True:
        out = process.stdout.read(MP3_READ_CHUNK)
        if not out: time.sleep(0.01); continue
        yield out

# -----------------------
# Siren Detector
# -----------------------
have_model = os.path.exists(MODEL_PATH)
if have_model:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ Siren model loaded")
else:
    print("⚠️ Siren model missing")

def preprocess_audio_segment(seg, sr=RESULT_SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    x = np.expand_dims(np.expand_dims(log_mel, 0), -1)
    return x.astype(np.float32)

state = {"label": "Unknown", "confidence": 0.0}
state_lock = threading.Lock()

def emergency_blink_lights(duration=10):
    """Blink all lights alternately for the given duration (in seconds)."""
    with emergency_lock:
        emergency_state["active"] = True
    print("🚨 Emergency Mode Activated — Blinking lights!")

    start_time = time.time()
    while time.time() - start_time < duration:
        red.on(); yellow.off(); green.off()
        time.sleep(0.5)
        red.off(); yellow.on()
        time.sleep(0.5)

    red.on(); yellow.off(); green.off()
    print("✅ Emergency Mode Ended — Resuming normal control.")
    with emergency_lock:
        emergency_state["active"] = False

def run_inference_on_buffer():
    if not have_model: return
    seg_len = RESULT_SR * MODEL_DURATION
    while True:
        with buffer_lock:
            buf = np.array(rolling_buffer, dtype=np.float32)
        needed = int(MIC_RATE * MODEL_DURATION)
        mic_seg = buf[-needed:] if buf.size >= needed else np.pad(buf, (needed-buf.size, 0))
        seg = librosa.resample(mic_seg, orig_sr=MIC_RATE, target_sr=RESULT_SR)
        if seg.size < seg_len: seg = np.pad(seg, (0, seg_len - seg.size))
        try:
            x = preprocess_audio_segment(seg)
            interpreter.set_tensor(input_details[0]['index'], x)
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            probs = tf.nn.softmax(out)[0]
            siren_prob = float(probs[1] if probs.shape[0] >= 2 else np.max(probs))
            if siren_prob > CONF_THRESHOLD:
                label = "🚨 Siren Detected"
                # Capture one frame and check for ambulance
                ret, frame = camera.read()
                if ret:
                    both_detected = detect_ambulance_and_siren(frame)
                    if both_detected:
                        print("🚑 Ambulance + Siren confirmed — triggering yellow light!")
                        threading.Thread(target=trigger_emergency_yellow, daemon=True).start()
            else:
                label = "✅ Normal Traffic"


        except Exception as e:
            print("Inference error:", e)
        time.sleep(INFER_INTERVAL)

if have_model:
    threading.Thread(target=run_inference_on_buffer, daemon=True).start()



# -----------------------
# YOLOv8 Vehicle Detection
# -----------------------
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print("✅ YOLOv8n loaded")
else:
    print("⚠️ YOLOv8n model missing — skipping detection")
    yolo_model = None

vehicle_classes = {"car", "truck", "bus", "motorcycle", "bike", "auto", "ambulance", "siren"}
vehicle_state = {"counts": {}}
vehicle_lock = threading.Lock()
gst_state = {"gst": 0.0}
gst_lock = threading.Lock()

# -----------------------
# Green Signal Time Calculator
# -----------------------
def calculate_gst(vehicle_counts, no_of_lanes):
    """
    Calculate Green Signal Time (GST) based on detected vehicle counts.
    Returns 0 if no vehicles detected.
    """
    if not vehicle_counts or sum(vehicle_counts.values()) == 0:
        return 0.0  # 🚫 No vehicles → no green time

    avg_times = {
        'bike': 3.8,
        'car': 5.74,
        'truck': 11.77,
        'auto': 6.73,
        'tempo': 6.53,
        'tractor': 8.98,
        'bus': 8.55
    }

    numerator = 0.0
    for vc, count in vehicle_counts.items():
        avg_time = avg_times.get(vc.lower(), 0)
        numerator += count * avg_time

    gst = numerator / (no_of_lanes + 1)

    # Clamp GST range
    if gst > 60:
        gst = 60
    elif gst < 10:
        gst = 10

    return round(gst, 2)


# -----------------------
# Video Generator
# -----------------------
def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.01)
            continue

        with light_lock:
            phase = light_state["phase"]

        # ⚙️ Run YOLO detection ONLY during Red light
        if yolo_model and phase == "Red":
            results = yolo_model.predict(frame, verbose=False)
            boxes = results[0].boxes
            cls_names = results[0].names
            counts = {}
            for box in boxes:
                label = cls_names[int(box.cls[0])]
                if label.lower() in vehicle_classes:
                    counts[label.capitalize()] = counts.get(label.capitalize(), 0) + 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,255), 2)
                    cv2.putText(frame, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            with vehicle_lock:
                vehicle_state["counts"] = counts

        # Overlay Siren info
        with state_lock:
            siren_label, conf = state["label"], state["confidence"]
        cv2.rectangle(frame, (10, 10), (630, 60), (0,0,0), -1)
        cv2.putText(frame, f"{siren_label} ({conf:.2f})",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# -----------------------
# Web Interface
# -----------------------
@app.route('/')
def index():
    return """
    <html>
    <head>
        <title>🚦 Smart Traffic + Siren + GST</title>
        <style>
            body { background:#111; color:#fff; font-family:Arial; text-align:center; }
            .info { margin-top:10px; font-size:20px; }
            .light { font-size:30px; margin-top:20px; }
            .countdown { font-size:25px; margin-top:10px; color:#0f0; }
            .hidden { display:none; }
            .visible { display:block; }
        </style>
        <script>
            async function updateStatus() {
                const res = await fetch('/live_data');
                const data = await res.json();

                document.getElementById('siren').innerHTML = data.siren.label + " (" + data.siren.confidence + ")";
                let vtext = '';
                for (const [k,v] of Object.entries(data.vehicles)) {
                    vtext += k + ": " + v + " ";
                }
                document.getElementById('vehicles').innerHTML = vtext || 'No vehicles detected';
                document.getElementById('gst').innerHTML = data.gst + " seconds";

                const light = data.light.phase;
                const countdown = data.light.countdown;
                document.getElementById('light').innerHTML = "Current Light: " + light;
                document.getElementById('countdown').innerHTML = "⏱ Countdown: " + countdown + "s";

                const video = document.getElementById('video');
                if (light === "Red") {
                    video.className = "visible";
                } else {
                    video.className = "hidden";
                }
            }
            setInterval(updateStatus, 1000);
            window.onload = updateStatus;
        </script>
    </head>
    <body>
        <h2>🚦 Raspberry Pi Smart Traffic Monitor</h2>
        <img id='video' src='/video_feed' width='640' class='visible'><br>
        <audio controls autoplay src='/audio_feed' type='audio/mp3'></audio>
        <div class='info'>
            <div><strong>Siren Status:</strong> <span id='siren'>Loading...</span></div>
            <div class='counts'><strong>Vehicle Count:</strong> <span id='vehicles'>Loading...</span></div>
            <div class='gst'><strong>Green Signal Time:</strong> <span id='gst'>Calculating...</span></div>
            <div class='light' id='light'>Current Light: Loading...</div>
            <div class='countdown' id='countdown'>⏱ Countdown: --s</div>
        </div>
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(stream_with_context(generate_frames()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio_feed')
def audio_feed():
    return Response(stream_with_context(generate_audio_mp3()),
                    mimetype='audio/mpeg')

@app.route('/live_data')
def live_data():
    with state_lock, vehicle_lock, gst_lock, light_lock:
        return jsonify({
            "siren": state,
            "vehicles": vehicle_state["counts"],
            "gst": gst_state["gst"],
            "light": light_state
        })

def handle_exit(sig, frame):
    print("Shutting down...")
    try: camera.release()
    except: pass
    try: mic.stop_stream(); mic.close()
    except: pass
    try: audio.terminate()
    except: pass
    os._exit(0)
signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == '__main__':
    print("🌍 Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)