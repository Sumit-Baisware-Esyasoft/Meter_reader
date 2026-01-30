import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Smart Meter Reader",
    page_icon="🔢",
    layout="centered"
)

st.title("🔍 Smart Meter Reading System")
st.caption("Diaphragm & 7-Segment Meter Detection")

# =============================
# LOAD MODELS (CACHE)
# =============================
@st.cache_resource
def load_models():
    diaphragm = YOLO("Diaphrgrm_meter.pt")
    seven_seg = YOLO("7_seg.pt")
    return diaphragm, seven_seg

diaphragm_model, seven_seg_model = load_models()

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.header("⚙️ Meter Configuration")

    meter_type = st.radio(
        "Select Meter Type",
        ["Diaphragm", "7 Segment"]
    )

    msn = st.text_input("Meter Serial Number (MSN)")
    ivrs = st.text_input("IVRS Number")

    if meter_type == "7 Segment":
        unit = st.selectbox(
            "Select Reading Type",
            ["kWh", "Voltage", "kVA", "MD"]
        )
    else:
        unit = "m³"

    conf_threshold = st.slider(
        "🎯 Detection Confidence",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.05
    )

    st.markdown("---")
    st.info(f"📌 Output Unit: **{unit}**")

# =============================
# CAMERA INPUT
# =============================
st.subheader("📷 Capture Meter Image")
camera_img = st.camera_input("Click meter image")

# =============================
# DIAPHRAGM PROCESSING
# =============================
def process_diaphragm(img, conf_thres):
    preds = diaphragm_model(img, imgsz=640, conf=conf_thres, iou=0.45)
    boxes = preds[0].boxes
    tokens = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = diaphragm_model.names[cls_id]

        x1, _, x2, _ = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        if label.isdigit() and conf >= conf_thres:
            tokens.append((x_center, label))

    tokens.sort(key=lambda x: x[0])
    reading = "".join(t[1] for t in tokens)

    return f"{reading} m³" if reading else None

# =============================
# 7-SEGMENT PROCESSING (FIXED)
# =============================
def process_7segment(img, unit, conf_thres):
    preds = seven_seg_model(
        img,
        imgsz=640,
        conf=conf_thres,
        iou=0.45
    )

    boxes = preds[0].boxes
    tokens = []

    h, w, _ = img.shape

    # LCD vertical band (middle area)
    y_min = int(h * 0.35)
    y_max = int(h * 0.65)

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = seven_seg_model.names[cls_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        box_height = y2 - y1

        # 🔒 FILTER NOISE
        if not (y_min <= y_center <= y_max):
            continue

        if box_height < h * 0.05:
            continue

        if label.isdigit() and conf >= conf_thres:
            tokens.append((x_center, label))
        elif label == "10" and conf >= (conf_thres - 0.1):
            tokens.append((x_center, "."))

    tokens.sort(key=lambda x: x[0])
    raw = "".join(t[1] for t in tokens)

    # CLEANUP
    if raw.count(".") > 1:
        raw = raw.replace(".", "", raw.count(".") - 1)
    if raw.startswith("."):
        raw = raw[1:]

    # SAFETY LIMIT
    if len(raw) > 8:
        raw = raw[:8]

    return f"{raw} {unit}" if raw else None

# =============================
# RUN PREDICTION
# =============================
if camera_img is not None:

    img = np.array(bytearray(camera_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    st.image(img, caption="Captured Image", use_container_width=True)

    with st.spinner("🔎 Detecting meter reading..."):
        try:
            if meter_type == "Diaphragm":
                result = process_diaphragm(img, conf_threshold)
            else:
                result = process_7segment(img, unit, conf_threshold)
        except Exception as e:
            st.error(str(e))
            result = None

    if result:
        st.success(f"✅ Detected Reading: **{result}**")

        if st.button("📤 Submit Reading"):
            st.toast("Reading submitted successfully!", icon="✅")
            st.write("### 📄 Submitted Details")
            st.json({
                "Meter Type": meter_type,
                "MSN": msn,
                "IVRS": ivrs,
                "Reading": result,
                "Confidence": conf_threshold
            })
    else:
        st.error("❌ Meter reading not detected")
