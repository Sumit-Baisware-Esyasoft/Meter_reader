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
# SIDEBAR INPUTS
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

    st.markdown("---")
    st.info(f"📌 Output Unit: **{unit}**")

# =============================
# CAMERA INPUT
# =============================
st.subheader("📷 Capture Meter Image")

camera_img = st.camera_input("Click meter image")

# =============================
# PROCESS IMAGE
# =============================
def process_diaphragm(img):
    preds = diaphragm_model(img, imgsz=640, conf=0.35, iou=0.45)
    boxes = preds[0].boxes
    tokens = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = diaphragm_model.names[cls_id]

        x1, _, x2, _ = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        if label.isdigit() and conf > 0.25:
            tokens.append((x_center, label))

    tokens.sort(key=lambda x: x[0])
    reading = "".join(t[1] for t in tokens)

    return reading + " m³" if reading else None


def process_7segment(img, unit):
    preds = seven_seg_model(img, imgsz=640, conf=0.20, iou=0.45)
    boxes = preds[0].boxes
    tokens = []

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = seven_seg_model.names[cls_id]

        x1, _, x2, _ = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        if label.isdigit() and conf > 0.25:
            tokens.append((x_center, label))
        elif label == "10" and conf > 0.15:
            tokens.append((x_center, "."))

    tokens.sort(key=lambda x: x[0])
    reading = "".join(t[1] for t in tokens)

    # Cleanup decimals
    if reading.count(".") > 1:
        reading = reading.replace(".", "", reading.count(".") - 1)
    if reading.startswith("."):
        reading = reading[1:]

    return f"{reading} {unit}" if reading else None

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
                result = process_diaphragm(img)
            else:
                result = process_7segment(img, unit)

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
                "Reading": result
            })
    else:
        st.error("❌ Meter reading not detected")
