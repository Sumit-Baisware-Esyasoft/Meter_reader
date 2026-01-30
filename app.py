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
# LOAD MODELS
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
        0.10, 0.80, 0.35, 0.05
    )

    st.markdown("---")
    st.info(f"📌 Output Unit: **{unit}**")

# =============================
# CAMERA INPUT
# =============================
st.subheader("📷 Capture Meter Image")
camera_img = st.camera_input("Click meter image")

# =============================
# DRAW BOUNDING BOXES
# =============================
def draw_boxes(img, boxes, class_names, valid_digits, conf_thres):
    draw = img.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = class_names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        is_valid = label in valid_digits and conf >= conf_thres
        color = (0, 255, 0) if is_valid else (0, 0, 255)

        cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            draw,
            f"{label} ({conf:.2f})",
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return draw

# =============================
# DIAPHRAGM PROCESSING
# =============================
def process_diaphragm(img, conf_thres):
    preds = diaphragm_model(img, imgsz=640, conf=conf_thres, iou=0.45)
    boxes = preds[0].boxes
    tokens = []

    for box in boxes:
        label = diaphragm_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        x1, _, x2, _ = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2

        if label.isdigit() and conf >= conf_thres:
            tokens.append((x_center, label))

    tokens.sort(key=lambda x: x[0])
    reading = "".join(d for _, d in tokens)

    return reading + " m³" if reading else None, boxes

# =============================
# 7 SEGMENT PROCESSING (FIXED)
# =============================
def process_7segment(img, unit, conf_thres):
    preds = seven_seg_model(img, imgsz=640, conf=conf_thres, iou=0.45)
    boxes = preds[0].boxes
    tokens = []

    h, _, _ = img.shape
    y_min, y_max = int(h * 0.35), int(h * 0.65)

    VALID_DIGITS = set("0123456789")

    for box in boxes:
        label = seven_seg_model.names[int(box.cls[0])]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        if not (y_min <= y_center <= y_max):
            continue

        if label in VALID_DIGITS and conf >= conf_thres:
            tokens.append((x_center, label))

    tokens.sort(key=lambda x: x[0])

    # Digit spacing filter (kills ghost '11')
    cleaned = []
    min_gap = 18
    for x, d in tokens:
        if not cleaned or (x - cleaned[-1][0]) >= min_gap:
            cleaned.append((x, d))

    raw = "".join(d for _, d in cleaned)

    if len(raw) > 8:
        raw = raw[:8]

    return (raw + f" {unit}") if raw else None, boxes

# =============================
# RUN PREDICTION
# =============================
if camera_img is not None:

    img = np.array(bytearray(camera_img.read()), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    st.subheader("📸 Original Image")
    st.image(img, use_container_width=True)

    with st.spinner("🔎 Detecting meter reading..."):
        if meter_type == "Diaphragm":
            result, boxes = process_diaphragm(img, conf_threshold)
            boxed_img = draw_boxes(
                img,
                boxes,
                diaphragm_model.names,
                set("0123456789"),
                conf_threshold
            )
        else:
            result, boxes = process_7segment(img, unit, conf_threshold)
            boxed_img = draw_boxes(
                img,
                boxes,
                seven_seg_model.names,
                set("0123456789"),
                conf_threshold
            )

    st.subheader("🟦 Detection Visualization")
    st.image(boxed_img, use_container_width=True)

    if result:
        st.success(f"✅ Detected Reading: **{result}**")

        if st.button("📤 Submit Reading"):
            st.toast("Reading submitted successfully!", icon="✅")
            st.json({
                "Meter Type": meter_type,
                "MSN": msn,
                "IVRS": ivrs,
                "Reading": result,
                "Confidence": conf_threshold
            })
    else:
        st.error("❌ Meter reading not detected")
