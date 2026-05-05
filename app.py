import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import winsound  # Asenkron siren sesi için

st.set_page_config(page_title="Milli IFF Savunma Portalı", layout="wide")
st.title("Milli Dost-Düşman Tanıma (IFF) Sistemi")

@st.cache_resource
def load_model():
    return YOLO("runs/detect/milli_iff_model/weights/best.pt")

model = load_model()

def trigger_alarm_async():
    winsound.PlaySound("siren.wav", winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NOWAIT)

def process_frame(frame):
    results = model(frame)
    is_locked = False
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            
            if conf > 0.75:
                is_locked = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls_name == "DUSMAN":
                    color = (0, 0, 255) # Kırmızı
                    trigger_alarm_async()
                    cv2.putText(frame, "!!! DUSMAN HEDEF !!!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                else:
                    color = (0, 255, 0) # Yeşil
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
    return frame, is_locked
