Python 3.14.2 (tags/v3.14.2:df79316, Dec  5 2025, 17:18:21) [MSC v.1944 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

st.title("🔥 Trajectoire Cyborg Escrime")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

def extract_trajectory(image: Image.Image):
    """Extrait trajectoires corps + épée → lignes droites"""
    img_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
...     results = pose.process(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
...     
...     if results.pose_landmarks:
...         h, w = img_rgb.shape[:2]
...         points = []
...         
...         # Points clés escrime : épaule, coude, poignet, hanche (épée)
...         keypoints = [12, 14, 16, 11, 13, 15]  # Droite corps + épée
...         
...         for kp_id in keypoints:
...             lm = results.pose_landmarks.landmark[kp_id]
...             x, y = int(lm.x * w), int(lm.y * h)
...             points.append((x, y))
...         
...         # Trajectoires = lignes droites entre points
...         img_lines = img_rgb.copy()
...         for i in range(len(points)-1):
...             cv2.line(img_lines, points[i], points[i+1], (0,255,255), 8)
...         
...         # Ligne épée longue (poignet → extension)
...         sword_end = (int(points[-1][0] + 100*(points[-1][0]-points[-2][0])), 
...                     int(points[-1][1] + 100*(points[-1][1]-points[-2][1])))
...         cv2.line(img_lines, points[-1], sword_end, (255,0,0), 12)
...         
...         return Image.fromarray(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
...     
...     return image
... 
... uploaded_file = st.file_uploader("📸 Photo geste escrime", type=["jpg", "jpeg", "png"])
... 
... if uploaded_file is not None:
...     image = Image.open(uploaded_file)
...     
...     col1, col2 = st.columns(2)
...     with col1:
...         st.image(image, caption="Geste original")
...     with col2:
...         trajectory_img = extract_trajectory(image)
...         st.image(trajectory_img, caption="🚀 Trajectoire Cyborg")
...     
...     st.balloons()
