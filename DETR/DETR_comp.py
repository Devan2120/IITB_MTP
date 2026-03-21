import torch
import cv2
import time
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForObjectDetection

# 1. Configuration
# DETR: "facebook/detr-resnet-50"
# RT-DETR: "PekingU/rtdetr_r50vd"
MODEL_NAME = "hustvl/yolos-tiny" #YOLOS
VIDEO_INPUT_PATH = r"F:\Edu\IITB\MTP\Code\Files\bird_video2.mp4" 
OUTPUT_FOLDER = r"F:\Edu\IITB\MTP\Code\Files\Out" 

# 2. Setup Output Directory
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
safe_model_name = MODEL_NAME.replace("/", "_")

print(f"Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 3. Setup Video Capture and Writer
cap = cv2.VideoCapture(VIDEO_INPUT_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Construct exact save paths
video_output_path = os.path.join(OUTPUT_FOLDER, f"video_{safe_model_name}.mp4")
csv_output_path = os.path.join(OUTPUT_FOLDER, f"data_{safe_model_name}.csv")

out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 4. Initialize Data Logging
inference_data = [] 
frame_count = 0

print(f"Starting inference. Video and Data will be saved to: {OUTPUT_FOLDER}")

# Initialize the progress bar
pbar = tqdm(total=total_frames, desc="Processing Video", unit="frame")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=rgb_frame, return_tensors="pt").to(device)

    # 1. Measure exact inference time
    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end_time = time.perf_counter()
    
    inference_time_sec = end_time - start_time
    current_fps = 1.0 / inference_time_sec

    # 2. Post-process (Keep threshold low to see all bird predictions)
    target_sizes = torch.tensor([rgb_frame.shape[:2]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

    frame_confidences = []

    # 3. Filter and Draw
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        class_name = model.config.id2label[label.item()].lower()
        
        # --- NEW: The Bird Filter ---
        # If the network thinks it is anything other than a bird, skip it entirely
        if class_name != "bird":
            continue

        # If it IS a bird, process the math and draw the box
        box = [int(i) for i in box.tolist()]
        confidence = round(score.item(), 2)
        frame_confidences.append(confidence)

        box_color = (0, 0, 255) # Swapped to Red so the target pops on screen
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
        text = f"{class_name}: {confidence}"
        cv2.putText(frame, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    cv2.putText(frame, f"FPS: {current_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    out.write(frame)
    pbar.update(1)

    # 4. Calculate stats for BIRDS ONLY and log to CSV
    max_conf = max(frame_confidences) if frame_confidences else 0.0
    avg_conf = sum(frame_confidences) / len(frame_confidences) if frame_confidences else 0.0

    inference_data.append({
        "Frame_Number": frame_count,
        "Inference_Time_sec": round(inference_time_sec, 5),
        "FPS": round(current_fps, 2),
        "Total_Birds": len(frame_confidences),
        "Max_Confidence": round(max_conf, 2),
        "Avg_Confidence": round(avg_conf, 2)
    })

# 5. Cleanup and Save Data
pbar.close()
cap.release()
out.release()

# Convert the logged data into a Pandas DataFrame and save to CSV
df = pd.DataFrame(inference_data)
df.to_csv(csv_output_path, index=False)

print(f"Finished! Video saved to: {video_output_path}")
print(f"Finished! Data saved to: {csv_output_path}")