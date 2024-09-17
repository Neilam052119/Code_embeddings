import time
from datetime import datetime
script_start_time = datetime.now()

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
import pandas as pd
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import matplotlib.pyplot as plt
import numpy as np

# Initialize PaddleOCR and Table Transformer
ocr = PaddleOCR(use_angle_cls=True, lang='en')
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Define paths
input_folder = r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\Code_merge\Delhi\JULY\Extracted_Images_Delhi-July-01-2024'
output_folder = r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\output'
final_excel_path = os.path.join(output_folder, 'keywords.xlsx')

#Define keywords
keywords = ["The Securitisation and Reconstruction of Financial Assets and Enforcement of Security Interest (SARFAESI) Act, 2002 ",
        "SARFAESI Act 2002" , "E-AUCTION UNDER SARFAESI ACT 2002"]

# Function to perform OCR and save results
def perform_ocr(image_path):
    result = ocr.ocr(image_path, cls=True)
    table_data = []
    for line in result:
        for word_info in line:
            text = word_info[1][0]
            confidence = word_info[1][1]
            bbox = word_info[0]
            table_data.append({"text": text, "bbox": bbox, "confidence": confidence})
    return table_data

# Function to detect table and perform OCR
def detect_table_and_ocr(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    
    if len(results["boxes"]) > 0:
        box = results["boxes"][0].tolist()
        box = [round(b, 2) for b in box]
        cropped_image = image.crop((box[0], box[1], box[2], box[3]))
        cropped_image_np = np.array(cropped_image)
        ocr_result = ocr.ocr(cropped_image_np, cls=True)
        table_rows = []
        for line in ocr_result:
            for word_info in line:
                text, confidence = word_info[1]
                table_rows.append({"text": text, "confidence": confidence})
        return table_rows
    return []

#Function to check for keywords in text data
def check_for_keywords(text_data):
    for entry in text_data:
         if any(keyword in entry['text'] for keyword in keywords):
             return "YES"
    return "NO"



# Main loop
all_data = []
for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    if os.path.isfile(image_path):
        # Check if the image contains a table or simple text
        table_data = detect_table_and_ocr(image_path)
        if table_data:
            contains_keywords = check_for_keywords(table_data)
        else:
            text_data = perform_ocr(image_path)
            contains_keywords = check_for_keywords(text_data)
        all_data.append({"Image": image_file, "Contains Keywords": contains_keywords})

# Save final results to Excel
df = pd.DataFrame(all_data)
df.to_excel(final_excel_path, index=False)
print(f"Final OCR results have been saved to {final_excel_path}")

# Visualize the results (optional)
for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        result = perform_ocr(image_path)
        boxes = [data['bbox'] for data in result]
        texts = [data['text'] for data in result]
        scores = [data['confidence'] for data in result]
        image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path= r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\Code_merge\thsarabunnew-webfont.ttf')
        image_with_boxes = Image.fromarray(image_with_boxes)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boxes)
        plt.show()
        image_with_boxes.save(os.path.join(output_folder, f'result_{image_file}'))



script_end_time = datetime.now()
print(f"Script execution time: {script_end_time - script_start_time}")