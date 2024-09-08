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
from fuzzywuzzy import fuzz
import json
import re

# Initialize PaddleOCR and Table Transformer
ocr = PaddleOCR(use_angle_cls=True, lang='en')
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Define paths
input_folder = r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\Delhi\JULY\Extracted_Images_Delhi-July-01-2024\Delhi_01Jul2024_crop_image-17.jpeg\articles'
output_folder = r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\output'
final_excel_path = os.path.join(output_folder, 'keywords.xlsx')

#Define keywords
keywords = ["Securitisation","Reconstruction","Financial","Enforcement","Security","Interest","SARFAESI","POSSESSION",'Immovable']

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
# import cv2
# import pytesseract
# from pytesseract import Output
# from PIL import Image
# import numpy as np

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     print("image",image)

#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Remove noise using a Gaussian blur
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Apply adaptive thresholding to get binary image
#     binary = cv2.adaptiveThreshold(blurred, 255,
#                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                    cv2.THRESH_BINARY, 11, 2)
   
#     return binary

# def ocr_image(image_path, config_options="--psm 6", extract_tables=False):
#     """
#     This function takes an image path and performs OCR on it.
#     :param image_path: Path to the image to be processed.
#     :param config_options: Configuration options for Tesseract. Default is --psm 6 (for block of text).
#     :param extract_tables: If set to True, returns structured table-like data.
#     :return: Extracted text or table (if extract_tables is True).
#     """
#     # Preprocess the image for better OCR results
#     processed_image = preprocess_image(image_path)

#     # Perform OCR on the image
#     data = pytesseract.image_to_data(processed_image, output_type=Output.DICT, config=config_options)

#     if extract_tables:
#         # Extract table-like structure if necessary
#         return extract_table_from_ocr(data)

#     # Otherwise, return the extracted plain text
#     extracted_text = " ".join(data['text'])
#     print("extracted_text",extracted_text)
#     1/0

#     return extracted_text

# def extract_table_from_ocr(ocr_data):
#     """
#     Parses the OCR data to extract table-like structure by identifying bounding boxes.
#     :param ocr_data: The raw data output from pytesseract.
#     :return: A structured representation of a table (as a 2D array of text).
#     """
#     num_boxes = len(ocr_data['level'])
#     rows = []
#     current_row = []
#     previous_y = None

#     for i in range(num_boxes):
#         # Extract position and text info from OCR data
#         (x, y, w, h, text) = (ocr_data['left'][i], ocr_data['top'][i],
#                               ocr_data['width'][i], ocr_data['height'][i],
#                               ocr_data['text'][i].strip())

#         # Skip empty text
#         if len(text) == 0:
#             continue

#         # New line based on vertical position change
#         if previous_y is None or abs(y - previous_y) > 10:  # Adjust the threshold as needed
#             if len(current_row) > 0:
#                 rows.append(current_row)
#             current_row = []
       
#         current_row.append(text)
#         previous_y = y

#     if len(current_row) > 0:
#         rows.append(current_row)

#     return rows

# # Example Usage
# image_path = r"C:\Users\neela\OneDrive\Desktop\Code_embeddings\Delhi\JULY\Extracted_Images_Delhi-July-01-2024\Delhi_01Jul2024_crop_image-17.jpeg\articles\im.jpg.jpg"
# ocr_text = ocr_image(image_path, extract_tables=True)  # Extracts tables from the image
# #print(ocr_text)
# # Split by common separators (comma, semicolon, question mark, etc.)
separators = r'[ ,;?!]+'


# Function to check for keywords in text data using fuzzy matching
def check_for_keywords(text_data, keywords, threshold=80):
    for entry in text_data:
        split_sentence = re.split(separators,entry['text'])
        print("split_sentence",split_sentence)
        for word in split_sentence:

            for keyword in keywords:
                if fuzz.partial_ratio(word.lower(), keyword.lower()) >= threshold:
                    return "YES"
            
        
    return "NO"


# Function to save extracted words to a .txt or .json file
def save_extracted_words(extracted_words, output_path, file_format='txt'):
    if file_format == 'txt':
        with open(output_path, 'w') as file:
            for word in extracted_words:
                file.write(f"{word}\n")
    elif file_format == 'json':
        with open(output_path, 'w') as file:
            json.dump(extracted_words, file, indent=4)
    else:
        raise ValueError("Unsupported file format. Use 'txt' or 'json'.")

# Main loop
all_data = []
#save_extracted_words(all_data, 'extracted_words.txt', file_format='txt')
extracted_words = []
print(os.listdir(input_folder))

for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    print(os.path.isfile(image_path))

    if os.path.isfile(image_path):
        #Check if the image contains a table or simple text
        table_data = detect_table_and_ocr(image_path)
        if table_data:
            extracted_words.extend([entry['text'] for entry in table_data])
            print("table_data",table_data)
            print("keywords",keywords)
            contains_keywords = check_for_keywords(table_data, keywords)
        else:
            text_data = perform_ocr(image_path)
            extracted_words.extend([entry['text'] for entry in text_data])
            contains_keywords = check_for_keywords(text_data, keywords)
        all_data.append({"Image": image_file, "Contains Keywords": contains_keywords})
        #text_data = ocr_image(image_path)
        #contains_keywords = check_for_keywords(text_data, keywords)
        #all_data.append({"Image": image_file, "Contains Keywords": contains_keywords})

    


# Debug print to check the contents of extracted_words
print("Extracted Words:", extracted_words)
print("All Data:", all_data)

# Save final results to Excel
df = pd.DataFrame(all_data)
df.to_excel(final_excel_path, index=False)
print(f"Final OCR results have been saved to {final_excel_path}")

# Save extracted words to .txt and .json files
save_extracted_words(extracted_words, 'extracted_words.txt', 'txt')
save_extracted_words(extracted_words, 'extracted_words.json', 'json')
print("Extracted words have been saved to extracted_words.txt and extracted_words.json")

# Visualize the results (optional)
for image_file in os.listdir(input_folder):
    image_path = os.path.join(input_folder, image_file)
    if os.path.isfile(image_path):
        image = Image.open(image_path)
        result = perform_ocr(image_path)
        boxes = [data['bbox'] for data in result]
        texts = [data['text'] for data in result]
        scores = [data['confidence'] for data in result]
        image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path= r'C:\Users\neela\OneDrive\Desktop\Code_embeddings\thsarabunnew-webfont.ttf')
        image_with_boxes = Image.fromarray(image_with_boxes)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_boxes)
        plt.show()
        image_with_boxes.save(os.path.join(output_folder, f'result_{image_file}'))



script_end_time = datetime.now()
print(f"Script execution time: {script_end_time - script_start_time}")

print("hello-world - test git commit with app")