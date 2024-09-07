import time
from datetime import datetime
import os
import fitz  # PyMuPDF
import re
print("PyMuPDF is successfully imported")

from ultralytics import YOLO  # Assuming this is the correct import for your YOLO model
from PIL import Image

script_start_time = datetime.now()

def parse_filename(pdf_name):
    """
    Attempts to extract the city, day, month, and year from various common filename formats.
    Example formats:
    - City-Month-Day-Year.pdf
    - City_Day-Month-Year.pdf
    - City_Month-Day-Year.pdf
    - City_YYYY-MM-DD.pdf
    """

    # Define regex patterns for different formats
    patterns = [
        r'(?P<city>[A-Za-z]+)[-_](?P<month>[A-Za-z]+)[-_](?P<day>\d{1,2})[-_](?P<year>\d{4})',  # City-Month-Day-Year
        r'(?P<city>[A-Za-z]+)[-_](?P<day>\d{1,2})[-_](?P<month>[A-Za-z]+)[-_](?P<year>\d{4})',  # City_Day-Month-Year
        r'(?P<city>[A-Za-z]+)[-_](?P<year>\d{4})[-_](?P<month>\d{1,2})[-_](?P<day>\d{1,2})',    # City_YYYY-MM-DD
    ]

    for pattern in patterns:
        match = re.match(pattern, pdf_name)
        if match:
            city = match.group('city')
            day = match.group('day')
            month = match.group('month')
            year = match.group('year')

            # Convert month to numerical if it's a month name
            try:
                date_obj = datetime.strptime(f"{day}-{month}-{year}", "%d-%b-%Y")
            except ValueError:
                try:
                    date_obj = datetime.strptime(f"{day}-{month}-{year}", "%d-%B-%Y")
                except ValueError:
                    date_obj = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")

            return city, date_obj

    raise ValueError(f"Filename {pdf_name} doesn't match any expected formats.")

def pdf_image(main_folder, model_weight_path):

    # Flags for path checks
    main_folder_exists = os.path.exists(main_folder)
    model_weight_path_exists = os.path.exists(model_weight_path)

    # Check if the main folder exists
    if not main_folder_exists:
        print(f"Folder {main_folder} does not exist")
        return
    else:
        print(f"Folder {main_folder} exists")

    # Check if the model weight path exists
    if not model_weight_path_exists:
        print(f"Folder {model_weight_path} does not exist")
        return
    else:
        print(f"Folder {model_weight_path} exists")

    # Process each folder within the main folder
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        if os.path.isdir(folder_path):
            pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
            if not pdf_files:
                print(f"No PDF files found in {folder_path}.")
                continue

            print(f"PDF files found in {folder_path}.")
            for pdf_file in pdf_files:
                pdf_path = os.path.join(folder_path, pdf_file)
                pdf_name = os.path.splitext(pdf_file)[0]

                # Attempt to parse the filename for city and date information
                try:
                    city, date_obj = parse_filename(pdf_name)
                except ValueError as e:
                    print(e)
                    continue

                document = fitz.open(pdf_path)
                saving_folder = os.path.join(folder_path, 'Extracted_Images_' + pdf_name)
                os.makedirs(saving_folder, exist_ok=True)
                print(f"Images will be saved in {saving_folder}")

                for index, page in enumerate(document, start=1):
                    pix = page.get_pixmap()
                    image_name = f"image-{index}.jpeg"
                    pix.save(os.path.join(saving_folder, image_name))

                # Initialize YOLO model and process images
                model = YOLO(model_weight_path)
                image_files = [file for file in os.listdir(saving_folder) if file.endswith('.jpeg')]
                for image_file in image_files:
                    image_path = os.path.join(saving_folder, image_file)
                    if os.path.exists(image_path) and os.access(image_path, os.R_OK):
                        image = Image.open(image_path)
                        results = model.predict(image, conf=0.20, imgsz=640, save=True, save_txt=True, save_conf=True,
                                                show_conf=True)
                        # Saving cropped images with city and date in the filename
                        crop_name = f"{city}_{date_obj.strftime('%d%b%Y')}_crop_{image_file}"
                        results[0].save_crop(os.path.join(saving_folder, crop_name))
                    else:
                        print(f"File {image_path} does not exist")

# Define the paths
main_folder = r"C:\Users\neela\OneDrive\Desktop\Code_embeddings\Delhi"
model_weight_path = r"C:\Users\neela\OneDrive\Desktop\Code_embeddings\last (1).pt"
pdf_image(main_folder, model_weight_path)

script_end_time = datetime.now()
print(f"Total script time taken : {script_end_time - script_start_time}")