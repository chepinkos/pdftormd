from ultralytics import YOLO
import torch
import time
from PIL import Image
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from transformers import TrOCRProcessor
from optimum.onnxruntime import ORTModelForVision2Seq

from pdf2image import convert_from_path
#import easyocr
import cv2

from texify.inference import batch_inference
from texify.model.model import load_model as texify_load_model
from texify.model.processor import load_processor as texify_load_processor
from IPython.display import display, Math, Latex

from jiwer import cer, wer

import requests

import pytesseract

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_directory = "./pictures"


model = YOLO("./yolo_model/best_v.pt")
model.to(device)
## Загрузка модели pix2text
processor = TrOCRProcessor.from_pretrained('breezedeus/pix2text-mfr')
pix2text = ORTModelForVision2Seq.from_pretrained(
    'breezedeus/pix2text-mfr',
    use_cache=False,
    use_io_binding=False,
    provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
)

pix2text.to(device)
##

## Загрузка модели Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata\rus.traindata"'
custom_config = r'--oem 3 --psm 6'


model_texify = texify_load_model()
processor_texify = texify_load_processor()
##

def tesseract_recognition(cropped_image):
    '''
    функция обрабатывает изображение для лучшего распознования его, с помощью tesseract
    также функция делает предикт моделью tesseract и возвращает текст
    '''
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

    resize = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    equalize = cv2.equalizeHist(resize)
    
    _, contrast = cv2.threshold(equalize, 50, 255, cv2.THRESH_BINARY)
    
    
    text = pytesseract.image_to_string(contrast, lang='rus', config=custom_config)
    print("Распознанный текст:", text)
    
    return text


def correct_text(text):
    '''
        функция испльзует API яндекса
        для исправления орографических ошибок
        в тексте
    '''
    url = "https://speller.yandex.net/services/spellservice.json/checkText"
    params = {"text": text}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        errors = response.json()
        corrected_text = text

        # Заменяем ошибки на первый предложенный вариант
        for error in errors:
            wrong_word = error["word"]  # Слово с ошибкой
            suggestion = error["s"][0]  # Первый предложенный вариант
            corrected_text = corrected_text.replace(wrong_word, suggestion)
        
        return corrected_text
    else:
        raise Exception(f"Ошибка подключения к API: {response.status_code}")

counter = 1
def get_text_detection(img_path, file_name):
    """
    Функция для распознавания текста и формул
    Функция получает путь к лситу для распознавания,
    модель yolo8 выделяет зоны с текстом, формулами и рисунками
    и раздаёт обнаруженные объекты по разным моделям.
    Текст передаём в tesseract
    На выходе получаем датафрейм с лейблом, координатами рамки и распознанным текстом или формулой,
    или ссылкой на изображение.
    Изображения сохраняются как отдельные jpeg файлы.
    """
    if os.path.splitext(img_path)[1] == '.pdf':
        img = convert_from_path(img_path, 500)
        img = np.array(img[0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = img_path
        
    results = model.predict(img, save_txt=False)
    print(len(results[0].boxes))
    result = results[0]
    df = pd.DataFrame(columns=['image', 'label', 'x1', 'y1', 'x2', 'y2', 'detected'])
    boxes = result.boxes.xyxy   # Координаты боксов
    scores = result.boxes.conf  # Вероятности
    nms_thresh = 0.3
    filtered_detections = torch.ops.torchvision.nms(boxes, scores, nms_thresh)
    for idx in filtered_detections:
        counter=1
        # Извлекаем данные для текущего бокса
        box = result.boxes[idx]  # Объект бокса
        label = result.names[box.cls.item()]  # Извлечение имени класса
        cords = [round(x) for x in box.xyxy[0].tolist()]  # Координаты в формате [x1, y1, x2, y2]
        prob = round(box.conf.item(), 2)  # Уверенность модели

        print("Object type:", label)
        print("Coordinates:", cords)
        print("Probability:", prob)
        print('---')

        # Определяем координаты границ текста
        x1, y1, x2, y2 = cords[0], cords[1], cords[2], cords[3]
        # Загружаем изображение
        if os.path.splitext(img_path)[1] == '.pdf':
            image = img
        else:
            image = cv2.imread(img)
        # Проверяем, что координаты находятся в пределах изображения
        h, w, _ = image.shape
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            # Обрезаем изображение по указанным координатам
            cropped_image = image[y1:y2, x1:x2]

            # Отображаем обрезанное изображение
            display(Image.fromarray(cropped_image))
        else:
            print(f"Skipped invalid box with coordinates: {cords}")

        if label == 'Text': 
            recognized_text = tesseract_recognition(cropped_image)
            
            corrected = correct_text(recognized_text)
            print("Исправленный текст:", corrected)
            new_row = pd.DataFrame({
                'image': [img_path],
                'label': [label],
                'x1': [x1],
                'y1': [y1],
                'x2': [x2],
                'y2': [y2],
                'detected': [corrected]
            })
            df = pd.concat([df, new_row], ignore_index=True)
            print(f'x1 = {x1} y_m = {(y1+y2)/2}')
        elif label == 'Formula':
            image_fps = Image.fromarray(cropped_image)
            #images = [Image.open(fp).convert('RGB') for fp in image_fps]
            pixel_values = processor(images=image_fps, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = pix2text.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            text = generated_text[0]
            text = text.replace('\\\\', '\\')
            text = f'${text}$  '
            print(text)
            print(f'x1 = {x1} y_m = {(y1+y2)/2}')
            new_row = pd.DataFrame({
                'image': [img_path],
                'label': [label],
                'x1': [x1],
                'y1': [y1],
                'x2': [x2],
                'y2': [y2],
                'detected': [text]
            })
            df = pd.concat([df, new_row], ignore_index=True)

        else:
            path_to_cropped_image = f"./pictures/{file_name}_pictures_{counter}.jpeg"
            print(path_to_cropped_image)
            Image.fromarray(cropped_image).save(path_to_cropped_image, 'JPEG')
            absolute_path = os.path.abspath(path_to_cropped_image)
            counter += 1

            new_row = pd.DataFrame({
                'image': [img_path],
                'label': [label],
                'x1': [x1],
                'y1': [y1],
                'x2': [x2],
                'y2': [y2],
                'detected': [f"""![picture_{counter}]({absolute_path})  \n"""]
            })
            df = pd.concat([df, new_row], ignore_index=True)
    df['y_mean'] = (df['y1'] + df['y2'])/2
    return df



    

def final_file_assembly(dir_path, out_path, file_name):
    '''
        функция собирает результаты работы моделей
        отрабатывающих в функции get_text_detection
        и сортирует их в правильном порядке по координатам
        и собирает в markdown файл
    '''
    df = get_text_detection(dir_path, file_name)
    df = df.sort_values(by=['y_mean'])
    df = df.reset_index(drop=True)

    
    df['y_order'] = 1
    for i in range(1, len(df)):
        if df['y_mean'].iloc[i] > df['y_mean'].iloc[i - 1] + 4:
            df.loc[i, 'y_order'] = df.loc[i - 1, 'y_order'] + 1
        else:
            df.loc[i, 'y_order'] = df.loc[i - 1, 'y_order']
    df = df.sort_values(by=['y_order', 'x1'])

    print(df['y_mean'])


    grouped = df.groupby('y_order')
    markdown_content = "\n".join(
        " ".join(row['detected'] for _, row in group.sort_values(by=['x1']).iterrows())
        for _, group in grouped
    )
    
    with open(out_path, "w") as file:
        file.write(markdown_content)
    return    




UPLOAD_DIR = "uploads"
RESULTS_DIR = "results"


os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/process/")
async def process_file(file: UploadFile):
    # Сохраняем загруженный файл
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Вызываем функцию обработки
    
    result_path = os.path.join(RESULTS_DIR, f"result_{file.filename}.rmd")
    final_file_assembly(file_path, result_path, file.filename)
    

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>File Processed</title>
    </head>
    <body>
        <h1>File processed successfully!</h1>
        <p>Your result file: <a href="/download/{os.path.basename(result_path)}" target="_blank">{os.path.basename(result_path)}</a></p>
        <button onclick="window.location.href='/'">Back to Upload</button>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/download/{filename}", response_class=FileResponse)
def download_file(filename: str):
    file_path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(file_path):
        return file_path
    return {"error": "File not found"}

@app.get("/", response_class=HTMLResponse)
def main_page():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>File Processing</title>
    </head>
    <body>
        <h1>Upload a file for processing</h1>
        <form action="/process/" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="application/pdf,image/*" required>
            <button type="submit">Upload</button>
        </form>
        <h2>Processed Files</h2>
        <ul>
    """

    # Список обработанных файлов
    files = os.listdir(RESULTS_DIR)
    for file in files:
        html_content += f'<li><a href="/download/{file}" target="_blank">{file}</a></li>'

    html_content += """
        </ul>
    </body>
    </html>
    """
    return html_content

# Статические файлы для сохранения загруженных изображений или других артефактов
app.mount("/static", StaticFiles(directory="static"), name="static")