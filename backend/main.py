from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from pptx import Presentation
import fitz  # PyMuPDF for PDFs
from transformers import BartForConditionalGeneration, BartTokenizer
from PIL import Image
import io
from fastapi.middleware.cors import CORSMiddleware
import clip
import torch
from clip.simple_tokenizer import SimpleTokenizer

# FastAPI App initialisieren
app = FastAPI()

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Erlaube nur dein Frontend
    allow_credentials=True,
    allow_methods=["*"],  # Erlaube alle Methoden (GET, POST, etc.)
    allow_headers=["*"],  # Erlaube alle Header (wie Authentifizierung)
)

# Verzeichnis für Uploads erstellen, falls es nicht existiert
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialisiere BART Modell und Tokenizer für die Zusammenfassung
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

# CLIP-Modell laden
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Funktion zum Extrahieren von Text und Bildern aus PPT
def extract_text_images_from_pptx(file, slide_number):
    presentation = Presentation(file)
    slide = presentation.slides[slide_number - 1]
    text = "\n".join([shape.text for shape in slide.shapes if shape.has_text_frame])
    images = [shape.image for shape in slide.shapes if shape.shape_type == 13]
    return text, images

# Funktion zum Extrahieren von Text und Bildern aus PDF
def extract_text_images_from_pdf(file, page_number):
    doc = fitz.open(file)
    page = doc.load_page(page_number - 1)
    text = page.get_text("text")
    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        base_image = doc.extract_image(xref)
        img_data = base_image["image"]
        images.append(Image.open(io.BytesIO(img_data)))
    return text, images

# Funktion zum Aufteilen des Textes in Segmente
def split_text(text, max_tokens=75):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        current_segment.append(word)
        tokens = clip.tokenize([" ".join(current_segment)], truncate=True)
        if tokens.shape[1] > max_tokens:
            segments.append(" ".join(current_segment[:-1]))
            current_segment = [word]

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments

# API-Endpunkt zum Hochladen von Dateien
@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    # Überprüfen, ob der Dateityp erlaubt ist
    if not file.filename.endswith(('.pdf', '.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Nur PDF und PowerPoint-Dateien sind erlaubt.")
    
    # Datei im Verzeichnis 'uploads' speichern
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_location, "wb") as buffer:
        buffer.write(await file.read())

    return JSONResponse(content={"message": f"Datei {file.filename} erfolgreich hochgeladen!", "file_path": file_location})

# API-Endpunkt zum Verarbeiten der Folie oder Seite (PPT oder PDF)
@app.post("/process_slide/")
async def process_slide(file_path: str = Form(...), slide_number: int = Form(...)):
    # Überprüfen, ob die Datei existiert
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Datei nicht gefunden.")
    
    # Bestimmen Sie den Dateityp und verarbeiten Sie ihn entsprechend
    text = ""
    images = []
    if file_path.endswith('.pdf'):
        text, images = extract_text_images_from_pdf(file_path, slide_number)
    elif file_path.endswith(('.ppt', '.pptx')):
        text, images = extract_text_images_from_pptx(file_path, slide_number)
    
    if not text:
        raise HTTPException(status_code=400, detail="Kein Text auf der ausgewählten Seite/Folie gefunden.")
    
    # Schritt 1: Text in Segmente aufteilen
    segments = split_text(text)

    # Schritt 2: Bildverarbeitung und Ähnlichkeitsberechnung
    all_text_for_summary = ""
    if images:
        image = preprocess(images[0]).unsqueeze(0).to(device)  # Nimm das erste Bild
        for segment in segments:
            text_input = clip.tokenize([segment]).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                text_features = clip_model.encode_text(text_input)
                similarity = (image_features @ text_features.T).item()

            print(f"Ähnlichkeit für Abschnitt: {similarity}")
            all_text_for_summary += " " + segment  # Hier kannst du zusätzliche Logik hinzufügen, um nur relevante Abschnitte zu speichern
    
    # Schritt 3: Text zusammenfassen mit BART
    inputs = bart_tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = bart_model.generate(inputs["input_ids"], num_beams=4, max_length=200, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return JSONResponse(content={"summary": summary})

