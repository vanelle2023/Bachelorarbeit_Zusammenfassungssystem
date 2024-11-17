from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pdf2image import convert_from_path
from pptx import Presentation
from sentence_transformers import SentenceTransformer
import pytesseract
import numpy as np
import spacy
import re
from typing import Optional
import shutil

# FastAPI App initialisieren
app = FastAPI(
    title="Multimodale Präsentations-Zusammenfassung API",
    description="API für die Analyse und Zusammenfassung von Präsentationsfolien"
)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezifische Origins definieren
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Konstanten
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

class TextCleaner:
    def __init__(self):
        try:
            self.nlp = spacy.load('de_core_news_sm')
        except:
            os.system('python -m spacy download de_core_news_sm')
            self.nlp = spacy.load('de_core_news_sm')

    def clean_text(self, text):
        text = ' '.join(text.split())
        text = text.replace('|', 'I')
        text = text.replace('1', 'I')
        text = text.replace('0', 'O')
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        
        doc = self.nlp(text)
        cleaned_sentences = []
        for sent in doc.sents:
            sentence = sent.text.strip()
            sentence = sentence[0].upper() + sentence[1:] if sentence else ""
            if sentence and not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
            cleaned_sentences.append(sentence)
        
        return ' '.join(cleaned_sentences)

    def remove_english_terms(self, text):
        common_replacements = {
            'enthusiastically': 'enthusiastisch',
            'with': 'mit',
            'Stakeholder': 'Interessengruppen',
        }
        for eng, ger in common_replacements.items():
            text = text.replace(eng, ger)
        return text

class MultimodalSummarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.text_cleaner = TextCleaner()

    async def process_file(self, file_path: str, slide_number: int):
        try:
            image, text = await self.extract_slide_content(file_path, slide_number)
            image_features = self.get_clip_features(image)
            
            if len(text) > 100:
                summary = self.summarize_text(text)
            else:
                summary = text
            
            text_embedding = self.sentence_model.encode(summary)
            
            # Bild temporär speichern für die Rückgabe
            temp_image_path = os.path.join(TEMP_DIR, f"slide_{slide_number}.png")
            image.save(temp_image_path)
            
            return {
                'original_text': text,
                'summary': summary,
                'image_path': temp_image_path,
                'image_features': image_features.tolist(),
                'text_embedding': text_embedding.tolist()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Fehler bei der Verarbeitung: {str(e)}")

    async def extract_slide_content(self, file_path: str, slide_number: int):
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            images = convert_from_path(file_path)
            slide_image = images[slide_number]
            custom_config = r'--oem 3 --psm 6 -l deu'
            text = pytesseract.image_to_string(slide_image, config=custom_config)
            
        elif file_extension in ['.pptx', '.ppt']:
            prs = Presentation(file_path)
            if slide_number >= len(prs.slides):
                raise HTTPException(status_code=400, detail=f"Ungültige Foliennummer. Die Präsentation hat nur {len(prs.slides)} Folien.")
            
            slide = prs.slides[slide_number]
            text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
            
            slide_image = self._extract_slide_image(slide)
        else:
            raise HTTPException(status_code=400, detail="Nicht unterstütztes Dateiformat")
        
        text = self.text_cleaner.clean_text(text)
        text = self.text_cleaner.remove_english_terms(text)
        
        return slide_image, text.strip()

    def _extract_slide_image(self, slide):
        temp_img_path = os.path.join(TEMP_DIR, "temp_slide.png")
        slide.export(temp_img_path)
        image = Image.open(temp_img_path)
        os.remove(temp_img_path)
        return image

    def get_clip_features(self, image):
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
        return image_features.cpu().numpy()

    def summarize_text(self, text, max_length=150):
        inputs = self.summarizer_tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = self.summarizer_model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
        )
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = self.text_cleaner.clean_text(summary)
        summary = self.text_cleaner.remove_english_terms(summary)
        return summary

# Globale Instanz des Summarizers
summarizer = MultimodalSummarizer()

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), slide_number: int = Form(0)):
    """
    Lädt eine Datei hoch und erstellt eine multimodale Zusammenfassung.
    
    - **file**: Die hochzuladende PDF- oder PowerPoint-Datei
    - **slide_number**: Die Nummer der zu analysierenden Folie (Standard: 0)
    """
    if not file.filename.endswith(('.pdf', '.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Nur PDF und PowerPoint-Dateien sind erlaubt.")
    
    try:
        # Datei speichern
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Datei verarbeiten
        result = await summarizer.process_file(file_location, slide_number)
        
        # Aufräumen
        os.remove(file_location)
        
        return JSONResponse(content={
            "message": f"Datei {file.filename} erfolgreich verarbeitet!",
            "result": result
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Wird beim Start der Anwendung ausgeführt"""
    # Temporäre Dateien aufräumen
    for dir_path in [UPLOAD_DIR, TEMP_DIR]:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Fehler beim Aufräumen von {file_path}: {e}")

@app.get("/health")
async def health_check():
    """Endpoint für Gesundheitsprüfung"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)