from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForVision2Seq
from open_flamingo import create_model_and_transforms
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

# Konfiguration von Tesseract für OCR (Texterkennung aus Bildern)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezifische Origins definieren
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definiert Verzeichnisse für das Hochladen von Dateien und temporäre Dateien
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

 # Klasse zur Textbereinigung und Anpassung
class TextCleaner:
    def __init__(self):
        try:
            # Lädt ein Spacy-Modell für die deutsche Sprache
            self.nlp = spacy.load('de_core_news_sm')
        except:
            # Lädt das Modell herunter, falls es nicht vorhanden ist
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

        # CLIP Model
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        # Summary Model
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

        # Sentence Embedding Model
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        # Text Cleaner
        self.text_cleaner = TextCleaner()

        # Load Flamingo model and processor using create_model_and_transforms
        self.flamingo_model, self.flamingo_image_processor, self.flamingo_tokenizer = create_model_and_transforms(
          clip_vision_encoder_path="ViT-B-32",  # Kleinere ViT-Variante
          clip_vision_encoder_pretrained="openai",
          lang_encoder_path="facebook/bart-large",  # Kompatibles Modell wie BART
          tokenizer_path="facebook/bart-large",
          cross_attn_every_n_layers=4,
          decoder_layers_attr_name="model.decoder.layers",
        )
        self.flamingo_model.to(self.device)

    def process_file(self, file_path, slide_number=0):
        try:
            image, text = self.extract_slide_content(file_path, slide_number)
            image_features = self.get_clip_features(image)

            summary = self.summarize_text(text) if len(text) > 100 else text
            text_embedding = self.sentence_model.encode(summary)

            # Prozessiere mit Flamingo
            flamingo_summary = self._generate_flamingo_summary(image, text)

            # Nutze Flamingo und BART zur finalen Zusammenfassung
            final_summary = self._generate_bart_summary(flamingo_summary)

            return {
                'original_text': text,
                'summary': final_summary,
                'image_path': f"slide_{slide_number}.png",
                'image_features': image_features.tolist(),
                'text_embedding': text_embedding.tolist()
            }
        except Exception as e:
            print(f"Fehler: {str(e)}")
            return None

    def _generate_flamingo_summary(self, image, text):
        # Bereite das Bild vor
        image_tensor = self.flamingo_image_processor(image).unsqueeze(0).to(self.device)
        vision_x = image_tensor.unsqueeze(0).unsqueeze(0)  # Erweitere Dimensionen

        # Bereite den Text vor
        context = f"Text: {text}"
        tokenizer_output = self.flamingo_tokenizer(
            [context] * vision_x.shape[0],  # Gleichen Kontext für jede Bildsequenz verwenden
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        lang_x = tokenizer_output.input_ids.to(self.device)
        attention_mask = tokenizer_output.attention_mask.to(self.device)

        # Generiere eine Zusammenfassung mit Flamingo
        outputs = self.flamingo_model.generate(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=attention_mask,
            max_new_tokens=100,
            do_sample=True,  # Sampling aktivieren
            top_k=50,
            #pad_token_id=self.flamingo_tokenizer.pad_token_id or self.flamingo_tokenizer.eos_token_id,
            #eos_token_id=self.flamingo_tokenizer.eos_token_id,
            num_beams=4
        )

        # Dekodiere die generierten Tokens
        flamingo_summary = self.flamingo_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return flamingo_summary

    def _generate_bart_summary(self, flamingo_summary):
        # Nutze BART für die finale Textzusammenfassung
        inputs = self.summarizer_tokenizer(flamingo_summary, max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = self.summarizer_model.generate(
            inputs["input_ids"],
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
        )
        summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = self.text_cleaner.clean_text(summary)
        return summary

    def extract_slide_content(self, file_path, slide_number=0):
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            images = convert_from_path(file_path)
            slide_image = images[slide_number]
            custom_config = r'--oem 3 --psm 6 -l deu'
            text = pytesseract.image_to_string(slide_image, config=custom_config)
            slide_image.save(f"slide_{slide_number}.png")

        elif file_extension in ['.pptx', '.ppt']:
            prs = Presentation(file_path)
            slide = prs.slides[slide_number]
            text = ""
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"

            # Folienbild exportieren
            slide_image = self._extract_slide_image(slide)
            slide_image.save(f"slide_{slide_number}.png")
        else:
            raise ValueError("Nicht unterstütztes Dateiformat")

        text = self.text_cleaner.clean_text(text)
        text = self.text_cleaner.remove_english_terms(text)

        return slide_image, text.strip()

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

    def _extract_slide_image(self, slide):
        temp_img_path = "temp_slide.png"
        slide.export(temp_img_path)
        image = Image.open(temp_img_path)
        os.remove(temp_img_path)
        return image

    def process_multiple_slides(self, file_path, start_slide=0, max_slides=3):
        try:
            # Ergebnisse für jede Folie speichern
            processed_slides = []

            # Iteriere über die gewünschten Folien
            for slide_number in range(start_slide, start_slide + max_slides):
                try:
                    result = self.process_file(file_path, slide_number=slide_number)
                    if result:
                        processed_slides.append(result)
                except IndexError:
                    # Wenn es weniger Folien als erwartet gibt
                    print(f"Folie {slide_number} existiert nicht in der Datei.")
                    break

            # Erstelle eine umfassende Zusammenfassung aus allen individuellen Zusammenfassungen
            all_summaries = " ".join([slide['summary'] for slide in processed_slides if 'summary' in slide])
            comprehensive_summary = self.summarize_text(all_summaries)

            return {
                'processed_slides': processed_slides,
                'comprehensive_summary': comprehensive_summary
            }

        except Exception as e:
            print(f"Fehler bei der Verarbeitung mehrerer Folien: {str(e)}")
            return None

# Globale Instanz des Summarizers
summarizer = MultimodalSummarizer()

@app.post("/get_slides_count/")
async def get_slides_count(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Nur PDF und PowerPoint-Dateien sind erlaubt.")
    
    try:
        # Datei speichern
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Anzahl der Folien ermitteln
        file_extension = os.path.splitext(file.filename)[1].lower()
        total_slides = 0

        if file_extension == '.pdf':
            # PDF in Bilder umwandeln, um die Seitenzahl zu bestimmen
            pages = convert_from_path(file_location)
            total_slides = len(pages)

        elif file_extension in ['.ppt', '.pptx']:
            # PowerPoint-Präsentation laden
            prs = Presentation(file_location)
            total_slides = len(prs.slides)

        # Temporäre Datei löschen
        os.remove(file_location)

        return {"total_slides": total_slides}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Ermitteln der Folienanzahl: {str(e)}")


@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...), start_slide: int = Form(...), max_slides: int = Form(...)):
    if not file.filename.endswith(('.pdf', '.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Nur PDF und PowerPoint-Dateien sind erlaubt.")
    
    try:
        # Datei speichern
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Datei verarbeiten
        result = summarizer.process_multiple_slides(file_location, start_slide=start_slide, max_slides=max_slides)
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