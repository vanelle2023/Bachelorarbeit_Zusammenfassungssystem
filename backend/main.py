from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import torch
import clip
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, AutoModelForVision2Seq
from transformers import BlipProcessor, BlipForConditionalGeneration
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

    import re

    def clean_text(self, text):
        # Entferne überflüssige Leerzeichen und ersetze unsinnige Zeichen
        text = ' '.join(text.split())
        text = text.replace('|', 'I')
        text = text.replace('1', 'I')
        text = text.replace('0', 'O')
        
        # Setze Leerzeichen zwischen kleinen und großen Buchstaben, wenn sie ohne Leerzeichen verbunden sind
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        
        # Bereinige den Text, um unerwünschte Zeichen zu entfernen (z.B. Sonderzeichen und Emojis)
        text = re.sub(r'[^a-zA-Z0-9äöüÄÖÜß .,?!]', '', text)
        
        # Verwende NLP, um den Text in Sätze zu zerlegen und zu säubern
        doc = self.nlp(text)
        cleaned_sentences = []
        for sent in doc.sents:
            sentence = sent.text.strip()
            
            # Setze den ersten Buchstaben eines Satzes groß
            sentence = sentence[0].upper() + sentence[1:] if sentence else ""
            
            # Falls der Satz kein abschließendes Satzzeichen hat, füge einen Punkt hinzu
            if sentence and not sentence[-1] in ['.', '!', '?']:
                sentence += '.'
            
            # Füge den Satz zur Liste der bereinigten Sätze hinzu
            cleaned_sentences.append(sentence)
        
        # Gib den bereinigten Text als einen einzelnen Textstring zurück
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

        # BLIP Model
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

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
          lang_encoder_path="facebook/bart-large",
          tokenizer_path="facebook/bart-large",
          cross_attn_every_n_layers=4,
          decoder_layers_attr_name="model.decoder.layers",
        )
        self.flamingo_model.to(self.device)
        # Ensure tokenizer uses left-padding for compatibility with decoder-only architectures
        self.flamingo_tokenizer.padding_side = 'left'

    def _describe_image_with_blip(self, image):
        """
        Beschreibt ein Bild mit BLIP.
        """
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            outputs = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
            description = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            return description
        except Exception as e:
            print(f"Fehler bei der Bildbeschreibung mit BLIP: {str(e)}")
            return "Keine Beschreibung verfügbar"

    def process_multiple_slides(self, file_path, start_slide=0, max_slides=1):
        try:
            slides_content = []
            for slide_number in range(start_slide, start_slide + max_slides):
                try:
                    slide_image, slide_text = self.extract_slide_content(file_path, slide_number)
                    slides_content.append((slide_image, slide_text))
                except IndexError:
                    print(f"Folie {slide_number} existiert nicht in der Datei.")
                    break

            if not slides_content:
                raise ValueError("Keine Folien zum Verarbeiten gefunden.")

            # Separate images and texts
            images = [content[0] for content in slides_content]
            texts = [content[1] for content in slides_content]

             # Verarbeite die Bilder mit BLIP
            image_descriptions = [self._describe_image_with_blip(image) or "Keine Bildbeschreibung verfügbar" for image in images]

            # Verarbeite Flamingo nur für die Folien mit Text
            flamingo_summaries = []
            for i, text in enumerate(texts):
                if text.strip():  # Überprüfen, ob Text vorhanden ist
                    flamingo_summaries.append("Keine Textzusammenfassung verfügbar")  # Keine Zusammenfassung generieren, wenn Text leer
                else:
                    flamingo_summaries.append(None)  # Hier könnte eine einfache Möglichkeit sein, auf Bildbeschreibung zu setzen

            # Verarbeite Flamingo in einem Batch und stelle sicher, dass eine gültige Ausgabe vorhanden ist
            try:
                final_flamingo_summaries = self._process_flamingo_batch(images, texts)
                if final_flamingo_summaries is None:
                    final_flamingo_summaries = ["Fehlerhafte Zusammenfassung" for _ in images]  # Fehlerbehandlung
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Flamingo: {str(e)}")
                final_flamingo_summaries = ["Fehlerhafte Zusammenfassung" for _ in images]  # Fehlerbehandlung

            # Kombiniere die Ergebnisse
            combined_summaries = []
            for i in range(len(images)):
                if  final_flamingo_summaries[i] != "Fehlerhafte Zusammenfassung" and  final_flamingo_summaries[i]:  # Sicherstellen, dass die Zusammenfassung gültig ist
                    combined_summaries.append(
                        f"Bildbeschreibung: {image_descriptions[i]}. Textzusammenfassung: { final_flamingo_summaries[i]}"
                    )
                else:
                    combined_summaries.append(f"Bildbeschreibung: {image_descriptions[i]}.")

            # Finalisiere die Zusammenfassung mit BART
            final_summaries = [self._generate_bart_summary(summary) for summary in combined_summaries]

            processed_slides = [
                {
                    'original_text': texts[i],
                    'image_description': image_descriptions[i],
                    'summary': final_summaries[i],
                    'image_path': f"slide_{start_slide + i}.png"
                }
                for i in range(len(images))
            ]

            # Kombiniere alle finalen Zusammenfassungen zu einer umfassenden
            all_summaries = " ".join(final_summaries)
            # Prüfe, ob sinnvolle Zusammenfassungen existieren
            if not all_summaries.strip():
                comprehensive_summary = "Diese Präsentation enthält nur folgende Bildbeschreibungen:\n"
                for i, desc in enumerate(image_descriptions):
                    comprehensive_summary += f"Folie {i + 1}: {desc}\n"
            else:
                comprehensive_summary = "Zusammenfassung der Präsentation:\n"
                for i, summary in enumerate(final_summaries):
                    comprehensive_summary += f"Folie {i + 1}: {summary}\n"

            return {
                'processed_slides': processed_slides,
                'comprehensive_summary': comprehensive_summary
            }

        except Exception as e:
            print(f"Fehler bei der Verarbeitung mehrerer Folien: {str(e)}")
            return None

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

    def _process_images_batch(self, images):
        try:
            image_tensors = [self.clip_preprocess(image).unsqueeze(0) for image in images]
            image_batch = torch.cat(image_tensors).to(self.device)
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_batch)
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Bild-Batches: {str(e)}")
            return None

    def _process_texts_batch(self, texts):
        try:
            text_embeddings = self.sentence_model.encode(texts, convert_to_tensor=True)
            return text_embeddings.cpu().numpy()
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Text-Batches: {str(e)}")
            return None

    def _process_flamingo_batch(self, images, texts):
        try:
            image_tensors = [self.flamingo_image_processor(image).unsqueeze(0).to(self.device) for image in images]
            vision_x = torch.cat([tensor.unsqueeze(0).unsqueeze(0) for tensor in image_tensors], dim=0)

            contexts = [f"{text}" for text in texts]
            tokenizer_outputs = self.flamingo_tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                padding_side='left',   # Ensure left padding
            )
            lang_x = tokenizer_outputs.input_ids.to(self.device)
            attention_mask = tokenizer_outputs.attention_mask.to(self.device)

            outputs = self.flamingo_model.generate(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                num_beams=4
            )
            flamingo_summaries = [
                self.flamingo_tokenizer.decode(output, skip_special_tokens=True) for output in outputs
            ]
            return flamingo_summaries
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von Flamingo-Batches: {str(e)}")
            return None

    def summarize_text(self, text, max_length=150, chunk_size=800):
        """
        Erstellt eine Zusammenfassung eines Textes. 
        Wenn der Text zu lang ist, wird er in Abschnitte unterteilt und schrittweise zusammengefasst.
        """
        try:
            # Text in handhabbare Teile aufteilen
            if len(text) > chunk_size:
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            else:
                chunks = [text]
            
            summaries = []
            for chunk in chunks:
                inputs = self.summarizer_tokenizer(chunk, max_length=1024, truncation=True, return_tensors="pt")
                summary_ids = self.summarizer_model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    min_length=40,
                    length_penalty=2.0,
                    num_beams=4,
                )
                summary = self.summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                summaries.append(summary)
            
            # Fasse die einzelnen Abschnitte erneut zusammen
            combined_summary = " ".join(summaries)
            if len(combined_summary) > chunk_size:
                # Zweite Zusammenfassung, falls die kombinierte zu lang ist
                return self.summarize_text(combined_summary, max_length=max_length)
            return combined_summary
        
        except Exception as e:
            print(f"Fehler bei der Textzusammenfassung: {str(e)}")
            return ""


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
        return slide_image, text.strip()

    def _extract_slide_image(self, slide):
        temp_img_path = "temp_slide.png"
        slide.export(temp_img_path)
        image = Image.open(temp_img_path)
        os.remove(temp_img_path)
        return image

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