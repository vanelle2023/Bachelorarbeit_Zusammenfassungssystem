from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from models import summarizer  # Import der MultimodalSummarizer-Klasse
from settings import UPLOAD_DIR  # Import der Verzeichniskonfigurationen

# Router-Instanz erstellen
router = APIRouter()

# Endpoint: Zählt die Anzahl der Folien in einer hochgeladenen Datei
@router.post("/get_slides_count/")
async def get_slides_count(file: UploadFile = File(...)):
    """
    Bestimmt die Anzahl der Folien in einer hochgeladenen PDF- oder PowerPoint-Datei.
    """
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
            from pdf2image import convert_from_path
            pages = convert_from_path(file_location)
            total_slides = len(pages)

        elif file_extension in ['.ppt', '.pptx']:
            # PowerPoint-Präsentation laden
            from pptx import Presentation
            prs = Presentation(file_location)
            total_slides = len(prs.slides)

        # Temporäre Datei löschen
        os.remove(file_location)

        return {"total_slides": total_slides}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Ermitteln der Folienanzahl: {str(e)}")


# Endpoint: Verarbeitet eine Datei und erstellt eine Zusammenfassung
@router.post("/uploadfile/")
async def upload_file(
    file: UploadFile = File(...),
    start_slide: int = Form(...),
    max_slides: int = Form(...)
):
    """
    Verarbeitet eine hochgeladene Datei (PDF oder PowerPoint) und erstellt eine Zusammenfassung
    für eine bestimmte Anzahl von Folien.
    """
    if not file.filename.endswith(('.pdf', '.ppt', '.pptx')):
        raise HTTPException(status_code=400, detail="Nur PDF und PowerPoint-Dateien sind erlaubt.")

    try:
        # Datei speichern
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Zusammenfassung mit MultimodalSummarizer erstellen
        result = summarizer.process_multiple_slides(file_location, start_slide=start_slide, max_slides=max_slides)

        # Temporäre Datei löschen
        os.remove(file_location)

        # Ergebnis zurückgeben
        return JSONResponse(content={
            "message": f"Datei {file.filename} erfolgreich verarbeitet!",
            "result": result
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
