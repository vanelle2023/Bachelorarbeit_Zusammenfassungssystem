from fastapi import FastAPI, File, UploadFile, HTTPException
import os
from fastapi.responses import JSONResponse

# FastAPI App initialisieren
app = FastAPI()

# Verzeichnis für Uploads erstellen, falls es nicht existiert
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    return JSONResponse(content={"message": f"Datei {file.filename} erfolgreich hochgeladen!"})
