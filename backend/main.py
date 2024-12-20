from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router  # Importiere den APIRouter aus routes.py

# FastAPI-Anwendung initialisieren
app = FastAPI(
    title="Multimodale Präsentations-Zusammenfassung API",
    description="API für die Analyse und Zusammenfassung von Präsentationsfolien",
    version="1.0.0"
)

# CORS-Middleware hinzufügen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezifische Origins verwenden
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routen aus routes.py einbinden
app.include_router(router)

# Startup-Ereignis für die Anwendung
@app.on_event("startup")
async def startup_event():
    """
    Wird beim Start der Anwendung ausgeführt.
    Kann verwendet werden, um globale Initialisierungen vorzunehmen.
    """
    import os
    from settings import UPLOAD_DIR, TEMP_DIR

    # Temporäre Dateien und Verzeichnisse aufräumen
    for dir_path in [UPLOAD_DIR, TEMP_DIR]:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Fehler beim Aufräumen von {file_path}: {e}")

# Health-Check-Endpoint
@app.get("/health")
async def health_check():
    """
    Ein einfacher Endpoint, um den Status der API zu überprüfen.
    """
    return {"status": "healthy"}

# Start der Anwendung
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)