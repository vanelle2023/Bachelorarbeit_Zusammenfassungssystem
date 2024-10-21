Anleitung zum Starten des Backend-Projekts: 

Voraussetzungen:
1. Stellen Sie sicher, dass Python (Version 3.7 oder höher) und pip installiert sind.
2. pip und python sollten in den Umgebungsvariablen Ihres Systems hinterlegt sein, damit sie von überall aufgerufen werden können.

Navigieren Sie zu dem Verzeichnis, in dem Ihr Backend-Projekt gespeichert ist. Zum Beispiel:
cd D:\backend

Überprüfen Sie die Struktur
Ihre Verzeichnisstruktur sollte in etwa so aussehen:
backend/
├── main.py
├── uploads/

Benötigte Pakete installieren:
Führen Sie folgende Befehle aus, um die erforderlichen Pakete zu installieren
pip install uvicorn
pip install python-multipart PyMuPDF pptx
pip install fastapi[all]
pip install git+https://github.com/openai/CLIP.git
pip install transformers torch torchvision

Starten des Servers:
Um das FastAPI-Backend zu starten, führen Sie den folgenden Befehl aus:
uvicorn main:app --reload

Zugriff auf die API:
Sobald der Server läuft, können Sie die API über Ihren Browser oder ein Tool wie Postman erreichen. Standardmäßig wird die API unter folgender Adresse bereitgestellt:
http://127.0.0.1:8000
Die interaktive API-Dokumentation ist unter /docs erreichbar:
http://127.0.0.1:8000/docs
