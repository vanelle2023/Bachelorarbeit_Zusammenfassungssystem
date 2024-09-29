
Anleitung zum Starten des Backend-Projekts: 

Voraussetzungen
Stellen Sie sicher, dass Python (3.7 oder höher) und pip installiert sind.
Installieren Sie die erforderlichen Pakete, falls noch nicht geschehen:
pip install fastapi uvicorn

Navigieren Sie zu dem Verzeichnis, in dem Ihr Backend-Projekt gespeichert ist. Zum Beispiel:
cd D:\backend

Überprüfen Sie die Struktur
Ihre Verzeichnisstruktur sollte in etwa so aussehen:
backend/
├── main.py
├── uploads/

Starten des Servers
Um das FastAPI-Backend zu starten, führen Sie den folgenden Befehl aus:
uvicorn main:app --reload
Zugriff auf die API

Sobald der Server läuft, können Sie die API über Ihren Browser oder ein Tool wie Postman erreichen. Standardmäßig wird die API unter folgender Adresse bereitgestellt:
http://127.0.0.1:8000
Die interaktive API-Dokumentation ist unter /docs erreichbar:
http://127.0.0.1:8000/docs

