# Multimodale Präsentations-Zusammenfassung

Dieses Projekt ist als Proof of Concept konzipiert, um die Leistungsfähigkeit der CLIP- und Flamingo-Modelle im Bereich der multimodalen Verarbeitung zu evaluieren. Konkret wurde eine Plattform entwickelt, die PDF- und PowerPoint-Dateien analysiert und automatisch zusammenfasst. Die technische Umsetzung erfolgt durch eine moderne Zwei-Tier-Architektur, bestehend aus einem Backend mit FastAPI und einem Frontend in React.

---

## **Inhaltsverzeichnis**

1. [Projektübersicht](#projektübersicht)
2. [Funktionen](#funktionen)
3. [Technologien](#technologien)
4. [Projektstruktur](#projektstruktur)
5. [Installation und Ausführung](#installation-und-ausführung)
6. [API-Endpunkte](#api-endpunkte)
7. [Frontend-Funktionen](#frontend-funktionen)

---

## **Projektübersicht**

Dieses System ermöglicht:
- Hochladen von PDF- oder PowerPoint-Dateien.
- Automatische Analyse der Folien mit Bild- und Textzusammenfassungen.
- Nutzung von neuronalen Modellen wie CLIP, BLIP und Flamingo.
- Bereitstellung einer API zur Interaktion mit dem Frontend.

---

## **Funktionen**

### **Backend**
- Analyse von Vorlesungsfolien.
- Bild- und Texterkennung.
- Generierung von Zusammenfassungen.
- Bildbeschreibung mit BLIP.

### **Frontend**
- Datei-Upload (PDF/PPT).
- Darstellung der zusammengefassten Ergebnisse.
- Interaktive Benutzeroberfläche mit React.

---

## **Technologien**

- **Backend:** 
  - Python, FastAPI
  - PyTorch, Hugging Face Transformers
  - pdf2image, python-pptx
- **Frontend:**
  - React
  - Material-UI

---

## **Projektstruktur**

- 📁 backend
  -  main.py # Haupt-Backend-Einstiegspunkt (FastAPI)
  -  routes.py # API-Endpunkte
  -  models.py # Modelle und Verarbeitungslogik  
  -  settings.py # Konfigurationen
  -  uploads/ # Temporäre Uploads
  -  temp/ # Temporäre Dateien
- 📁 frontend
  - 📁 components  # UI-Komponenten
  - 📁 pages   # Seiten (z. B. Home,Kontakt) 
  - 📁 public/ # Statische Dateien
  -  .gitignore # Ignorierte Dateien für Git
  -  package.json # Node-Abhängigkeiten für das Frontend
-  README.md # Projektdokumentation 

---

## **Installation und Ausführung**

### **Voraussetzungen**
- **Backend:** Python (3.8+), Pip
- **Frontend:** Node.js (16+), npm oder yarn

### **Schritte**

1. **Backend installieren**
   ```
   cd backend/
   pip install uvicorn
   pip install Pillow
   pip install spacy
   pip install deep-translator
   pip install open-flamingo
   pip install python-multipart PyMuPDF 
   pip install python-pptx
   pip install fastapi[all]
   pip install git+https://github.com/openai/CLIP.git
   pip install transformers torch torchvision
   pip install sentence-transformers
   ```
   Um das FastAPI-Backend zu starten, führen Sie den folgenden Befehl aus:
   ``` 
   uvicorn main:app --reload
   ```
   **Hinweis:**

   **pdf2image** erfordert poppler-utils für die Konvertierung von PDF in Bilder: 

    Linux:
    ```
    sudo apt install poppler-utils
    ```

    Windows: Lade Poppler herunter und füge es zu deinem PATH hinzu.
    macOS:
    ```
    brew install poppler
    ```

   **Spacy Sprachmodell (Deutsch):**
   ```
   python -m spacy download de_core_news_sm
   ```
2. **Frontend installieren**
```
cd frontend/
npm install
npm run start
```

---

## **Api-Endpunkte**

1. **/get_slides_count/ (POST)**

Beschreibung: Ermittelt die Anzahl der Folien in einer Datei.
Parameter: Datei (PDF/PPT).
Antwort: { "total_slides": 10 }

2. **/uploadfile/ (POST)**

Beschreibung: Verarbeitet die Datei und erstellt eine Zusammenfassung.
Parameter:
Datei (PDF/PPT)
start_slide: Startfoliennummer
max_slides: Anzahl der zu verarbeitenden Folien
Antwort: { "result": {...} }

3. **/health (GET)**

Beschreibung: Prüft, ob das Backend funktioniert.
Antwort: { "status": "healthy" }

---

## **Frontend-Funktionen:**

1. Datei-Upload:
Ermöglicht das Hochladen von PDF- und PPT-Dateien.

2. Zusammenfassungsanzeige:
Zeigt die generierten Zusammenfassungen und Bildbeschreibungen an.

3. Interaktive Navigation:
Benutzerfreundliches Interface mit React und Material-UI.
