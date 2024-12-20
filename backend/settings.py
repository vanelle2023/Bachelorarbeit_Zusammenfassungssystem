import os
import pytesseract

# **Verzeichnisse für das Speichern hochgeladener Dateien und temporärer Daten**
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"

# Verzeichnisse erstellen, falls sie noch nicht existieren
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# **Tesseract-OCR Konfiguration**
# Pfad zur Tesseract-Installation (auf Windows-Systemen erforderlich)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Setzen des TESSDATA_PREFIX für Sprachdaten
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

# **Globale Konfigurationen**
# Hier können weitere Konfigurationsoptionen hinzugefügt werden, z. B. Debug-Modus, Logging, etc.
DEBUG_MODE = True  # Aktiviert Debugging (in der Produktion auf `False` setzen)
ALLOWED_FILE_TYPES = ['.pdf', '.ppt', '.pptx']  # Erlaubte Dateiformate
