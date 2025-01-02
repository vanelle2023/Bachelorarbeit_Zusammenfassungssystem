import os
import pytesseract

# **Verzeichnisse für das Speichern hochgeladener Dateien und temporärer Daten**
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"

# Verzeichnisse erstellen, falls sie noch nicht existieren
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# **Globale Konfigurationen**
# Hier können weitere Konfigurationsoptionen hinzugefügt werden, z. B. Debug-Modus, Logging, etc.
DEBUG_MODE = True  # Aktiviert Debugging (in der Produktion auf `False` setzen)
ALLOWED_FILE_TYPES = ['.pdf', '.ppt', '.pptx']  # Erlaubte Dateiformate
