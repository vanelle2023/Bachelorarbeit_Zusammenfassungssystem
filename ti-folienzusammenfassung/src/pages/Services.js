import React, { useState } from 'react';
import { Box, Button, Typography, TextField, FormControl, CircularProgress } from '@mui/material';

const Services = () => {
  const [file, setFile] = useState(null);
  const [startSlide, setStartSlide] = useState(0);
  const [maxSlides, setMaxSlides] = useState(3);
  const [loading, setLoading] = useState(false);
  const [summaryResult, setSummaryResult] = useState(null);
  const [error, setError] = useState(null);
  const [totalSlides, setTotalSlides] = useState(0);

  const handleFileUpload = async (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && ['application/pdf', 'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'].includes(uploadedFile.type)) {
      setFile(uploadedFile);
      setError(null);
      
      // Optional: Anzahl der Folien ermitteln
      try {
        const formData = new FormData();
        formData.append('file', uploadedFile);
        const response = await fetch('http://127.0.0.1:8000/get_slides_count/', {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        setTotalSlides(data.total_slides);
      } catch (error) {
        console.error('Fehler beim Ermitteln der Folienanzahl:', error);
      }
    } else {
      setError('Nur PDF-, PPT- oder PPTX-Dateien sind zulässig.');
      setFile(null);
    }
  };

  const handleSlideInputChange = (setter) => (e) => {
    const value = parseInt(e.target.value, 10);
    if (!isNaN(value) && value >= 0 && (!totalSlides || value <= totalSlides)) {
      setter(value);
      setError(null);
    } else {
      setError('Ungültige Eingabe für Foliennummer.');
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError('Bitte laden Sie eine Datei hoch.');
      return;
    }

    setLoading(true);
    setError(null);
    setSummaryResult(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('start_slide', startSlide.toString());
    formData.append('max_slides', maxSlides.toString());

    try {
      const response = await fetch('http://127.0.0.1:8000/uploadfile/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Ein Fehler ist aufgetreten');
      }

      const data = await response.json();
      setSummaryResult(data);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Box sx={{ padding: '2rem' }}>
        <Typography variant="h5" gutterBottom>
          Fassen Sie mehrere Vorlesungsfolien zusammen.
        </Typography>
      </Box>
      <Box sx={{ padding: '2rem', display: 'flex', justifyContent: 'space-between' }}>
        {/* Linke Seite: Datei-Upload und Optionen */}
        <Box sx={{ width: '45%' }}>
          {/* Datei-Upload */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <Typography variant="body1" sx={{ mb: 3 }} gutterBottom>
              Laden Sie eine Datei hoch (nur PDF, PPT, PPTX):
            </Typography>
            <TextField
              type="file"
              inputProps={{ accept: '.pdf,.ppt,.pptx' }}
              onChange={handleFileUpload}
              error={!!error && error.includes('Datei')}
              helperText={error && error.includes('Datei') ? error : ''}
            />
          </FormControl>

          {/* Startfolie */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <TextField
              label="Startfolie"
              type="number"
              value={startSlide}
              onChange={handleSlideInputChange(setStartSlide)}
              error={!!error && error.includes('Startfolie')}
              helperText={totalSlides ? `Verfügbare Folien: 0-${totalSlides - 1}` : ''}
            />
          </FormControl>

          {/* Anzahl der Folien */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <TextField
              label="Anzahl der Folien"
              type="number"
              value={maxSlides}
              onChange={handleSlideInputChange(setMaxSlides)}
              error={!!error && error.includes('Anzahl')}
            />
          </FormControl>

          {/* Button zum Zusammenfassen */}
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <Button
              variant="contained"
              color="primary"
              sx={{
                mt: 5,
                bgcolor: 'black',
                fontWeight: 500,
                color: 'white',
                borderRadius: '20px',
                padding: '13px 25px',
                '&:hover': {
                  bgcolor: 'gray',
                },
                '&:disabled': {
                  bgcolor: 'rgba(0, 0, 0, 0.12)',
                },
              }}
              onClick={handleSubmit}
              disabled={loading || !file}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Zusammenfassen'}
            </Button>
          </Box>
        </Box>

        {/* Rechte Seite: Ergebnisanzeige */}
        <Box
          sx={{
            width: '45%',
            border: '1px solid #ccc',
            borderRadius: '8px',
            padding: '1rem',
            backgroundColor: '#f9f9f9',
          }}
        >
          <Typography variant="h6" gutterBottom>
            Zusammenfassungsergebnis
          </Typography>

          <Box
            sx={{
              height: '600px',
              overflowY: 'auto',
              padding: '1rem',
              border: '1px solid #ddd',
              borderRadius: '4px',
              backgroundColor: 'white',
            }}
          >
            {loading && (
              <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary">
                  Folien werden verarbeitet...
                </Typography>
              </Box>
            )}

            {error && (
              <Typography color="error">
                {error}
              </Typography>
            )}

            {summaryResult && (
              <Box>
                <Typography variant="h6" sx={{ mt: 4 }}>
                  Umfassende Zusammenfassung:
                </Typography>
                <Typography variant="body2">
                  {summaryResult.result['comprehensive_summary']}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>
      </Box>
    </>
  );
};

export default Services;