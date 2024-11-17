import React, { useState } from 'react';
import { Box, Button, Typography, TextField, FormControl, CircularProgress } from '@mui/material';

const Services = () => {
  const [file, setFile] = useState(null);
  const [slideNumber, setSlideNumber] = useState(0);
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

  const handleSlideNumberChange = (e) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 0 && (!totalSlides || value < totalSlides)) {
      setSlideNumber(value);
      setError(null);
    } else {
      setError(`Bitte geben Sie eine gültige Foliennummer zwischen 0 und ${totalSlides - 1} ein.`);
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
    formData.append('slide_number', slideNumber.toString());

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
      
      setSummaryResult({
        originalText: data.result.original_text,
        summary: data.result.summary,
        imagePath: data.result.image_path,
        currentSlide: slideNumber
      });
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
          Fassen Sie Vorlesungsfolien im Handumdrehen zusammen.
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

          {/* Foliennummer mit Validierung */}
          <FormControl fullWidth sx={{ mb: 3 }}>
            <TextField
              label="Foliennummer"
              type="number"
              value={slideNumber}
              onChange={handleSlideNumberChange}
              error={!!error && error.includes('Foliennummer')}
              helperText={
                error && error.includes('Foliennummer')
                  ? error
                  : totalSlides
                  ? `Verfügbare Folien: 0-${totalSlides - 1}`
                  : ''
              }
              inputProps={{ 
                min: 0,
                max: totalSlides > 0 ? totalSlides - 1 : undefined
              }}
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
                }
              }}
              onClick={handleSubmit}
              disabled={loading || !file || (error && error.includes('Foliennummer'))}
            >
              {loading ? <CircularProgress size={24} color="inherit" /> : 'Jetzt zusammenfassen'}
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
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Zusammenfassungsergebnis</span>
            {summaryResult && (
              <Typography variant="body2" component="span" sx={{ color: 'text.secondary' }}>
                Folie {summaryResult.currentSlide}
                {totalSlides > 0 && ` von ${totalSlides - 1}`}
              </Typography>
            )}
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
              <Box sx={{ 
                display: 'flex', 
                justifyContent: 'center', 
                alignItems: 'center', 
                height: '100%',
                flexDirection: 'column',
                gap: 2
              }}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary">
                  Folie wird verarbeitet...
                </Typography>
              </Box>
            )}
            
            {error && (
              <Box sx={{ 
                p: 2, 
                bgcolor: '#ffebee', 
                borderRadius: 1,
                border: '1px solid #ffcdd2'
              }}>
                <Typography color="error">
                  {error}
                </Typography>
              </Box>
            )}

            {summaryResult && (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {/* Folienbild */}
                {summaryResult.imagePath && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle1" gutterBottom sx={{ color: 'text.primary', fontWeight: 500 }}>
                      Folienbild:
                    </Typography>
                    <Box sx={{ 
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      overflow: 'hidden',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
                    }}>
                      <img
                        src={`http://127.0.0.1:8000${summaryResult.imagePath}`}
                        alt="Folienvorschau"
                        style={{ 
                          maxWidth: '100%', 
                          height: 'auto',
                          display: 'block'
                        }}
                      />
                    </Box>
                  </Box>
                )}
                
                {/* Originaltext */}
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" gutterBottom sx={{ color: 'text.primary', fontWeight: 500 }}>
                    Originaltext:
                  </Typography>
                  <Box sx={{ 
                    p: 2,
                    bgcolor: '#f5f5f5',
                    borderRadius: '4px',
                    border: '1px solid #e0e0e0'
                  }}>
                    <Typography variant="body2" sx={{ 
                      whiteSpace: 'pre-wrap',
                      color: 'text.secondary'
                    }}>
                      {summaryResult.originalText || 'Kein Originaltext verfügbar'}
                    </Typography>
                  </Box>
                </Box>

                {/* Zusammenfassung */}
                <Box>
                  <Typography variant="subtitle1" gutterBottom sx={{ color: 'text.primary', fontWeight: 500 }}>
                    Zusammenfassung:
                  </Typography>
                  <Box sx={{ 
                    p: 2,
                    bgcolor: '#e3f2fd',
                    borderRadius: '4px',
                    border: '1px solid #bbdefb'
                  }}>
                    <Typography variant="body1" sx={{ 
                      whiteSpace: 'pre-wrap',
                      color: 'text.primary'
                    }}>
                      {summaryResult.summary || 'Keine Zusammenfassung verfügbar'}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            )}
          </Box>
        </Box>
      </Box>
    </>
  );
};

export default Services;