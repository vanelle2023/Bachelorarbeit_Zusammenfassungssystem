import React, { useState } from 'react';
import { Box, Button, Typography, TextField, Select, MenuItem, FormControl, InputLabel, RadioGroup, FormControlLabel, Radio } from '@mui/material';

const Services = () => {
  const [file, setFile] = useState(null);
  const [summaryType, setSummaryType] = useState('paragraph');
  const [language, setLanguage] = useState('de');

  const handleFileUpload = (e) => {
    const uploadedFile = e.target.files[0];
    if (uploadedFile && ['application/pdf', 'application/vnd.ms-powerpoint', 'application/vnd.openxmlformats-officedocument.presentationml.presentation'].includes(uploadedFile.type)) {
      setFile(uploadedFile);
    } else {
      alert('Nur PDF-, PPT- oder PPTX-Dateien sind zulässig.');
      setFile(null);
    }
  };

  const handleSummaryTypeChange = (e) => {
    setSummaryType(e.target.value);
  };

  const handleLanguageChange = (e) => {
    setLanguage(e.target.value);
  };

  const handleSubmit = () => {
    if (!file) {
      alert('Bitte laden Sie eine Datei hoch, bevor Sie die Zusammenfassung erstellen.');
      return;
    }
    // Hier kannst du die Logik hinzufügen, um die Datei zu verarbeiten und die Zusammenfassung zu erstellen.
    alert('Zusammenfassung wird erstellt...');
  };

  return (
    <>
    <Box sx={{ padding: '2rem' }}>
    <Typography variant="h5" gutterBottom>
          Fassen Sie Vorlesungsfolien im Bereich Softwaretechnik, Rechnernetze und Betriebssystem im Handumdrehen zusammen.
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
            inputProps={{ accept: '.pdf,.ppt,.pptx' }}  // Akzeptiere nur bestimmte Dateiformate
            onChange={handleFileUpload}
          />
        </FormControl>

        {/* Auswahl: Absatz oder Stichpunkte */}
        <FormControl component="fieldset" sx={{ mb: 3 }}>
          <Typography variant="body1" sx={{ mb: 3 }} gutterBottom>
            Wählen Sie den Zusammenfassungstyp:
          </Typography>
          <RadioGroup row value={summaryType} sx={{ mb: 3 }} onChange={handleSummaryTypeChange}>
            <FormControlLabel value="paragraph" control={<Radio />} label="Absatz" />
            <FormControlLabel value="bullet_points" control={<Radio />} label="Stichpunkte" />
          </RadioGroup>
        </FormControl>

        {/* Sprachauswahl */}
        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel id="language-select-label">Sprache</InputLabel>
          <Select
            labelId="language-select-label"
            id="language-select"
            value={language}
            onChange={handleLanguageChange}
          >
            <MenuItem value="de">Deutsch</MenuItem>
            <MenuItem value="en">Englisch</MenuItem>
          </Select>
        </FormControl>

        {/* Button zum Zusammenfassen */}
        <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
          <Button variant="contained" color="primary" sx={{
              mt: 5, 
              bgcolor: 'black', 
              fontWeight: 500,
              color: 'white',
              borderRadius: '20px', 
              padding: '13px 25px', 
              '&:hover': {
                bgcolor: 'gray',
              },
            }} onClick={handleSubmit}>
            Jetzt zusammenfassen
          </Button>
        </Box>
      </Box>

      {/* Rechte Seite: Zusammenfassungsergebnis */}
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
          Zusammenfassungsergebnis:
        </Typography>
        <Box sx={{ height: '300px', overflowY: 'auto', padding: '1rem', border: '1px solid #ddd' }}>
          {/* Hier wird die Zusammenfassung angezeigt */}
          <Typography variant="body1">
            {/* Platzhalter für die tatsächliche Zusammenfassung */}
            Hier wird die Zusammenfassung angezeigt, nachdem die Datei analysiert wurde.
          </Typography>
        </Box>
      </Box>
    </Box>
    </>
  );
};

export default Services;
