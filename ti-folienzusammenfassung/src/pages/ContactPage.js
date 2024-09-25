import React, { useState } from 'react';
import { Box, Button, TextField, Typography } from '@mui/material';
import contactImage from '../assets/images/contactImage.png';

const ContactPage = () => {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    message: '',
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    // Hier kann man die Formulardaten weiterverarbeiten oder an eine API senden
    console.log('Formulardaten:', formData);
  };

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100vh',
        backgroundColor: '#E5E2DA',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: '2rem',
      }}
    >
      {/* Hauptbox für Text und Bild */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          gap: '2rem',
          maxWidth: '1200px',
          width: '100%',
        }}
      >
        {/* Linker Bereich für Text und Formular */}
        <Box sx={{ flex: 1 }}>
          <Typography variant="h3" gutterBottom>
            Nimm Kontakt mit uns auf
          </Typography>
          <Typography variant="body1" paragraph>
            Wir freuen uns über Ihr Feedback. Sollten Sie auf Fehlermeldungen oder technische Probleme bei der Nutzung des Systems stoßen, oder möchten Sie Verbesserungsvorschläge oder neue Ideen zur Weiterentwicklung des Systems einreichen, zögern Sie nicht, uns eine Nachricht zu hinterlassen. Wir bemühen uns, Ihnen innerhalb von 2-3 Werktagen zu antworten.
          </Typography>
          <form onSubmit={handleSubmit}>
            <Typography sx={{marginBottom: 2}}>Name (erforderlich)</Typography>
            <Box display="flex" gap={30} mb={2}>
              <Typography>Vorname</Typography>
              <Typography>Nachname</Typography>
            </Box>
            <Box display="flex" gap={2} mb={2}>
              <TextField
                fullWidth
                label="Vorname"
                name="firstName"
                value={formData.firstName}
                onChange={handleChange}
                required
                InputProps={{
                  sx: {
                    borderRadius: '8px',
                  },
                }}
              />
              <TextField
                fullWidth
                label="Nachname"
                name="lastName"
                value={formData.lastName}
                onChange={handleChange}
                required
                InputProps={{
                  sx: {
                    borderRadius: '8px',
                  },
                }}
              />
            </Box>
            <Typography sx={{marginBottom: 2}}>E-Mail-Adresse (erforderlich)</Typography>
            <TextField
              fullWidth
              label="E-Mail-Adresse"
              name="email"
              value={formData.email}
              onChange={handleChange}
              required
              sx={{ mb: 2 }}
              InputProps={{
                sx: {
                  borderRadius: '8px',
                },
              }}
            />
            <Typography sx={{marginBottom: 2}}>Nachricht (erforderlich)</Typography>
            <TextField
              fullWidth
              label="Nachricht"
              name="message"
              value={formData.message}
              onChange={handleChange}
              required
              multiline
              rows={4}
              sx={{ mb: 2 }}
              InputProps={{
                sx: {
                  borderRadius: '8px',
                },
              }}
            />
            <Button
              type="submit"
              variant="contained"
              sx={{
                backgroundColor: '#121C2C', 
                color: '#FFFFFF', 
                borderRadius: '8px',
                '&:hover': {
                  backgroundColor: '#000000', 
                },
              }}
            >
              Senden
            </Button>
          </form>
        </Box>

        {/* Rechter Bereich für das Bild */}
        <Box
          sx={{
            flex: 1,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <img
            src={contactImage}
            alt="Kontakt Bild"
            style={{
              maxWidth: '100%',
              height: 'auto',
              borderRadius: '8px',
            }}
          />
        </Box>
      </Box>
    </Box>
  );
};

export default ContactPage;
