import React from 'react';
import { Box, Typography, Container, Button } from '@mui/material';
import introImage from '../assets/images/introImage.png'
import homepage from '../assets/images/homepage.jpg'
import { Link } from 'react-router-dom';

const Home = () => {
  return (
    <>
      <Box
        sx={{
          backgroundImage: `url(${introImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          height: '100vh', 
          color: 'white',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        <Container>
          <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 2 }}>
            Herzlich willkommen bei SmartSummaries
          </Typography>
          <Typography variant="h5" sx={{ mb: 4 }}>
            Eine Plattform entwickelt im Rahmen einer Bachelorarbeit zur automatischen Zusammenfassung von Veranstaltungsfolien.
          </Typography>
          <Typography variant="body1" sx={{ fontSize: '1.2rem', mb: 2 }}>
            SmartSummaries nutzt fortschrittliche multimodale neuronale Netze, um Texte und Bilder aus Vorlesungsfolien zu analysieren und prägnante Zusammenfassungen zu erstellen. 
            Ziel ist es, den Lernprozess zu vereinfachen und die wichtigsten Informationen effizient bereitzustellen.
          </Typography>
        </Container>
      </Box>
      <Box
        sx={{
          mt: 5, 
          padding: '2rem', 
        }}
      >
        <Typography variant="h1" sx={{fontWeight: 900}} gutterBottom>
          Was kann hier zusammengefasst werden?
        </Typography>
        <Typography variant="body1" sx={{lineHeight: '2'}}>
          Mit SmartSummaries können Sie Vorlesungsfolien im PDF- oder PowerPoint-Format hochladen. 
          Die Plattform erstellt präzise Zusammenfassungen für jede einzelne Seite, basierend auf einer innovativen Kombination von Text- und Bildanalysen.
        </Typography>
      </Box>
      <Box sx={{
          display: 'flex',
          alignItems: 'flex-start',
          padding: '2rem',
        }}>
        <Box sx={{ mr: 4 }}> {/* Abstand zwischen Bild und Text */}
          <img src={homepage} alt="Prozess" style={{ maxWidth: '600px', borderRadius: '8px' }} />
        </Box>
        <Box>
          <Typography variant="h1" sx={{fontWeight: 900}} gutterBottom>
              Wie läuft der Prozess?
          </Typography>
          <Typography variant="body1" sx={{mt: 4, mb: 2}}>
              1. Laden Sie Ihre Vorlesungsfolien in PDF- oder PowerPoint-Format hoch.
          </Typography>
          <Typography variant="body1" sx={{mb: 2, lineHeight: '2'}}>
              2. Unsere Plattform analysiert die Folien mit Hilfe multimodaler neuronaler Netze und erstellt automatische Zusammenfassungen, die Text- und visuelle Inhalte integrieren.
          </Typography>
          <Typography variant="body1"sx={{mb: 2, lineHeight: '2'}}>
              3. Sie erhalten eine prägnante, textuelle Zusammenfassung, die einfach zu kopieren und weiterzuverwenden ist.
          </Typography>
          <Link to="/services" style={{ textDecoration: 'none' }}>
          <Button
            variant="contained"
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
            }}
          >
            ZUSAMMENFASSEN
          </Button>
          </Link>
        </Box>
      </Box>
    </>
  );
};

export default Home;
