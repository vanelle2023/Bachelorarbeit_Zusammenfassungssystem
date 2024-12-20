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
            Deiner Plattform für intelligente und effiziente Zusammenfassungen von Vorlesungsfolien.
          </Typography>
          <Typography variant="body1" sx={{ fontSize: '1.2rem', mb: 2 }}>
            Egal, ob du Student oder Dozent bist, unsere Plattform hilft dir dabei, umfangreiche Vorlesungsfolien prägnant
            zusammenzufassen. Unser Ziel ist es, das Lernen und Lehren zu erleichtern, indem wir die wichtigsten
            Informationen aus Präsentationen extrahieren und übersichtlich aufbereiten.
          </Typography>
          <Typography variant="body1" sx={{ fontSize: '1.2rem' }}>
            Erstelle auf einfache Weise Zusammenfassungen, die dir helfen, Inhalte schneller zu erfassen und zu verstehen.
            Mit SmartSummaries erhältst du stets einen prägnanten Überblick, der dir hilft, den roten Faden nicht zu
            verlieren.
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
          Mit SmartSummaries können Sie Vorlesungsfolien in den gängigen Formaten wie PDF und PowerPoint hochladen. 
          Die Plattform ist darauf spezialisiert, Inhalte aus den Bereichen Softwaretechnik, Betriebssysteme und Rechnernetze zu verarbeiten. 
          Unsere Systeme analysieren sowohl Text als auch visuelle Informationen, um Ihnen eine prägnante Zusammenfassung zu bieten.
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
              2. Unsere Plattform analysiert die Folien und erstellt eine automatische Zusammenfassung, die sowohl Text als auch visuelle Elemente berücksichtigt.
          </Typography>
          <Typography variant="body1"sx={{mb: 2, lineHeight: '2'}}>
              3. Sie erhalten die Zusammenfassung in Markdown-Format, die Sie direkt nutzen oder nach Ihren Wünschen bearbeiten können. 
              Alle Zusammenfassungen können im PDF Format exportiert werden, um sie für spätere Zwecke zu verwenden.
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
