import React from 'react';
import { Box, Typography, Link } from '@mui/material';

const Footer = () => {
  return (
    <Box
      sx={{
        backgroundColor: '#E5E2DA', // beige Hintergrundfarbe
        padding: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}
    >
      {/* Logo */}
      <Typography
        variant="h6"
        sx={{
          fontWeight: 'bold',
          fontSize: '1.5rem',
          color: '#121C2C', // dunkelblauer Text
        }}
      >
        SmartSummaries
      </Typography>

      {/* Standort */}
      <Box sx={{ textAlign: 'left' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#121C2C' }}>
          Standort
        </Typography>
        <Typography>Flughafenallee 10</Typography>
        <Typography>28199 Bremen</Typography>
        <Typography>Hochschule Bremen</Typography>
        <Typography>Fakultät 4</Typography>
      </Box>

      {/* Kontakt */}
      <Box sx={{ textAlign: 'left' }}>
        <Typography variant="h6" sx={{ fontWeight: 'bold', color: '#121C2C' }}>
          Kontakt
        </Typography>
        <Link
          href="mailto:vlemalieu@stud.hs-bremen.de"
          underline="hover"
          sx={{ color: '#121C2C' }}
        >
          vlemalieu@stud.hs-bremen.de
        </Link>
        <Typography>+49 176 85437598</Typography>
      </Box>
    </Box>
  );
};

export default Footer;
