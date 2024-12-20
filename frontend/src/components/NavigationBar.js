// NavigationBar.js
import React from 'react';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import { Link } from 'react-router-dom';

const NavigationBar = () => {
  return (
    <AppBar
      position="static"
      sx={{
        backgroundColor: '#E5E2DA', // light beige/grey color as seen in the screenshot
        boxShadow: 'none', // Remove shadow
      }}
    >
      <Toolbar sx={{ justifyContent: 'space-between' }}>
        {/* Logo/Title */}
        <Typography
          variant="h6"
          sx={{
            flexGrow: 1,
            fontWeight: 'bold',
            fontSize: '1.5rem',
            color: '#121C2C', // Dark blue color for the text
          }}
        >
          SmartSummaries
        </Typography>

        {/* Navigation Links */}
        <Button
          component={Link}
          to="/"
          sx={{
            color: '#121C2C', // Same dark blue color for buttons
            textTransform: 'none', // Keep text case as in original
            fontSize: '1.1rem',
          }}
        >
          Home
        </Button>
        <Button
          component={Link}
          to="/services"
          sx={{
            color: '#121C2C',
            textTransform: 'none',
            fontSize: '1.1rem',
          }}
        >
          Services
        </Button>
        <Button
          component={Link}
          to="/kontakt"
          sx={{
            color: '#121C2C',
            textTransform: 'none',
            fontSize: '1.1rem',
          }}
        >
          Kontakt
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default NavigationBar;
