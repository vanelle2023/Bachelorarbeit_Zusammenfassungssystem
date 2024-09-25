import { createTheme } from '@mui/material/styles';
import '@fontsource/space-grotesk';
import '@fontsource/raleway';

const theme = createTheme({
  typography: {
    fontFamily: 'Raleway, sans-serif',
    h1: {
      fontFamily: 'Space Grotesk, sans-serif',
      fontSize: '2rem',
    },
    h2: {
      fontFamily: 'Space Grotesk, sans-serif',
      fontSize: '1.75rem',
    },
    h3: {
      fontFamily: 'Space Grotesk, sans-serif',
      fontSize: '1.5rem',
    },
    body1: {
      fontFamily: 'Raleway, sans-serif',
      fontSize: '1rem',
    },
    button: {
      fontFamily: 'Raleway, sans-serif',
      fontSize: '1rem',
      textTransform: 'none',
    },
  },
  palette: {
    background: {
      default: '#E5E2DA', // Gleiche Hintergrundfarbe wie die NavigationBar
    },
    text: {
      primary: '#121C2C', // Dunkelblauer Farbton für den Text
    },
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          backgroundColor: '#E5E2DA', // Setze die Hintergrundfarbe für den gesamten Body
        },
      },
    },
  },
});

export default theme;
