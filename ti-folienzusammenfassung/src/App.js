import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import NavigationBar from './components/NavigationBar';
import Footer from './components/Footer';
import Home from './pages/Home';
import Services from './pages/Services';
import ContactPage from './pages/ContactPage';
import theme from './components/theme';
import { CssBaseline } from '@mui/material';

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline>
        <Router>
          <div className="App">
            <NavigationBar />
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/services" element={<Services />} />
              <Route path="/kontakt" element={<ContactPage />} />
            </Routes>
            <Footer /> {/* Fußzeile ebenfalls hier einfügen */}
          </div>
        </Router>
      </CssBaseline>
    </ThemeProvider>
  );
}

export default App;
