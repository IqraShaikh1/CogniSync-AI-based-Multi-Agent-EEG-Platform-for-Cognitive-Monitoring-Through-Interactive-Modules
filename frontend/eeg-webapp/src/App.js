// App.js
import React from "react";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Pages
import MainPage from "./pages/MainPage";
import MentalHealthPage from "./pages/MentalHealthModule"; // your mental health EEG module
import FocusAttentionPage from "./pages/Focus_Attention";   // your focus/attention EEG module

export default function App() {
  return (
    <Router>
      <Routes>
        {/* Home / Landing page */}
        <Route path="/" element={<MainPage />} />
        {/* Mental Health Module */}
        <Route path="/mental-health" element={<MentalHealthPage />} />
        {/* Focus & Attention Tracking */}
        <Route path="/focus-tracking" element={<FocusAttentionPage />} />
        {/* Fallback route */}
        <Route
          path="*"
          element={
            <div className="p-6 text-white">
              <h1 className="text-2xl font-bold mb-4">Page Not Found</h1>
              <p>
                Go back <a href="/" className="text-blue-400 underline">home</a>.
              </p>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}
