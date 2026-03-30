// App.js
import React from "react";
import "./App.css";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Pages
import MainPage          from "./pages/MainPage";
import MentalHealthPage  from "./pages/MentalHealthModule";
import FocusAttentionPage from "./pages/Focus_Attention";
import MoodEmotion       from "./pages/mood_emotion";
import SeizureAlertsPage from "./pages/Seizurealerts";
import ModuleManager     from "./pages/ModuleManager";   // ← NEW
import BrainJournal from './pages/Brainjournal';

export default function App() {
  return (
    <Router>
      <Routes>
        {/* Home / Landing page */}
        <Route path="/"                element={<MainPage />} />

        {/* Module Manager */}
        <Route path="/module-manager"  element={<ModuleManager />} />   {/* ← NEW */}

        {/* Active modules */}
        <Route path="/mental-health"   element={<MentalHealthPage />} />
        <Route path="/focus-tracking"  element={<FocusAttentionPage />} />
        <Route path="/mood-emotion"    element={<MoodEmotion />} />
        <Route path="/seizure-alerts"  element={<SeizureAlertsPage />} />
        <Route path="/brain-journal"    element={<BrainJournal />} />

        {/* Placeholder routes for modules not yet implemented */}
        <Route path="/fatigue"          element={<PlaceholderPage name="Fatigue Detection" />} />
        <Route path="/sleep-monitoring" element={<PlaceholderPage name="Sleep Stage Monitoring" />} />
        <Route path="/meditation"       element={<PlaceholderPage name="Meditation Assistant" />} />
        <Route path="/brain-games"      element={<PlaceholderPage name="Brain-Controlled Games" />} />

        {/* Fallback */}
        <Route
          path="*"
          element={
            <div className="p-6 text-white bg-gray-950 min-h-screen">
              <h1 className="text-2xl font-bold mb-4">Page Not Found</h1>
              <p>Go back <a href="/" className="text-blue-400 underline">home</a>.</p>
            </div>
          }
        />
      </Routes>
    </Router>
  );
}

// Simple placeholder for modules still in development
function PlaceholderPage({ name }) {
  return (
    <div className="p-6 text-white bg-gray-950 min-h-screen flex flex-col items-center justify-center">
      <div className="text-6xl mb-6">🚧</div>
      <h1 className="text-2xl font-bold mb-2">{name}</h1>
      <p className="text-gray-400 mb-6">This module is under development.</p>
      <a href="/" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition text-sm font-semibold">
        ← Back to Home
      </a>
    </div>
  );
}