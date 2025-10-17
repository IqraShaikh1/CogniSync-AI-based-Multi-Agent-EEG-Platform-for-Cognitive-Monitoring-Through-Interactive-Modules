import React from "react";
import { Link } from "react-router-dom";
import { HeartPulse, Focus, BatteryCharging, Moon, Brain, Gamepad2, Smile, NotebookText, AlertTriangle } from "lucide-react";
import mentalImage from "../assets/hero.png";

export default function MainPage() {
  const modules = [
    { name: "Mental Health Detection", icon: <HeartPulse className="w-6 h-6" />, path: "/mental-health" },
    { name: "Focus and Attention Tracking", icon: <Focus className="w-6 h-6" />, path: "/focus-tracking" },
    { name: "Fatigue Detection", icon: <BatteryCharging className="w-6 h-6" />, path: "/fatigue" },
    { name: "Sleep Stage Monitoring", icon: <Moon className="w-6 h-6" />, path: "/sleep-monitoring" },
    { name: "Meditation Assistant", icon: <Brain className="w-6 h-6" />, path: "/meditation" },
    { name: "Brain-Controlled Games", icon: <Gamepad2 className="w-6 h-6" />, path: "/brain-games" },
    { name: "Mood and Emotion Recognition", icon: <Smile className="w-6 h-6" />, path: "/mood-emotion" },
    { name: "Daily Brain Journal", icon: <NotebookText className="w-6 h-6" />, path: "/brain-journal" },
    { name: "Seizure and Abnormal Activity Alerts", icon: <AlertTriangle className="w-6 h-6" />, path: "/seizure-alerts" },
  ];

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      
      {/* Hero Image */}
      <div className="relative w-full h-[500px] sm:h-[600px] md:h-[700px]">
        <img src={mentalImage}
          alt="CogniSync"
          className="w-full h-full object-cover opacity-70"
        />
        <h1 className="absolute inset-0 flex items-center justify-center text-4xl sm:text-5xl md:text-6xl font-bold text-white drop-shadow-lg">
          CogniSync
        </h1>
      </div>

      {/* Description */}
      <div className="p-6 text-center max-w-3xl mx-auto">
        <p className="text-lg text-gray-300">
          CogniSync is your AI-powered EEG companion for mental health insights, focus tracking,
          fatigue monitoring, sleep analysis, meditation guidance, brain-controlled games, mood recognition,
          journaling, and seizure detection — all in one place.
        </p>
      </div>

      {/* Module Buttons */}
      <div className="p-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 max-w-5xl mx-auto">
        {modules.map((mod, index) => (
          <Link
            key={index}
            to={mod.path}
            className="bg-gray-900 hover:bg-gray-800 rounded-xl p-4 flex flex-col items-center justify-center transition"
          >
            <div className="mb-2">{mod.icon}</div>
            <span className="text-center text-sm font-medium">{mod.name}</span>
          </Link>
        ))}
      </div>
    </div>
  );
}
