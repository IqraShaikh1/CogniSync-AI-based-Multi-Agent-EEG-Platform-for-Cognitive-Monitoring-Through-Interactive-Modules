
// pages/MainPage.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  HeartPulse, Focus, BatteryCharging, Moon, Brain,
  Gamepad2, Smile, NotebookText, AlertTriangle, Server,
} from "lucide-react";
import mentalImage from "../assets/hero.png";

const MANAGER_URL = "http://127.0.0.1:9000";
const POLL_MS     = 4000;

const modules = [
  { key: "mental-health",    name: "Mental Health Detection",            icon: <HeartPulse className="w-6 h-6" />,     path: "/mental-health"    },
  { key: "focus-tracking",   name: "Focus and Attention Tracking",       icon: <Focus className="w-6 h-6" />,           path: "/focus-tracking"   },
  { key: "fatigue",          name: "Fatigue Detection",                  icon: <BatteryCharging className="w-6 h-6" />, path: "/fatigue"          },
  { key: "sleep-monitoring", name: "Sleep Stage Monitoring",             icon: <Moon className="w-6 h-6" />,            path: "/sleep-monitoring" },
  { key: "meditation",       name: "Meditation Assistant",               icon: <Brain className="w-6 h-6" />,           path: "/meditation"       },
  { key: "brain-games",      name: "Brain-Controlled Games",             icon: <Gamepad2 className="w-6 h-6" />,        path: "/brain-games"      },
  { key: "mood-emotion",     name: "Mood and Emotion Recognition",       icon: <Smile className="w-6 h-6" />,           path: "/mood-emotion"     },
  { key: "brain-journal",    name: "Daily Brain Journal",                icon: <NotebookText className="w-6 h-6" />,    path: "/brain-journal"    },
  { key: "seizure-alerts",   name: "Seizure and Abnormal Activity Alerts", icon: <AlertTriangle className="w-6 h-6" />, path: "/seizure-alerts"   },
];

// Small status dot shown on each module card
function StatusDot({ state }) {
  if (!state || state === "stopped")
    return <span className="w-2 h-2 rounded-full bg-gray-600 flex-shrink-0" title="Stopped" />;
  if (state === "running")
    return <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse flex-shrink-0" title="Running" />;
  if (state === "starting")
    return <span className="w-2 h-2 rounded-full bg-yellow-400 animate-pulse flex-shrink-0" title="Starting…" />;
  if (state === "missing")
    return <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" title="Script missing" />;
  return null;
}

export default function MainPage() {
  const [moduleStatus,  setModuleStatus]  = useState({});
  const [managerOnline, setManagerOnline] = useState(false);

  // Poll manager for live status
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res  = await fetch(`${MANAGER_URL}/manager/status`, { signal: AbortSignal.timeout(2000) });
        const data = await res.json();
        if (data.modules) {
          setModuleStatus(data.modules);
          setManagerOnline(true);
        }
      } catch {
        setManagerOnline(false);
      }
    };

    fetchStatus();
    const id = setInterval(fetchStatus, POLL_MS);
    return () => clearInterval(id);
  }, []);

  const runningCount = Object.values(moduleStatus).filter(m => m?.state === "running").length;

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* Hero */}
      <div className="relative w-full h-[500px] sm:h-[600px] md:h-[700px]">
        <img
          src={mentalImage}
          alt="CogniSync"
          className="w-full h-full object-cover opacity-70"
        />
        <h1 className="absolute inset-0 flex items-center justify-center
          text-4xl sm:text-5xl md:text-6xl font-bold text-white drop-shadow-lg">
          CogniSync
        </h1>
      </div>

      {/* Description */}
      <div className="p-6 text-center max-w-3xl mx-auto">
        <p className="text-lg text-gray-300">
          CogniSync is your AI-powered EEG companion for mental health insights, focus tracking,
          fatigue monitoring, sleep analysis, meditation guidance, brain-controlled games, mood
          recognition, journaling, and seizure detection — all in one place.
        </p>
      </div>

      {/* Module Manager banner */}
      <div className="max-w-5xl mx-auto px-6 mb-2">
        <Link
          to="/module-manager"
          className="flex items-center justify-between gap-4 bg-gray-900 hover:bg-gray-800
            border border-gray-700 hover:border-gray-500 rounded-xl px-5 py-4 transition group"
        >
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-600/20 rounded-lg border border-blue-500/30">
              <Server className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <div className="font-semibold text-sm text-white">Module Manager</div>
              <div className="text-xs text-gray-400 mt-0.5">
                Start, stop and monitor all backend module servers
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 flex-shrink-0">
            {managerOnline ? (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-green-400
                bg-green-900/30 border border-green-800/50 px-2.5 py-1 rounded-full">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                {runningCount} running
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs font-semibold text-red-400
                bg-red-900/20 border border-red-800/40 px-2.5 py-1 rounded-full">
                <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                Offline
              </span>
            )}
            <span className="text-gray-500 group-hover:text-white transition text-sm">→</span>
          </div>
        </Link>
      </div>

      {/* Module Grid */}
      <div className="p-6 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 max-w-5xl mx-auto">
        {modules.map((mod) => {
          const status = moduleStatus[mod.key];
          const state  = status?.state || (managerOnline ? "stopped" : null);

          return (
            <Link
              key={mod.key}
              to={mod.path}
              className="bg-gray-900 hover:bg-gray-800 rounded-xl p-4 flex flex-col items-center
                justify-center transition relative group border border-gray-800 hover:border-gray-600"
            >
              {/* Status dot — top-right corner */}
              {state && (
                <div className="absolute top-3 right-3">
                  <StatusDot state={state} />
                </div>
              )}

              <div className="mb-2">{mod.icon}</div>
              <span className="text-center text-sm font-medium">{mod.name}</span>

              {/* "Not running" nudge */}
              {managerOnline && state && state !== "running" && state !== "starting" && (
                <span className="mt-2 text-xs text-gray-600">backend stopped</span>
              )}
            </Link>
          );
        })}
      </div>
    </div>
  );
}