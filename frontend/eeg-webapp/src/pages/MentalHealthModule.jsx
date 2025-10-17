// MentalHealthPage.jsx
// This component contains the core UI and logic for the mental health detection page.
// It is designed to be rendered as a standalone page by react-router-dom.

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import FFT from "fft.js";
import { Link } from "react-router-dom"; // Import Link for the back button

// === Configuration ===
const SAMPLE_RATE_DEFAULT = 250; // Hz
const WINDOW_SEC = 2; // seconds for PSD window
const WINDOW_SIZE_DEFAULT = SAMPLE_RATE_DEFAULT * WINDOW_SEC;

const BANDS = [
  { key: "delta", label: "Delta (0.5-4)", lo: 0.5, hi: 4 },
  { key: "theta", label: "Theta (4-8)", lo: 4, hi: 8 },
  { key: "alpha", label: "Alpha (8-12)", lo: 8, hi: 12 },
  { key: "beta", label: "Beta (12-30)", lo: 12, hi: 30 },
];

// --- No TASKS array needed after task selection panel is removed. ---

// simple localStorage helper
function useLocalStorage(key, initial) {
  const [state, setState] = useState(() => {
    try {
      const v = localStorage.getItem(key);
      return v ? JSON.parse(v) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(state));
    } catch {}
  }, [key, state]);
  return [state, setState];
}

// --- DSP helpers ---
function hannWindow(N) {
  const w = new Array(N);
  for (let i = 0; i < N; i++)
    w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1));
  return w;
}

function welchBandPowers(samples, fs) {
  const N = samples.length;
  if (N <= 0) return { delta: 0, theta: 0, alpha: 0, beta: 0 };
  const f = new FFT(N);
  const hann = hannWindow(N);
  const complexIn = f.createComplexArray();
  for (let i = 0; i < N; i++) {
    complexIn[2 * i] = samples[i] * hann[i];
    complexIn[2 * i + 1] = 0;
  }
  const out = f.createComplexArray();
  f.transform(out, complexIn);
  const psd = new Array(Math.floor(N / 2)).fill(0);
  for (let k = 0; k < psd.length; k++) {
    const re = out[2 * k],
      im = out[2 * k + 1];
    psd[k] = (re * re + im * im) / (fs * N);
  }
  const df = fs / N;
  const bandPower = {};
  BANDS.forEach(({ key, lo, hi }) => {
    const kLo = Math.max(1, Math.floor(lo / df));
    const kHi = Math.min(psd.length - 1, Math.ceil(hi / df));
    let sum = 0;
    for (let k = kLo; k <= kHi; k++) sum += psd[k];
    bandPower[key] = sum;
  });
  return bandPower;
}

function computeEngagement(p) {
  const beta = p.beta || 0,
    alpha = p.alpha || 0,
    theta = p.theta || 0;
  const denom = alpha + theta || 1e-6;
  let e = beta / denom;
  // heuristic scaling
  e = Math.max(0, Math.min(100, e * 100));
  return e;
}

// --- Web Serial helper ---
function useSerial(onSample, onStatus) {
  const portRef = useRef(null);
  const readerRef = useRef(null);

  async function connectSerial() {
    try {
      if (!("serial" in navigator))
        throw new Error("Web Serial API not available in this browser.");
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      onStatus && onStatus({ connected: true, message: "Serial open" });
      portRef.current = port;
      const decoder = new TextDecoderStream();
      const readableStreamClosed = port.readable.pipeTo(decoder.writable);
      const reader = decoder.readable.getReader();
      readerRef.current = reader;
      let buf = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += value;
        let idx;
        while ((idx = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line) continue;
          const parts = line.split(/[\t, ]+/).filter(Boolean);
          const val = Number(parts[parts.length - 1]);
          if (!Number.isNaN(val)) onSample(val);
        }
      }
    } catch (e) {
      console.error(e);
      onStatus &&
        onStatus({ connected: false, message: e.message || "Serial error" });
    }
  }

  async function disconnectSerial() {
    try {
      if (readerRef.current) {
        try {
          await readerRef.current.cancel();
        } catch {}
        readerRef.current = null;
      }
      if (portRef.current) {
        try {
          await portRef.current.close();
        } catch {}
        portRef.current = null;
      }
      onStatus && onStatus({ connected: false, message: "Serial closed" });
    } catch (e) {
      onStatus && onStatus({ connected: false, message: "Close error" });
    }
  }

  return { connectSerial, disconnectSerial };
}

// --- Mental Health Page Component ---
export default function MentalHealthPage() {
  // states
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Not connected");
  const [fs, setFs] = useState(SAMPLE_RATE_DEFAULT);
  const [windowSize, setWindowSize] = useState(WINDOW_SIZE_DEFAULT);
  // Removed selectedTask and autoStopSec from state
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [samples, setSamples] = useState([]);
  const [waveData, setWaveData] = useState([]);
  const [bandPowers, setBandPowers] = useState({
    delta: 0,
    theta: 0,
    alpha: 0,
    beta: 0,
  });
  const [engagement, setEngagement] = useState(0);
  const [results, setResults] = useState(null);
  const [history, setHistory] = useLocalStorage("mh_history", []);
  const timerRef = useRef(null);
  const [useCsv, setUseCsv] = useState(false);
  const csvBufferRef = useRef([]);

  // Default recording duration (60s) for when there's no task selection
  const RECORDING_DURATION = 60;

  // serial hooks
  const onSample = (v) => {
    if (!recording) return;
    setSamples((prev) => {
      const next = [...prev, v];
      if (next.length > windowSize) next.splice(0, next.length - windowSize);
      return next;
    });
  };
  const onStatus = ({ connected, message }) => {
    setConnected(!!connected);
    setStatusMsg(message || "");
  };
  const { connectSerial, disconnectSerial } = useSerial(onSample, onStatus);

  useEffect(() => {
    setWindowSize(fs * WINDOW_SEC);
  }, [fs]);

  // CSV playback support
  const onCsvUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const text = await file.text();
    const lines = text.split(/\r?\n/).filter(Boolean);
    const values = lines
      .map((l) => {
        const p = l.split(/[\t, ]+/).filter(Boolean);
        const v = Number(p[p.length - 1]);
        return Number.isNaN(v) ? null : v;
      })
      .filter((v) => v !== null);
    csvBufferRef.current = values;
    alert("CSV loaded: " + values.length + " samples");
  };

  // when samples change compute features
  useEffect(() => {
    if (samples.length === 0) return;
    const chart = samples
      .slice(-Math.min(samples.length, fs * 4))
      .map((v, i) => ({ t: i / fs, v }));
    setWaveData(chart);
    if (samples.length >= windowSize) {
      const bp = welchBandPowers(samples.slice(-windowSize), fs);
      setBandPowers(bp);
      setEngagement(computeEngagement(bp));
    }
  }, [samples, fs, windowSize]);

  // recording controls
  const startRecording = async () => {
    setResults(null);
    setSamples([]);
    setElapsed(0);
    setRecording(true);
    // if CSV play loaded, push to samples periodically
    if (useCsv && csvBufferRef.current.length) {
      let idx = 0;
      const step = Math.max(1, Math.floor(fs / 10));
      const loop = () => {
        if (!recording) return; // stopped
        const batch = csvBufferRef.current.slice(idx, idx + step);
        idx += step;
        setSamples((prev) => {
          const next = [...prev, ...batch];
          if (next.length > windowSize)
            next.splice(0, next.length - windowSize);
          return next;
        });
        if (idx < csvBufferRef.current.length && recording)
          setTimeout(loop, 100);
      };
      setTimeout(loop, 100);
    }

    timerRef.current = setInterval(() => {
      setElapsed((prev) => {
        const nxt = prev + 1;
        if (nxt >= RECORDING_DURATION) stopRecording(true);
        return nxt;
      });
    }, 1000);
  };
  const stopRecording = (auto = false) => {
    setRecording(false);
    clearInterval(timerRef.current);
    // compute lightweight heuristic result
    const bp = bandPowers;
    const e = engagement;
    const stress = Math.max(
      0,
      Math.min(100, 100 - e + (bp.theta * 10) / (bp.alpha + 1e-6))
    );
    const mood =
      e > 60 && bp.alpha > bp.theta
        ? "Calm / Positive"
        : stress > 60
        ? "Stressed"
        : "Neutral";
    const res = {
      when: Date.now(),
      task: "Manual Session", // Task name is now a static string
      duration: elapsed,
      bandPowers: bp,
      engagement: e,
      stress: Math.round(stress),
      mood,
      auto,
    };
    setResults(res);
    setHistory((prev) => [res, ...prev].slice(0, 200));
  };

  return (
    <div className="p-6 bg-gray-950 text-white min-h-screen">
      <div className="flex items-center gap-4 mb-8">
        <Link to="/" className="text-gray-400 hover:text-white transition">
          <span className="text-xl">←</span>
        </Link>
        <h1 className="text-2xl font-bold">Mental Health Detection</h1>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <Card title="Device & Setup">
            <div className="flex items-center gap-4 mb-4">
              <button
                onClick={connectSerial}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
              >
                Connect Serial
              </button>
              <button
                onClick={disconnectSerial}
                className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
              >
                Disconnect
              </button>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={useCsv}
                  onChange={(e) => setUseCsv(e.target.checked)}
                  className="form-checkbox text-blue-600"
                />
                CSV playback
              </label>
              {useCsv && (
                <input
                  type="file"
                  accept=".csv,.txt"
                  onChange={onCsvUpload}
                  className="text-sm"
                />
              )}
            </div>
            <div className="text-sm text-gray-400">
              <b>Status:</b> {connected ? "Connected" : "Not connected"} —{" "}
              {statusMsg}
            </div>
            <div className="mt-2 text-sm text-gray-400">
              <b>Electrode:</b> Fp1 (active) | Ref: A1
            </div>
          </Card>

          <Card title="Calibration & Signal Check" className="mt-6">
            <CalibrationPanel
              samples={samples}
              fs={fs}
              onGood={() => {
                alert("Signal looks good");
              }}
            />
          </Card>

          <Card title="Recording" className="mt-6">
            <div className="text-sm text-gray-400 mb-4">
              Click "Start Recording" to begin a {RECORDING_DURATION}s EEG
              session.
            </div>
            <div className="mt-4 flex items-center gap-4">
              {!recording ? (
                <button
                  onClick={startRecording}
                  className="px-6 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition disabled:opacity-50"
                  disabled={!connected && !useCsv}
                >
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={() => stopRecording(false)}
                  className="px-6 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 transition"
                >
                  Stop
                </button>
              )}
              <span className="text-gray-300">
                Elapsed: {elapsed}s / {RECORDING_DURATION}s
              </span>
            </div>
          </Card>

          <Card title="Live Waveform" className="mt-6">
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={waveData}
                  margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                  <XAxis
                    dataKey="t"
                    tickFormatter={(v) => v.toFixed(1)}
                    label={{
                      value: "Time (s)",
                      position: "insideBottomRight",
                      offset: -5,
                      fill: "#cbd5e0",
                    }}
                    stroke="#cbd5e0"
                  />
                  <YAxis width={60} stroke="#cbd5e0" />
                  <Tooltip
                    formatter={(v) =>
                      typeof v === "number" ? v.toFixed(2) : v
                    }
                    contentStyle={{
                      backgroundColor: "#2d3748",
                      border: "none",
                    }}
                    labelStyle={{ color: "#cbd5e0" }}
                    itemStyle={{ color: "#cbd5e0" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="v"
                    stroke="#4299e1"
                    dot={false}
                    strokeWidth={1.5}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>

          <Card title="Processing & Results" className="mt-6">
            {results ? (
              <div>
                <div className="mb-4">
                  <b className="text-lg text-white">Predicted State:</b>{" "}
                  <span className="text-xl font-semibold text-blue-400">
                    {results.mood}
                  </span>{" "}
                  (Stress {results.stress})
                </div>
                <div className="flex flex-col md:flex-row gap-4 items-center">
                  <div className="w-full md:w-1/2 h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={BANDS.map((b) => ({
                          name: b.key.toUpperCase(),
                          power: bandPowers[b.key] || 0,
                        }))}
                        margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#4a5568" />
                        <XAxis dataKey="name" stroke="#cbd5e0" />
                        <YAxis stroke="#cbd5e0" />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "#2d3748",
                            border: "none",
                          }}
                          labelStyle={{ color: "#cbd5e0" }}
                          itemStyle={{ color: "#cbd5e0" }}
                        />
                        <Bar dataKey="power" fill="#a0aec0" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="w-full md:w-1/2">
                    <div className="mb-2 text-gray-300">
                      <b>Engagement</b>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2.5">
                      <div
                        className="h-2.5 rounded-full"
                        style={{
                          width: Math.max(3, Math.min(100, engagement)) + "%",
                          backgroundColor:
                            engagement > 70
                              ? "#10b981"
                              : engagement > 40
                              ? "#f59e0b"
                              : "#ef4444",
                        }}
                      />
                    </div>
                    <div className="mt-4 text-sm text-gray-400">
                      <b>When:</b> {new Date(results.when).toLocaleString()}
                    </div>
                    <div className="mt-2">
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(
                            JSON.stringify(results, null, 2)
                          );
                          alert("Copied result JSON");
                        }}
                        className="px-4 py-2 bg-gray-800 text-sm text-gray-300 rounded-lg hover:bg-gray-700 transition"
                      >
                        Copy JSON
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-500">
                No results yet. Run a recording to see session summary. (ML
                model integration point)
              </div>
            )}
          </Card>
        </div>

        <div className="lg:col-span-1">
          <Card title="Quick Metrics">
            <div className="mb-2">
              <b className="text-gray-300">Alpha</b>
            </div>
            <progress
              value={Math.min(
                100,
                Math.log10((bandPowers.alpha || 0) + 1) * 40
              )}
              max={100}
              className="w-full h-2 rounded-full"
            />
            <div className="mt-4 mb-2">
              <b className="text-gray-300">Beta</b>
            </div>
            <progress
              value={Math.min(100, Math.log10((bandPowers.beta || 0) + 1) * 40)}
              max={100}
              className="w-full h-2 rounded-full"
            />
            <div className="mt-4 mb-2">
              <b className="text-gray-300">Theta</b>
            </div>
            <progress
              value={Math.min(
                100,
                Math.log10((bandPowers.theta || 0) + 1) * 40
              )}
              max={100}
              className="w-full h-2 rounded-full"
            />
            <div className="mt-6">
              <b className="text-gray-300">Engagement</b>
            </div>
            <div className="text-3xl font-bold text-blue-400 mt-1">
              {Math.round(engagement)}%
            </div>
          </Card>

          <Card title="History" className="mt-6">
            <div className="max-h-96 overflow-y-auto">
              {history.length ? (
                history.map((h, i) => (
                  <div
                    key={i}
                    className="border-b border-gray-700 p-2 last:border-b-0"
                  >
                    <div className="text-sm">
                      <b className="text-white">{h.task}</b> —{" "}
                      <span className="text-gray-400">
                        {new Date(h.when).toLocaleString()}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 mt-1">
                      Mood: {h.mood} | Stress: {h.stress} | Engagement:{" "}
                      {Math.round(h.engagement)}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500">No sessions yet.</div>
              )}
            </div>
            <div className="mt-4">
              <button
                onClick={() => {
                  setHistory([]);
                  alert("History cleared");
                }}
                className="px-4 py-2 bg-gray-800 text-sm text-gray-300 rounded-lg hover:bg-gray-700 transition"
              >
                Clear History
              </button>
            </div>
          </Card>

          <Card title="Settings" className="mt-6">
            <div className="mb-4">
              <label className="text-sm text-gray-300">
                Sampling rate (Hz):{" "}
                <input
                  type="number"
                  value={fs}
                  onChange={(e) =>
                    setFs(Number(e.target.value) || SAMPLE_RATE_DEFAULT)
                  }
                  className="w-20 px-2 py-1 bg-gray-800 border border-gray-700 rounded-md text-gray-300"
                />
              </label>
            </div>
            <div className="text-xs text-gray-500">
              Notch filter 50Hz applied in preprocessing. Data stored locally by
              default.
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

// Reusable card component for layout
function Card({ title, children, className }) {
  return (
    <div
      className={`bg-gray-900 border border-gray-800 rounded-xl p-6 ${className}`}
    >
      {title && (
        <h2 className="text-lg font-semibold text-white mb-4">{title}</h2>
      )}
      <div>{children}</div>
    </div>
  );
}

function CalibrationPanel({ samples, fs, onGood }) {
  const rmsVal = useMemo(() => {
    if (!samples.length) return 0;
    const s = samples.slice(-Math.min(samples.length, fs));
    const sum = s.reduce((a, v) => a + v * v, 0);
    return Math.sqrt(sum / s.length);
  }, [samples, fs]);

  const signalStatus = useMemo(() => {
    if (rmsVal < 0.01)
      return { text: "No signal / check electrode", color: "text-red-400" };
    if (rmsVal > 500)
      return {
        text: "Too large amplitude — possible artifact",
        color: "text-yellow-400",
      };
    return { text: "Signal OK", color: "text-green-400" };
  }, [rmsVal]);

  return (
    <div>
      <div className="text-sm text-gray-300 mb-2">
        Signal RMS: <span className="font-mono">{rmsVal.toFixed(2)}</span>
      </div>
      <div
        className={`p-4 bg-gray-800 rounded-lg flex items-center justify-center h-20 ${signalStatus.color}`}
      >
        {signalStatus.text}
      </div>
      <div className="mt-4">
        <button
          onClick={onGood}
          className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition"
        >
          Mark OK
        </button>
      </div>
    </div>
  );
}
