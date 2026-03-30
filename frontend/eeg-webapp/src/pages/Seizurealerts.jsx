import React, { useEffect, useRef, useState } from "react";
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
  Cell,
} from "recharts";
import jsPDF from "jspdf";

// ==============================
// CONSTANTS
// ==============================
const SAMPLE_RATE_DEFAULT = 250;
const WINDOW_SEC          = 2;
const WINDOW_SIZE_DEFAULT = SAMPLE_RATE_DEFAULT * WINDOW_SEC; // 500 samples
const API_BASE            = "http://127.0.0.1:8000";
const RECORDING_DURATION  = 120;
const API_CALL_INTERVAL   = 3000;

const BANDS = [
  { key: "alpha", label: "Alpha (8-12 Hz)",  lo: 8,  hi: 12, color: "#10B981" },
  { key: "beta",  label: "Beta (13-30 Hz)",  lo: 13, hi: 30, color: "#F59E0B" },
  { key: "theta", label: "Theta (4-7 Hz)",   lo: 4,  hi: 7,  color: "#3B82F6" },
];

const ALERT_CONFIG = {
  LOW:    { color: "#10B981", bg: "bg-green-900/30",  border: "border-green-500/50",  text: "text-green-400",  label: "🟢 LOW",    desc: "Normal EEG activity detected" },
  MEDIUM: { color: "#F59E0B", bg: "bg-yellow-900/30", border: "border-yellow-500/50", text: "text-yellow-400", label: "🟡 MEDIUM", desc: "Abnormal patterns observed — monitoring closely" },
  HIGH:   { color: "#EF4444", bg: "bg-red-900/30",    border: "border-red-500/50",    text: "text-red-400",    label: "🔴 HIGH",   desc: "SEIZURE ACTIVITY DETECTED — seek medical attention" },
};

// ==============================
// FOCUS FEATURE EXTRACTION
// ==============================
function computeFocusFeatures(samples, fs = 250) {
  const n = samples.length;
  if (n < 256) return null;

  const seg = samples.slice(-256);
  const N   = seg.length;

  const win = seg.map((v, i) => v * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1))));

  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let k = 0; k <= N / 2; k++) {
    let rk = 0, ik = 0;
    for (let i = 0; i < N; i++) {
      const angle = (2 * Math.PI * k * i) / N;
      rk += win[i] * Math.cos(angle);
      ik -= win[i] * Math.sin(angle);
    }
    re[k] = rk;
    im[k] = ik;
  }

  const psd   = [];
  const freqs = [];
  for (let k = 0; k <= N / 2; k++) {
    freqs.push((k * fs) / N);
    psd.push((re[k] * re[k] + im[k] * im[k]) / (N * N));
  }

  function trapz(band_lo, band_hi) {
    let power = 0;
    for (let k = 0; k < freqs.length - 1; k++) {
      if (freqs[k] >= band_lo && freqs[k + 1] <= band_hi) {
        power += 0.5 * (psd[k] + psd[k + 1]) * (freqs[k + 1] - freqs[k]);
      }
    }
    return power;
  }

  const alpha_p = trapz(8, 12);
  const beta_p  = trapz(13, 30);
  const theta_p = trapz(4, 7);

  const alpha_beta_ratio = alpha_p / (beta_p + 1e-10);
  const mean             = samples.reduce((a, x) => a + x, 0) / samples.length;
  const signal_variance  = samples.reduce((acc, v) => acc + (v - mean) ** 2, 0) / samples.length;
  const attention_index  = beta_p / (alpha_p + theta_p + 1e-10);

  return { alpha_power: alpha_p, beta_p, theta_p, alpha_beta_ratio, signal_variance, attention_index, alpha_p };
}

function computeBandPowers(samples, fs = 250) {
  const features = computeFocusFeatures(samples, fs);
  if (!features) return { alpha: 0, beta: 0, theta: 0 };
  return { alpha: features.alpha_p, beta: features.beta_p, theta: features.theta_p };
}

// ==============================
// API HELPERS
// ==============================
async function sendToApi(endpoint, payload) {
  try {
    const response = await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const err = await response.text();
      console.error(`API Error ${response.status}:`, err);
      return { error: `HTTP ${response.status}` };
    }
    return await response.json();
  } catch (err) {
    console.error("API Error:", err);
    return { error: err.message };
  }
}

// ==============================
// CSV UPLOAD HELPER
// ==============================
async function sendCsvToApi(file, windowSamples = 500) {
  try {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("window_samples", String(windowSamples));

    const response = await fetch(`${API_BASE}/predict_csv`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const err = await response.text();
      console.error(`CSV API Error ${response.status}:`, err);
      return { error: `HTTP ${response.status}: ${err}` };
    }
    return await response.json();
  } catch (err) {
    console.error("CSV API Error:", err);
    return { error: err.message };
  }
}

// ==============================
// PDF GENERATOR — CSV RESULT
// ==============================
function downloadCsvPDF(csvResult, filename) {
  const doc       = new jsPDF();
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const margin    = 20;
  let y           = 20;

  const addPageIfNeeded = (needed = 10) => {
    if (y + needed > pageHeight - 20) {
      doc.addPage();
      y = 20;
    }
  };

  // Title
  doc.setFontSize(20); doc.setFont(undefined, "bold");
  doc.text("EEG Seizure Detection Report — CSV Analysis", pageWidth / 2, y, { align: "center" }); y += 14;

  doc.setFontSize(10); doc.setFont(undefined, "normal");
  doc.text(`Generated: ${new Date().toLocaleString()}`, margin, y); y += 6;
  doc.text(`Source file: ${csvResult.csv_info?.filename || "N/A"}`, margin, y); y += 6;
  doc.text(`Total CSV rows: ${csvResult.csv_info?.total_rows?.toLocaleString() || "N/A"}`, margin, y); y += 6;
  doc.text(`Window size: ${csvResult.csv_info?.window_samples || 500} samples (2 s @ 250 Hz)`, margin, y); y += 12;

  doc.setDrawColor(200); doc.line(margin, y, pageWidth - margin, y); y += 10;

  // Final Decision
  const s = csvResult.summary;
  const isSeizure = s.final_decision === "SEIZURE DETECTED";
  doc.setFontSize(16); doc.setFont(undefined, "bold");
  doc.text("Final Decision", margin, y); y += 8;
  doc.setFontSize(20);
  doc.setTextColor(isSeizure ? 239 : 34, isSeizure ? 68 : 197, isSeizure ? 68 : 94);
  doc.text(s.final_decision, margin, y); doc.setTextColor(0); y += 14;

  // Summary stats box
  doc.setFontSize(13); doc.setFont(undefined, "bold");
  doc.text("Session Summary", margin, y); y += 8;
  doc.setFontSize(10); doc.setFont(undefined, "normal");

  const stats = [
    ["Total windows analyzed", String(s.total_windows)],
    ["Seizure windows",        String(s.seizure_windows)],
    ["No Seizure windows",     String(s.normal_windows)],
    ["Seizure ratio",          (s.seizure_ratio * 100).toFixed(2) + "%"],
    ["Avg seizure probability",  (s.avg_seizure_prob * 100).toFixed(4) + "%"],
  ];
  stats.forEach(([k, v]) => {
    doc.text(k + ":", margin + 5, y);
    doc.text(v, margin + 100, y);
    y += 6;
  });
  y += 8;

  // Decision rule note
  doc.setFontSize(9); doc.setTextColor(120);
  doc.text("* Decision rule: SEIZURE DETECTED when seizure ratio ≥ 30%", margin, y);
  doc.setTextColor(0); y += 12;

  doc.setDrawColor(200); doc.line(margin, y, pageWidth - margin, y); y += 10;

  // Per-window table header
  doc.setFontSize(13); doc.setFont(undefined, "bold");
  doc.text("Per-Window Predictions", margin, y); y += 8;

  doc.setFontSize(9); doc.setFont(undefined, "bold");
  const col = [margin, margin + 35, margin + 75, margin + 110];
  doc.text("Window #",      col[0], y);
  doc.text("Timestamp (ms)", col[1], y);
  doc.text("Prediction",    col[2], y);
  doc.text("Seizure Prob",  col[3], y);
  y += 5;
  doc.setDrawColor(180); doc.line(margin, y, pageWidth - margin, y); y += 4;

  doc.setFont(undefined, "normal");
  csvResult.window_results.forEach((row, idx) => {
    addPageIfNeeded(7);
    const isS = row.pred_label === 1;
    if (isS) doc.setTextColor(220, 50, 50);
    doc.text(String(idx + 1),                          col[0], y);
    doc.text(String(row.timestamp_ms),                 col[1], y);
    doc.text(row.label,                                col[2], y);
    doc.text((row.seizure_prob * 100).toFixed(4) + "%", col[3], y);
    if (isS) doc.setTextColor(0);
    y += 6;
  });

  // Footer
  const totalPages = doc.internal.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    doc.setFontSize(8); doc.setTextColor(128);
    doc.text(
      `EEG Seizure Detection System  |  Page ${p} of ${totalPages}`,
      pageWidth / 2, pageHeight - 10, { align: "center" }
    );
    doc.setTextColor(0);
  }

  const outName = `Seizure_CSV_Report_${new Date().toISOString().split("T")[0]}.pdf`;
  doc.save(outName);
}

// ==============================
// MAIN COMPONENT
// ==============================
export default function SeizureAlertsPage() {
  // ---- existing live-recording state ----
  const [connected,         setConnected]         = useState(false);
  const [statusMsg,         setStatusMsg]          = useState("Not connected");
  const [fs,                setFs]                 = useState(SAMPLE_RATE_DEFAULT);
  const [windowSize,        setWindowSize]         = useState(WINDOW_SIZE_DEFAULT);
  const [recording,         setRecording]          = useState(false);
  const [elapsed,           setElapsed]            = useState(0);
  const [samples,           setSamples]            = useState([]);
  const [sampleCount,       setSampleCount]        = useState(0);
  const [waveData,          setWaveData]           = useState([]);
  const [bandPowers,        setBandPowers]         = useState({ alpha: 0, beta: 0, theta: 0 });
  const [prediction,        setPrediction]         = useState(null);
  const [label,             setLabel]              = useState(null);
  const [seizureProb,       setSeizureProb]        = useState(0);
  const [alertLevel,        setAlertLevel]         = useState("LOW");
  const [history,           setHistory]            = useState([]);
  const [apiStatus,         setApiStatus]          = useState("idle");
  const [processingCount,   setProcessingCount]    = useState(0);
  const [recordingComplete, setRecordingComplete]  = useState(false);
  const [finalResult,       setFinalResult]        = useState(null);
  const [alarmActive,       setAlarmActive]        = useState(false);

  // ---- NEW: CSV upload state ----
  const [csvFile,           setCsvFile]            = useState(null);
  const [csvStatus,         setCsvStatus]          = useState("idle"); // idle | loading | done | error
  const [csvResult,         setCsvResult]          = useState(null);
  const [csvError,          setCsvError]           = useState("");
  const [csvWindowSamples,  setCsvWindowSamples]   = useState(500);
  const csvInputRef = useRef(null);

  const sessionTrackingRef = useRef([]);
  const timerRef           = useRef(null);
  const portRef            = useRef(null);
  const readerRef          = useRef(null);
  const recordingRef       = useRef(false);
  const samplesBufferRef   = useRef([]);
  const lastApiCallRef     = useRef(0);
  const alarmTimeoutRef    = useRef(null);

  useEffect(() => { setWindowSize(fs * WINDOW_SEC); }, [fs]);
  useEffect(() => { recordingRef.current = recording; }, [recording]);

  useEffect(() => {
    if (alarmActive) {
      clearTimeout(alarmTimeoutRef.current);
      alarmTimeoutRef.current = setTimeout(() => setAlarmActive(false), 8000);
    }
  }, [alarmActive]);

  // ------------------------------------------------------------------
  // Main processing loop
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!recording || samples.length < windowSize) return;
    const now = Date.now();
    if (now - lastApiCallRef.current < API_CALL_INTERVAL) return;

    const processData = async () => {
      try {
        const window = samples.slice(-windowSize);
        const features = computeFocusFeatures(window, fs);
        if (!features) return;

        const bp = { alpha: features.alpha_p, beta: features.beta_p, theta: features.theta_p };
        setBandPowers(bp);

        const waveWindow = Math.min(samples.length, fs * 2);
        setWaveData(
          samples
            .slice(-waveWindow)
            .map((v, i) => ({ t: (i / fs).toFixed(2), v: v.toFixed(2) }))
        );

        setApiStatus("fetching");
        lastApiCallRef.current = now;

        const apiRes = await sendToApi("/predict", {
          alpha_power:      features.alpha_p,
          beta_power:       features.beta_p,
          theta_power:      features.theta_p,
          alpha_beta_ratio: features.alpha_beta_ratio,
          signal_variance:  features.signal_variance,
          attention_index:  features.attention_index,
        });

        if (apiRes && !apiRes.error) {
          setPrediction(apiRes.prediction);
          setLabel(apiRes.label);
          setSeizureProb(apiRes.seizure_prob || 0);
          setAlertLevel(apiRes.alert_level || "LOW");
          setApiStatus("success");
          setProcessingCount(c => c + 1);

          if (apiRes.alert_level === "HIGH") setAlarmActive(true);

          const entry = {
            timestamp:   Date.now(),
            elapsed,
            label:       apiRes.label,
            prediction:  apiRes.prediction,
            seizureProb: apiRes.seizure_prob || 0,
            alertLevel:  apiRes.alert_level || "LOW",
            bandPowers:  { ...bp },
            features:    { ...features },
          };
          sessionTrackingRef.current.push(entry);
        } else {
          setApiStatus("error");
        }
      } catch (err) {
        setApiStatus("error");
      }
    };

    processData();
  }, [samples.length, recording, windowSize, fs, elapsed]);

  // ------------------------------------------------------------------
  // Serial Connection
  // ------------------------------------------------------------------
  const connectSerial = async () => {
    try {
      if (!("serial" in navigator)) throw new Error("Web Serial API not supported in this browser");
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      portRef.current = port;
      setConnected(true);
      setStatusMsg("Serial connected");

      const decoder    = new TextDecoderStream();
      port.readable.pipeTo(decoder.writable);
      const reader     = decoder.readable.getReader();
      readerRef.current = reader;

      let lineBuffer       = "";
      let validSampleCount = 0;

      const readLoop = async () => {
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            lineBuffer += value;
            let newlineIdx;
            while ((newlineIdx = lineBuffer.indexOf("\n")) >= 0) {
              const line = lineBuffer.slice(0, newlineIdx).trim();
              lineBuffer = lineBuffer.slice(newlineIdx + 1);
              if (!line) continue;
              const parts = line.split(/[\t,\s]+/).filter(Boolean);
              let val = parseFloat(parts[parts.length - 1]);
              if (isNaN(val)) { const m = line.match(/[-+]?\d*\.?\d+/); if (m) val = parseFloat(m[0]); }
              if (!isNaN(val) && isFinite(val)) {
                validSampleCount++;
                setSampleCount(validSampleCount);
                if (recordingRef.current) {
                  samplesBufferRef.current.push(val);
                  if (samplesBufferRef.current.length > windowSize * 2)
                    samplesBufferRef.current = samplesBufferRef.current.slice(-windowSize);
                  if (samplesBufferRef.current.length % 50 === 0)
                    setSamples([...samplesBufferRef.current]);
                }
              }
            }
          }
        } catch (e) { console.error("Read loop error:", e); }
      };
      readLoop();
    } catch (e) {
      setConnected(false);
      setStatusMsg(e.message);
    }
  };

  const disconnectSerial = async () => {
    try {
      setRecording(false);
      clearInterval(timerRef.current);
      if (readerRef.current) { try { await readerRef.current.cancel().catch(() => {}); } catch (_) {} readerRef.current = null; }
      if (portRef.current) { try { await Promise.race([portRef.current.close(), new Promise(r => setTimeout(r, 1000))]); } catch (_) {} portRef.current = null; }
      setConnected(false); setStatusMsg("Disconnected");
      setSamples([]); setSampleCount(0); setPrediction(null);
    } catch (_) {
      setConnected(false); setStatusMsg("Disconnected");
      portRef.current = null; readerRef.current = null;
    }
  };

  // ------------------------------------------------------------------
  // Recording Controls
  // ------------------------------------------------------------------
  const startRecording = () => {
    samplesBufferRef.current = [];
    sessionTrackingRef.current = [];
    lastApiCallRef.current = 0;
    setSamples([]); setSampleCount(0); setElapsed(0);
    setPrediction(null); setLabel(null); setSeizureProb(0);
    setAlertLevel("LOW"); setProcessingCount(0);
    setRecording(true); setApiStatus("idle");
    setRecordingComplete(false); setFinalResult(null);
    setAlarmActive(false);

    timerRef.current = setInterval(() => {
      setElapsed(prev => {
        const next = prev + 1;
        if (next >= RECORDING_DURATION) { setTimeout(() => stopRecording(), 100); return RECORDING_DURATION; }
        return next;
      });
    }, 1000);
  };

  const stopRecording = () => {
    setRecording(false);
    clearInterval(timerRef.current);
    const trackingData = sessionTrackingRef.current;

    if (trackingData.length > 0) {
      const seizureEntries = trackingData.filter(e => e.prediction === 1);
      const avgSeizureProb = trackingData.reduce((s, e) => s + e.seizureProb, 0) / trackingData.length;
      const maxSeizureProb = Math.max(...trackingData.map(e => e.seizureProb));
      const seizureRatio   = seizureEntries.length / trackingData.length;
      const dominantLabel  = seizureRatio >= 0.30 ? "SEIZURE DETECTED" : "NORMAL";
      const finalAlert     = maxSeizureProb >= 0.75 ? "HIGH" : maxSeizureProb >= 0.40 ? "MEDIUM" : "LOW";

      const avgBandPowers = {
        alpha: trackingData.reduce((s, e) => s + (e.bandPowers.alpha || 0), 0) / trackingData.length,
        beta:  trackingData.reduce((s, e) => s + (e.bandPowers.beta  || 0), 0) / trackingData.length,
        theta: trackingData.reduce((s, e) => s + (e.bandPowers.theta || 0), 0) / trackingData.length,
      };

      const report = {
        dominantLabel, finalAlert, avgSeizureProb, maxSeizureProb, seizureRatio,
        seizureWindows: seizureEntries.length,
        totalWindows:   trackingData.length,
        avgBandPowers,
        timestamp:   new Date().toLocaleTimeString(),
        sampleCount: samplesBufferRef.current.length,
        duration:    elapsed,
        trackingData,
      };

      setFinalResult(report);
      setHistory(prev => [report, ...prev].slice(0, 50));
    }

    if (elapsed >= RECORDING_DURATION - 5) setRecordingComplete(true);
  };

  // ------------------------------------------------------------------
  // API Test
  // ------------------------------------------------------------------
  const testApi = async () => {
    try {
      const res  = await fetch(`${API_BASE}/test`);
      const data = await res.json();
      if (data.status === "success") {
        alert("✅ Seizure API is working!\n\n" + JSON.stringify(data, null, 2));
        setApiStatus("success");
      } else {
        alert("❌ API Error"); setApiStatus("error");
      }
    } catch (_) {
      alert("❌ Cannot connect to Flask API!\n\nMake sure the server is running:\npython seizure_app.py");
      setApiStatus("error");
    }
  };

  // ------------------------------------------------------------------
  // PDF — Live session
  // ------------------------------------------------------------------
  const downloadPDF = () => {
    if (!finalResult) return;
    const doc       = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin    = 20;
    let y           = 20;

    doc.setFontSize(20); doc.setFont(undefined, "bold");
    doc.text("EEG Seizure Detection Report", pageWidth / 2, y, { align: "center" }); y += 15;

    doc.setFontSize(10); doc.setFont(undefined, "normal");
    doc.text(`Session Date: ${new Date().toLocaleDateString()}`, margin, y);     y += 6;
    doc.text(`Session Time: ${finalResult.timestamp}`, margin, y);              y += 6;
    doc.text(`Duration: ${Math.floor(finalResult.duration / 60)}:${(finalResult.duration % 60).toString().padStart(2, "0")} minutes`, margin, y); y += 12;

    doc.setDrawColor(200); doc.line(margin, y, pageWidth - margin, y); y += 10;

    doc.setFontSize(14); doc.setFont(undefined, "bold");
    doc.text("Final Decision", margin, y); y += 8;
    doc.setFontSize(18);
    doc.setTextColor(finalResult.dominantLabel === "SEIZURE DETECTED" ? 239 : 34,
                     finalResult.dominantLabel === "SEIZURE DETECTED" ? 68  : 197,
                     finalResult.dominantLabel === "SEIZURE DETECTED" ? 68  : 94);
    doc.text(finalResult.dominantLabel, margin, y); doc.setTextColor(0); y += 12;

    doc.setFontSize(14); doc.setFont(undefined, "bold");
    doc.text("Alert Level", margin, y); y += 8;
    doc.setFontSize(12); doc.setFont(undefined, "normal");
    doc.text(finalResult.finalAlert, margin, y); y += 12;

    doc.setFontSize(14); doc.setFont(undefined, "bold");
    doc.text("Seizure Probability Stats", margin, y); y += 8;
    doc.setFontSize(10); doc.setFont(undefined, "normal");
    doc.text(`Average seizure probability : ${(finalResult.avgSeizureProb * 100).toFixed(1)}%`, margin + 5, y); y += 6;
    doc.text(`Peak seizure probability    : ${(finalResult.maxSeizureProb * 100).toFixed(1)}%`, margin + 5, y); y += 6;
    doc.text(`Seizure windows             : ${finalResult.seizureWindows} / ${finalResult.totalWindows}`, margin + 5, y); y += 6;
    doc.text(`Seizure window ratio        : ${(finalResult.seizureRatio * 100).toFixed(1)}%`, margin + 5, y); y += 12;

    doc.setFontSize(14); doc.setFont(undefined, "bold");
    doc.text("Average EEG Band Powers", margin, y); y += 8;
    doc.setFontSize(10); doc.setFont(undefined, "normal");
    Object.entries(finalResult.avgBandPowers).forEach(([band, val]) => {
      doc.text(`${band.charAt(0).toUpperCase() + band.slice(1)}: ${val.toFixed(8)}`, margin + 5, y); y += 6;
    }); y += 10;

    doc.setFontSize(14); doc.setFont(undefined, "bold");
    doc.text("Session Statistics", margin, y); y += 8;
    doc.setFontSize(10); doc.setFont(undefined, "normal");
    doc.text(`Total predictions : ${finalResult.totalWindows}`, margin + 5, y); y += 6;
    doc.text(`Total samples     : ${finalResult.sampleCount?.toLocaleString() || "N/A"}`, margin + 5, y); y += 6;
    doc.text(`Sample rate       : ${fs} Hz`, margin + 5, y);

    y = doc.internal.pageSize.getHeight() - 20;
    doc.setFontSize(8); doc.setTextColor(128);
    doc.text("Generated by EEG Seizure Detection & Monitoring System", pageWidth / 2, y, { align: "center" });

    const filename = `Seizure_Report_${new Date().toISOString().split("T")[0]}_${new Date().toTimeString().split(" ")[0].replace(/:/g, "-")}.pdf`;
    doc.save(filename);
  };

  // ------------------------------------------------------------------
  // NEW: CSV Handlers
  // ------------------------------------------------------------------
  const handleCsvFileChange = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setCsvFile(f);
    setCsvResult(null);
    setCsvError("");
    setCsvStatus("idle");
  };

  const handleCsvUpload = async () => {
    if (!csvFile) return;
    setCsvStatus("loading");
    setCsvError("");
    setCsvResult(null);

    const res = await sendCsvToApi(csvFile, csvWindowSamples);

    if (res.error) {
      setCsvStatus("error");
      setCsvError(res.error);
    } else {
      setCsvStatus("done");
      setCsvResult(res);
    }
  };

  const clearCsvUpload = () => {
    setCsvFile(null);
    setCsvResult(null);
    setCsvError("");
    setCsvStatus("idle");
    if (csvInputRef.current) csvInputRef.current.value = "";
  };

  // ------------------------------------------------------------------
  // Derived UI values
  // ------------------------------------------------------------------
  const alertCfg      = ALERT_CONFIG[alertLevel] || ALERT_CONFIG.LOW;
  const probPct       = (seizureProb * 100).toFixed(1);
  const bandChartData = BANDS.map(({ key, label: lbl, color }) => ({
    name:  lbl.split(" ")[0],
    value: parseFloat((bandPowers[key] || 0).toFixed(8)),
    color,
  }));

  // CSV chart data
  const csvChartData = csvResult
    ? csvResult.window_results.map((r, i) => ({
        window:      i + 1,
        seizureProb: parseFloat((r.seizure_prob * 100).toFixed(4)),
        label:       r.label,
        isSeizure:   r.pred_label === 1,
      }))
    : [];

  // ==============================
  // RENDER
  // ==============================
  return (
    <div className="p-6 bg-gray-950 text-white min-h-screen">

      {/* Alarm Banner */}
      {alarmActive && (
        <div className="fixed top-0 left-0 right-0 z-50 bg-red-600 text-white text-center py-4 text-xl font-bold animate-pulse shadow-2xl">
          🚨 SEIZURE ALERT — HIGH PROBABILITY DETECTED — SEEK MEDICAL ATTENTION 🚨
        </div>
      )}

      <div className={`flex items-center gap-4 mb-8 ${alarmActive ? "mt-14" : ""}`}>
        <h1 className="text-3xl font-bold">⚡ EEG Seizure Detection & Alerts</h1>
        <span className={`px-3 py-1 rounded-full text-sm font-semibold ${alertCfg.bg} ${alertCfg.text} border ${alertCfg.border}`}>
          {alertCfg.label}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT — main panels */}
        <div className="lg:col-span-2 space-y-6">

          {/* Device Setup */}
          <Card title="📡 Device Setup & Connection">
            <div className="flex items-center gap-3 mb-4 flex-wrap">
              <button onClick={connectSerial} disabled={connected}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold">
                {connected ? "✅ Connected" : "🔌 Connect Serial"}
              </button>
              <button onClick={disconnectSerial} disabled={!connected}
                className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed">
                Disconnect
              </button>
              <button onClick={testApi}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-semibold">
                🔧 Test API
              </button>
              <label className="flex items-center gap-2 text-sm">
                <span className="text-gray-400">Sample Rate:</span>
                <select value={fs} onChange={e => setFs(Number(e.target.value))} disabled={recording}
                  className="bg-gray-800 text-white px-3 py-1 rounded border border-gray-700 disabled:opacity-50">
                  <option value={250}>250 Hz (default)</option>
                  <option value={128}>128 Hz</option>
                  <option value={256}>256 Hz</option>
                </select>
              </label>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm bg-gray-800 p-4 rounded-lg">
              <div>
                <span className="text-gray-400">Serial Device:</span>{" "}
                <span className={connected ? "text-green-400 font-semibold" : "text-gray-500"}>
                  {connected ? "✅ Connected" : "⚫ Not connected"}
                </span>
              </div>
              <div>
                <span className="text-gray-400">API Status:</span>{" "}
                <span className={
                  apiStatus === "success"  ? "text-green-400 font-semibold" :
                  apiStatus === "error"    ? "text-red-400 font-semibold"   :
                  apiStatus === "fetching" ? "text-yellow-400 font-semibold" : "text-gray-500"
                }>
                  {apiStatus === "success"  && "✅ Connected"}
                  {apiStatus === "error"    && "❌ Error"}
                  {apiStatus === "fetching" && "⏳ Processing"}
                  {apiStatus === "idle"     && "⚫ Idle"}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Total Samples:</span>{" "}
                <span className={`font-mono font-bold ${sampleCount > 0 ? "text-green-400" : "text-red-400"}`}>
                  {sampleCount.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Predictions:</span>{" "}
                <span className="font-mono font-bold text-blue-400">{processingCount}</span>
              </div>
              {recording && (
                <div className="col-span-2">
                  <span className="text-gray-400">Buffer Status:</span>{" "}
                  <span className="text-cyan-400 font-mono">{samples.length}/{windowSize} samples</span>
                  <div className="w-full bg-gray-700 rounded-full h-2 mt-2 overflow-hidden">
                    <div className="h-full bg-cyan-500 transition-all duration-300"
                      style={{ width: `${Math.min(100, (samples.length / windowSize) * 100)}%` }} />
                  </div>
                </div>
              )}
            </div>

            {apiStatus === "error" && (
              <div className="mt-4 bg-red-900/20 border border-red-500/50 rounded-lg p-4 text-red-300 text-sm">
                <div className="font-semibold mb-2">⚠️ API Connection Error</div>
                <div>Make sure Flask server is running:</div>
                <code className="block bg-black/30 p-2 rounded mt-2">python seizure_app.py</code>
              </div>
            )}
          </Card>

          {/* Recording Controls */}
          <Card title="🎬 Recording Controls">
            <div className="flex items-center gap-4 flex-wrap">
              <button onClick={startRecording} disabled={recording || !connected}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg">
                {recording ? "🔴 Recording..." : "▶️ Start Recording"}
              </button>
              <button onClick={stopRecording} disabled={!recording}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg">
                ⏹️ Stop
              </button>
              <div className="text-2xl font-mono font-bold">
                {Math.floor(elapsed / 60)}:{(elapsed % 60).toString().padStart(2, "0")}
                <span className="text-gray-500 text-lg"> / {Math.floor(RECORDING_DURATION / 60)}:00</span>
              </div>
              {recordingComplete && (
                <div className="ml-auto px-4 py-2 bg-green-600 text-white rounded-lg font-semibold animate-pulse">
                  ✅ Complete!
                </div>
              )}
            </div>
            {!connected && !recording && (
              <div className="mt-4 text-yellow-400 text-sm">⚠️ Please connect your EEG device first</div>
            )}
          </Card>

          {/* Live Prediction */}
          <Card>
            {recordingComplete && finalResult && (
              <div className="mb-6 bg-gradient-to-r from-green-900/30 to-blue-900/30 border-2 border-green-500/50 rounded-xl p-6 animate-pulse">
                <div className="text-center">
                  <div className="text-green-400 text-2xl font-bold mb-2">🎉 Recording Complete!</div>
                  <div className="text-gray-300 text-sm mb-4">2-minute session finished</div>
                  <div className="grid grid-cols-2 gap-4 text-left bg-gray-800/50 rounded-lg p-4">
                    <div>
                      <div className="text-gray-400 text-xs">Final Decision</div>
                      <div className={`text-lg font-bold ${finalResult.dominantLabel === "SEIZURE DETECTED" ? "text-red-400" : "text-green-400"}`}>
                        {finalResult.dominantLabel}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Alert Level</div>
                      <div className={`text-lg font-bold ${ALERT_CONFIG[finalResult.finalAlert]?.text}`}>
                        {finalResult.finalAlert}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Avg Seizure Prob</div>
                      <div className="text-white text-lg font-bold">{(finalResult.avgSeizureProb * 100).toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Peak Seizure Prob</div>
                      <div className="text-red-400 text-lg font-bold">{(finalResult.maxSeizureProb * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div className="text-center py-8">
              {label ? (
                <div>
                  <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Current Detection</div>
                  <div className={`text-6xl font-bold mb-6 ${label === "Seizure" ? "text-red-400 animate-pulse" : "text-green-400"}`}>
                    {label === "Seizure" ? "⚡ SEIZURE" : "✅ NO SEIZURE"}
                  </div>

                  <div className={`inline-block px-6 py-2 rounded-full text-lg font-bold mb-8 border-2 ${alertCfg.bg} ${alertCfg.text} ${alertCfg.border}`}>
                    {alertCfg.label} — {alertCfg.desc}
                  </div>

                  <div className="mt-4 mb-6">
                    <div className="text-sm text-gray-400 mb-3 uppercase tracking-wide">Seizure Probability</div>
                    <div className="flex items-center justify-center gap-6">
                      <div className={`text-5xl font-bold ${parseFloat(probPct) >= 75 ? "text-red-400" : parseFloat(probPct) >= 40 ? "text-yellow-400" : "text-green-400"}`}>
                        {probPct}%
                      </div>
                      <div className="w-80 bg-gray-800 rounded-full h-8 overflow-hidden border-2 border-gray-700">
                        <div
                          className={`h-full transition-all duration-1000 ${
                            parseFloat(probPct) >= 75 ? "bg-gradient-to-r from-red-700 to-red-500" :
                            parseFloat(probPct) >= 40 ? "bg-gradient-to-r from-yellow-600 to-yellow-400" :
                            "bg-gradient-to-r from-green-600 to-green-400"
                          }`}
                          style={{ width: `${probPct}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-center gap-6 text-xs text-gray-400 mt-2">
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500 inline-block" /> Low (&lt;40%)</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-yellow-500 inline-block" /> Medium (40–74%)</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500 inline-block" /> High (≥75%)</span>
                  </div>
                </div>
              ) : (
                <div className="text-gray-500 py-16">
                  <div className="text-8xl mb-6">⚡</div>
                  <div className="text-2xl font-semibold mb-2">
                    {recording
                      ? (samples.length < windowSize
                          ? `Collecting data... ${samples.length}/${windowSize}`
                          : "Analysing EEG signals...")
                      : "Ready to start"}
                  </div>
                  {recording && samples.length < windowSize && (
                    <div className="text-base mt-4 text-gray-400">
                      Need {windowSize - samples.length} more samples for first prediction
                    </div>
                  )}
                  {!recording && (
                    <div className="text-base mt-4 text-gray-400">
                      Connect your device and click "Start Recording"
                    </div>
                  )}
                </div>
              )}
            </div>
          </Card>

          {/* Band Powers Chart */}
          <Card title="📊 EEG Band Powers (Focus Feature Standard)">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bandChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" tickFormatter={v => v.toExponential(1)} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: "8px", color: "#fff" }}
                  formatter={v => [v.toExponential(4), "Power"]}
                />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {bandChartData.map((entry, i) => <Cell key={`cell-${i}`} fill={entry.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="grid grid-cols-3 gap-3 mt-6 text-xs">
              {BANDS.map(({ key, label: lbl, color }) => (
                <div key={key} className="bg-gray-800 p-3 rounded-lg text-center border border-gray-700">
                  <div className="text-gray-400 mb-1">{lbl.split(" ")[0]}</div>
                  <div className="font-bold text-base" style={{ color }}>{(bandPowers[key] || 0).toExponential(4)}</div>
                  <div className="text-gray-500 text-xs mt-1">{lbl.split(" ")[1]}</div>
                </div>
              ))}
            </div>
          </Card>

          {/* Live Waveform */}
          <Card title="〰️ Live EEG Signal (Last 2 seconds)">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={waveData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="t" stroke="#9CA3AF" label={{ value: "Time (s)", position: "insideBottom", offset: -5 }} />
                <YAxis stroke="#9CA3AF" label={{ value: "Amplitude", angle: -90, position: "insideLeft" }} />
                <Tooltip contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: "8px", color: "#fff" }} />
                <Line type="monotone" dataKey="v"
                  stroke={alertLevel === "HIGH" ? "#EF4444" : alertLevel === "MEDIUM" ? "#F59E0B" : "#10B981"}
                  dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          {/* ============================================================
              NEW: CSV Upload & Analysis
              ============================================================ */}
          <Card title="📂 CSV File Analysis">
            <div className="space-y-5">

              {/* Upload row */}
              <div className="flex flex-wrap items-end gap-4">
                <div className="flex-1 min-w-[220px]">
                  <label className="block text-gray-400 text-xs mb-1 uppercase tracking-wide">Select CSV File</label>
                  <input
                    ref={csvInputRef}
                    type="file"
                    accept=".csv"
                    onChange={handleCsvFileChange}
                    className="block w-full text-sm text-gray-300
                      file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0
                      file:text-sm file:font-semibold file:bg-blue-600 file:text-white
                      hover:file:bg-blue-700 cursor-pointer"
                  />
                  {csvFile && (
                    <div className="mt-1 text-xs text-green-400">
                      ✅ {csvFile.name} ({(csvFile.size / 1024).toFixed(1)} KB)
                    </div>
                  )}
                </div>

                <div>
                  <label className="block text-gray-400 text-xs mb-1 uppercase tracking-wide">Window Samples</label>
                  <select
                    value={csvWindowSamples}
                    onChange={e => setCsvWindowSamples(Number(e.target.value))}
                    className="bg-gray-800 text-white px-3 py-2 rounded border border-gray-700 text-sm"
                  >
                    <option value={500}>500 (2 s @ 250 Hz)</option>
                    <option value={256}>256 (1 s @ 256 Hz)</option>
                    <option value={512}>512 (2 s @ 256 Hz)</option>
                    <option value={250}>250 (1 s @ 250 Hz)</option>
                  </select>
                </div>

                <button
                  onClick={handleCsvUpload}
                  disabled={!csvFile || csvStatus === "loading"}
                  className="px-5 py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg font-semibold transition disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {csvStatus === "loading" ? "⏳ Analysing..." : "🔬 Analyse CSV"}
                </button>

                {(csvFile || csvResult) && (
                  <button onClick={clearCsvUpload}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition">
                    ✕ Clear
                  </button>
                )}
              </div>

              {/* Expected format hint */}
              <div className="bg-gray-800/60 border border-gray-700 rounded-lg p-3 text-xs text-gray-400">
                <span className="text-gray-300 font-semibold">Expected CSV columns: </span>
                timestamp_ms, eeg_value, alpha_power, beta_power, theta_power, alpha_beta_ratio, signal_variance, attention_index
              </div>

              {/* Error */}
              {csvStatus === "error" && (
                <div className="bg-red-900/20 border border-red-500/40 rounded-lg p-4 text-red-300 text-sm">
                  <div className="font-semibold mb-1">❌ Analysis failed</div>
                  <div>{csvError}</div>
                </div>
              )}

              {/* Results */}
              {csvStatus === "done" && csvResult && (() => {
                const s = csvResult.summary;
                const isSeizure = s.final_decision === "SEIZURE DETECTED";
                return (
                  <div className="space-y-5">

                    {/* Decision Banner */}
                    <div className={`rounded-xl p-5 border-2 text-center ${isSeizure
                      ? "bg-red-900/30 border-red-500/60"
                      : "bg-green-900/30 border-green-500/60"}`}>
                      <div className={`text-3xl font-bold mb-1 ${isSeizure ? "text-red-400" : "text-green-400"}`}>
                        {isSeizure ? "⚡ SEIZURE DETECTED" : "✅ NORMAL"}
                      </div>
                      <div className="text-gray-300 text-sm">{csvResult.csv_info?.filename}</div>
                    </div>

                    {/* Summary stats — matches testing.py console output exactly */}
                    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
                      <div className="text-sm font-bold text-gray-300 mb-3 uppercase tracking-wide border-b border-gray-700 pb-2">
                        📋 Session Summary
                      </div>
                      <div className="grid grid-cols-2 gap-x-8 gap-y-2 text-sm font-mono">
                        <div className="text-gray-400">Total windows analyzed</div>
                        <div className="text-white font-bold">{s.total_windows}</div>

                        <div className="text-gray-400">Seizure windows</div>
                        <div className={`font-bold ${s.seizure_windows > 0 ? "text-red-400" : "text-white"}`}>
                          {s.seizure_windows}
                        </div>

                        <div className="text-gray-400">No Seizure windows</div>
                        <div className="text-green-400 font-bold">{s.normal_windows}</div>

                        <div className="text-gray-400">Seizure ratio</div>
                        <div className={`font-bold ${s.seizure_ratio >= 0.30 ? "text-red-400" : "text-white"}`}>
                          {(s.seizure_ratio * 100).toFixed(2)}%
                        </div>

                        <div className="text-gray-400">Avg seizure probability</div>
                        <div className="text-white font-bold">{(s.avg_seizure_prob * 100).toFixed(4)}%</div>

                        <div className="text-gray-400">Decision rule</div>
                        <div className="text-gray-500 text-xs">Seizure ratio ≥ 30%</div>
                      </div>
                    </div>

                    {/* Per-window seizure probability chart */}
                    <div>
                      <div className="text-sm font-bold text-gray-300 mb-3 uppercase tracking-wide">
                        📈 Seizure Probability — Per Window
                      </div>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={csvChartData}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                          <XAxis dataKey="window" stroke="#9CA3AF" label={{ value: "Window #", position: "insideBottom", offset: -5 }} />
                          <YAxis stroke="#9CA3AF" domain={[0, 100]} tickFormatter={v => v + "%"} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#1F2937", border: "1px solid #374151", borderRadius: "8px", color: "#fff" }}
                            formatter={(v, name) => [v.toFixed(4) + "%", "Seizure Prob"]}
                          />
                          {/* Threshold reference line at 30% */}
                          <Line type="monotone" dataKey="seizureProb"
                            stroke="#EF4444" dot={{ fill: "#EF4444", r: 2 }} strokeWidth={2} isAnimationActive={false} />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Per-window table (scrollable) */}
                    <div>
                      <div className="text-sm font-bold text-gray-300 mb-2 uppercase tracking-wide">
                        🗂 Per-Window Predictions
                      </div>
                      <div className="overflow-auto max-h-64 rounded-lg border border-gray-700">
                        <table className="w-full text-xs font-mono">
                          <thead className="sticky top-0 bg-gray-800">
                            <tr className="text-gray-400">
                              <th className="px-3 py-2 text-left">#</th>
                              <th className="px-3 py-2 text-left">Timestamp (ms)</th>
                              <th className="px-3 py-2 text-left">Prediction</th>
                              <th className="px-3 py-2 text-left">Seizure Prob</th>
                            </tr>
                          </thead>
                          <tbody>
                            {csvResult.window_results.map((row, i) => (
                              <tr key={i}
                                className={`border-t border-gray-700/50 ${row.pred_label === 1 ? "bg-red-900/20" : "hover:bg-gray-800/50"}`}>
                                <td className="px-3 py-1 text-gray-400">{i + 1}</td>
                                <td className="px-3 py-1 text-gray-300">{row.timestamp_ms}</td>
                                <td className={`px-3 py-1 font-bold ${row.pred_label === 1 ? "text-red-400" : "text-green-400"}`}>
                                  {row.label}
                                </td>
                                <td className="px-3 py-1 text-gray-300">{(row.seizure_prob * 100).toFixed(4)}%</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Download CSV PDF */}
                    <button
                      onClick={() => downloadCsvPDF(csvResult, csvFile?.name || "eeg")}
                      className="w-full px-5 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition flex items-center justify-center gap-2"
                    >
                      <span className="text-xl">📄</span>
                      Download PDF Report
                    </button>
                  </div>
                );
              })()}
            </div>
          </Card>
          {/* ============================================================
              END: CSV Upload & Analysis
              ============================================================ */}

        </div>

        {/* RIGHT — sidebar */}
        <div className="space-y-6">

          {/* PDF Report Card */}
          <Card title="📊 Session Report">
            <div className="bg-gradient-to-br from-red-900/20 to-blue-900/20 rounded-lg p-6 border-2 border-red-500/30 space-y-4">
              <div className="text-center mb-4">
                <div className="text-red-400 text-xl font-bold mb-2">⚡ Seizure Monitor</div>
                <div className="text-gray-400 text-sm mb-4">Download full 2-minute live session analysis</div>
                <button onClick={downloadPDF} disabled={!finalResult}
                  className={`mt-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl flex items-center gap-2 mx-auto ${!finalResult ? "opacity-50 cursor-not-allowed" : ""}`}>
                  <span className="text-xl">📄</span>
                  <span>Download PDF Report</span>
                </button>
                {!finalResult && (
                  <div className="text-yellow-400 text-sm mt-2">PDF available after recording completes</div>
                )}
              </div>
            </div>
          </Card>

          {/* Session History */}
          <Card title="📜 Session History">
            {history.length > 0 ? (
              <div className="space-y-3 max-h-[900px] overflow-y-auto pr-2">
                {history.map((h, i) => (
                  <div key={i} className="bg-gray-800 p-4 rounded-lg border-l-4 hover:bg-gray-750 transition"
                    style={{ borderColor: h.dominantLabel === "SEIZURE DETECTED" ? "#EF4444" : "#10B981" }}>
                    <div className={`font-bold text-lg mb-2 ${h.dominantLabel === "SEIZURE DETECTED" ? "text-red-400" : "text-green-400"}`}>
                      {h.dominantLabel}
                    </div>
                    <div className="text-sm text-gray-300 mb-1">
                      Alert: <span className={`font-semibold ${ALERT_CONFIG[h.finalAlert]?.text}`}>{h.finalAlert}</span>
                    </div>
                    <div className="text-sm text-gray-300 mb-1">
                      Avg prob: <span className="text-white font-semibold">{(h.avgSeizureProb * 100).toFixed(1)}%</span>
                    </div>
                    <div className="text-xs text-gray-400 mb-1">
                      Seizure windows: {h.seizureWindows}/{h.totalWindows} ({(h.seizureRatio * 100).toFixed(0)}%)
                    </div>
                    <div className="text-xs text-gray-500">{h.timestamp}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-center py-12">
                <div className="text-4xl mb-3">📝</div>
                <div>No sessions yet</div>
                <div className="text-sm mt-2">Start recording to see results</div>
              </div>
            )}
          </Card>

          {/* Quick Guide */}
          <Card title="ℹ️ Quick Guide">
            <div className="text-sm space-y-3 text-gray-300">
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">1. Start Backend</div>
                <code className="text-xs text-green-400">python seizure_app.py</code>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">2. Live Mode</div>
                <div>Connect serial device → Start Recording</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">3. CSV Mode</div>
                <div>Upload a focus2.csv file → Analyse CSV → Download PDF</div>
              </div>
              <div className="bg-red-900/30 border border-red-500/40 p-3 rounded">
                <div className="font-semibold text-red-400 mb-1">⚠️ Alert Thresholds</div>
                <div className="text-xs space-y-1">
                  <div>🟢 LOW    — &lt;40% seizure probability</div>
                  <div>🟡 MEDIUM — 40–74% seizure probability</div>
                  <div>🔴 HIGH   — ≥75% seizure probability</div>
                </div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">Decision Rule</div>
                <div className="text-xs text-gray-400">Seizure ratio ≥ 30% of windows → SEIZURE DETECTED</div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

// ==============================
// Reusable Card Component
// ==============================
function Card({ title, children, className = "" }) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-xl p-6 shadow-xl ${className}`}>
      {title && <h2 className="text-xl font-bold text-white mb-4 border-b border-gray-800 pb-3">{title}</h2>}
      <div>{children}</div>
    </div>
  );
}