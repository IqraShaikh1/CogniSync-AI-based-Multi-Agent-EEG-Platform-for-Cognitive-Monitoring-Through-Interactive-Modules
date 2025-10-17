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

// Label mapping - converts numeric labels to readable names
const LABEL_NAMES = {
  0: "Baseline",
  1: "Focus1", 
  2: "Focus2",
  3: "Distracted"
};

const SAMPLE_RATE_DEFAULT = 256;
const WINDOW_SEC = 4;
const WINDOW_SIZE_DEFAULT = SAMPLE_RATE_DEFAULT * WINDOW_SEC;

const BANDS = [
  { key: "delta", label: "Delta (0.5-4 Hz)", lo: 0.5, hi: 4, color: "#8B5CF6" },
  { key: "theta", label: "Theta (4-8 Hz)", lo: 4, hi: 8, color: "#3B82F6" },
  { key: "alpha", label: "Alpha (8-12 Hz)", lo: 8, hi: 12, color: "#10B981" },
  { key: "beta", label: "Beta (12-30 Hz)", lo: 12, hi: 30, color: "#F59E0B" },
];

// Simple FFT-based band power computation
function computeBandPowers(samples, fs) {
  const n = samples.length;
  if (n < 64) return { delta: 0, theta: 0, alpha: 0, beta: 0 };
  
  // Apply Hanning window
  const windowed = samples.map((val, i) => 
    val * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)))
  );
  
  // Simple DFT for band powers
  const bandPowers = { delta: 0, theta: 0, alpha: 0, beta: 0 };
  const df = fs / n;
  
  for (let k = 1; k < n / 2; k++) {
    const freq = k * df;
    let re = 0, im = 0;
    
    for (let i = 0; i < n; i++) {
      const angle = -2 * Math.PI * k * i / n;
      re += windowed[i] * Math.cos(angle);
      im += windowed[i] * Math.sin(angle);
    }
    
    const power = (re * re + im * im) / (n * n);
    
    // Assign to bands
    if (freq >= 0.5 && freq < 4) bandPowers.delta += power;
    else if (freq >= 4 && freq < 8) bandPowers.theta += power;
    else if (freq >= 8 && freq < 12) bandPowers.alpha += power;
    else if (freq >= 12 && freq < 30) bandPowers.beta += power;
  }
  
  return bandPowers;
}

async function sendToApi(data) {
  try {
    const response = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: { 
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`API Error ${response.status}:`, errorText);
      return { error: `HTTP ${response.status}` };
    }
    
    return await response.json();
  } catch (err) {
    console.error("API Error:", err);
    return { error: err.message };
  }
}

export default function FocusAttentionPage() {
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Not connected");
  const [fs, setFs] = useState(SAMPLE_RATE_DEFAULT);
  const [windowSize, setWindowSize] = useState(WINDOW_SIZE_DEFAULT);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [samples, setSamples] = useState([]);
  const [sampleCount, setSampleCount] = useState(0);
  const [waveData, setWaveData] = useState([]);
  const [bandPowers, setBandPowers] = useState({ delta: 0, theta: 0, alpha: 0, beta: 0 });
  const [focusLevel, setFocusLevel] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState("idle");
  const [processingCount, setProcessingCount] = useState(0);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const sessionTrackingRef = useRef([]);
  
  const timerRef = useRef(null);
  const portRef = useRef(null);
  const readerRef = useRef(null);
  const recordingRef = useRef(false);
  const samplesBufferRef = useRef([]);
  const lastApiCallRef = useRef(0);

  const RECORDING_DURATION = 120;
  const API_CALL_INTERVAL = 3000;

  useEffect(() => {
    setWindowSize(fs * WINDOW_SEC);
  }, [fs]);

  useEffect(() => {
    recordingRef.current = recording;
  }, [recording]);

  useEffect(() => {
    if (!recording || samples.length < windowSize) return;
    
    const now = Date.now();
    if (now - lastApiCallRef.current < API_CALL_INTERVAL) return;
    
    const processData = async () => {
      try {
        console.log(`🔬 Processing ${samples.length} samples...`);
        
        const bp = computeBandPowers(samples.slice(-windowSize), fs);
        console.log("📊 Band Powers:", bp);
        setBandPowers(bp);
        
        const waveWindow = Math.min(samples.length, fs * 2);
        setWaveData(
          samples
            .slice(-waveWindow)
            .map((v, i) => ({ t: (i / fs).toFixed(2), v: v.toFixed(2) }))
        );
        
        setApiStatus("fetching");
        lastApiCallRef.current = now;
        
        const apiRes = await sendToApi({
          signal: samples.slice(-windowSize),
          fs: fs,
          delta: bp.delta,
          theta: bp.theta,
          alpha: bp.alpha,
          beta: bp.beta
        });
        
        if (apiRes && !apiRes.error) {
          const predictionNum = apiRes.prediction;
          const predictionName = LABEL_NAMES[predictionNum] || `State ${predictionNum}`;
          
          setPrediction(predictionName);
          setProbabilities(apiRes.probabilities || null);
          setFocusLevel(apiRes.focus_level || 0);
          setApiStatus("success");
          setProcessingCount(c => c + 1);
          
          const trackingEntry = {
            timestamp: Date.now(),
            elapsed: elapsed,
            state: predictionName,
            stateNum: predictionNum,
            focusLevel: apiRes.focus_level || 0,
            probabilities: apiRes.probabilities,
            bandPowers: { ...bp }
          };
          sessionTrackingRef.current.push(trackingEntry);
          
          console.log("✅ Prediction:", predictionName, `(${predictionNum})`);
          console.log("📊 Model Focus Level:", apiRes.focus_level?.toFixed(1) + "%");
        } else {
          setApiStatus("error");
          console.error("⚠️ API error:", apiRes?.error);
        }
        
      } catch (error) {
        console.error("❌ Processing error:", error);
        setApiStatus("error");
      }
    };
    
    processData();
  }, [samples.length, recording, windowSize, fs, elapsed]);

  const connectSerial = async () => {
    try {
      if (!("serial" in navigator)) {
        throw new Error("Web Serial API not supported in this browser");
      }
      
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      portRef.current = port;
      
      setConnected(true);
      setStatusMsg("Serial connected");
      
      const decoder = new TextDecoderStream();
      port.readable.pipeTo(decoder.writable);
      const reader = decoder.readable.getReader();
      readerRef.current = reader;
      
      let lineBuffer = "";
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
              let val = NaN;
              
              if (parts.length > 0) {
                val = parseFloat(parts[parts.length - 1]);
              }
              
              if (isNaN(val)) {
                const match = line.match(/[-+]?\d*\.?\d+/);
                if (match) val = parseFloat(match[0]);
              }
              
              if (!isNaN(val) && isFinite(val)) {
                validSampleCount++;
                setSampleCount(validSampleCount);
                
                if (recordingRef.current) {
                  samplesBufferRef.current.push(val);
                  
                  if (samplesBufferRef.current.length > windowSize * 2) {
                    samplesBufferRef.current = samplesBufferRef.current.slice(-windowSize);
                  }
                  
                  if (samplesBufferRef.current.length % 50 === 0) {
                    setSamples([...samplesBufferRef.current]);
                  }
                }
                
                if (validSampleCount <= 5) {
                  console.log(`✅ Sample ${validSampleCount}:`, val);
                }
              }
            }
          }
        } catch (e) {
          console.error("Read loop error:", e);
        }
      };
      
      readLoop();
      
    } catch (e) {
      console.error("Serial connection error:", e);
      setConnected(false);
      setStatusMsg(e.message);
    }
  };

  const disconnectSerial = async () => {
    try {
      setRecording(false);
      clearInterval(timerRef.current);
      
      if (readerRef.current) {
        try {
          await readerRef.current.cancel().catch(() => {});
        } catch (e) {}
        readerRef.current = null;
      }
      
      if (portRef.current) {
        try {
          const closePromise = portRef.current.close();
          await Promise.race([
            closePromise,
            new Promise(resolve => setTimeout(resolve, 1000))
          ]);
        } catch (e) {}
        portRef.current = null;
      }
      
      setConnected(false);
      setStatusMsg("Disconnected");
      setSamples([]);
      setSampleCount(0);
      setPrediction(null);
      console.log("✅ Disconnected successfully");
    } catch (e) {
      console.log("Disconnect completed with errors (ignored)");
      setConnected(false);
      setStatusMsg("Disconnected");
      portRef.current = null;
      readerRef.current = null;
    }
  };

  const testApi = async () => {
    try {
      const response = await fetch("http://127.0.0.1:8000/test");
      const data = await response.json();
      
      if (data.status === "success") {
        alert("✅ API is working!\n\n" + JSON.stringify(data, null, 2));
        setApiStatus("success");
      } else {
        alert("❌ API Error");
        setApiStatus("error");
      }
    } catch (err) {
      alert("❌ Cannot connect to Flask API!\n\nMake sure the server is running:\npython app.py");
      setApiStatus("error");
    }
  };

  const startRecording = () => {
    console.log("🎬 Recording started");
    samplesBufferRef.current = [];
    setSamples([]);
    setSampleCount(0);
    setElapsed(0);
    setPrediction(null);
    setProbabilities(null);
    setProcessingCount(0);
    setRecording(true);
    setApiStatus("idle");
    setRecordingComplete(false);
    setFinalResult(null);
    sessionTrackingRef.current = [];
    lastApiCallRef.current = 0;

    timerRef.current = setInterval(() => {
      setElapsed((prev) => {
        const next = prev + 1;
        if (next >= RECORDING_DURATION) {
          console.log("⏰ Timer reached 2:00, stopping recording...");
          setTimeout(() => stopRecording(), 100);
          return RECORDING_DURATION;
        }
        return next;
      });
    }, 1000);
  };

  const stopRecording = () => {
    console.log("⏹️ Recording stopped");
    setRecording(false);
    clearInterval(timerRef.current);
    
    const trackingData = sessionTrackingRef.current;
    
    if (trackingData.length > 0) {
      const stateDurations = {};
      const stateOccurrences = {};
      
      trackingData.forEach((entry, index) => {
        const state = entry.state;
        const duration = index < trackingData.length - 1 
          ? (trackingData[index + 1].elapsed - entry.elapsed)
          : (RECORDING_DURATION - entry.elapsed);
        
        stateDurations[state] = (stateDurations[state] || 0) + duration;
        stateOccurrences[state] = (stateOccurrences[state] || 0) + 1;
      });
      
      const avgFocusLevel = trackingData.reduce((sum, entry) => sum + entry.focusLevel, 0) / trackingData.length;
      
      const avgBandPowers = {
        delta: trackingData.reduce((sum, e) => sum + (e.bandPowers.delta || 0), 0) / trackingData.length,
        theta: trackingData.reduce((sum, e) => sum + (e.bandPowers.theta || 0), 0) / trackingData.length,
        alpha: trackingData.reduce((sum, e) => sum + (e.bandPowers.alpha || 0), 0) / trackingData.length,
        beta: trackingData.reduce((sum, e) => sum + (e.bandPowers.beta || 0), 0) / trackingData.length,
      };
      
      const dominantState = Object.entries(stateDurations)
        .sort(([, a], [, b]) => b - a)[0][0];
      
      const avgProbabilities = {};
      Object.keys(LABEL_NAMES).forEach(stateNum => {
        const probs = trackingData
          .map(e => e.probabilities?.[stateNum] || 0)
          .filter(p => p > 0);
        avgProbabilities[stateNum] = probs.length > 0
          ? probs.reduce((sum, p) => sum + p, 0) / probs.length
          : 0;
      });
      
      const report = {
        prediction: dominantState,
        probabilities: avgProbabilities,
        focusLevel: avgFocusLevel,
        bandPowers: avgBandPowers,
        timestamp: new Date().toLocaleTimeString(),
        sampleCount: samplesBufferRef.current.length,
        duration: elapsed,
        stateDurations,
        stateOccurrences,
        dominantState,
        totalPredictions: trackingData.length,
        trackingData: trackingData
      };
      
      setFinalResult(report);
      setHistory((prev) => [report, ...prev].slice(0, 50));
      console.log("📊 Session Report Generated:", report);
    } else if (prediction) {
      const result = {
        prediction,
        probabilities,
        focusLevel,
        bandPowers,
        timestamp: new Date().toLocaleTimeString(),
        sampleCount: samplesBufferRef.current.length,
        duration: elapsed
      };
      setFinalResult(result);
      setHistory((prev) => [result, ...prev].slice(0, 50));
    }
    
    if (elapsed >= RECORDING_DURATION - 5) {
      setRecordingComplete(true);
      console.log("✅ Recording completed successfully!");
    }
  };

  const downloadPDF = () => {
    if (!finalResult) return;
    
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    const margin = 20;
    let y = 20;
    
    doc.setFontSize(20);
    doc.setFont(undefined, 'bold');
    doc.text("EEG Focus & Attention Report", pageWidth / 2, y, { align: 'center' });
    y += 15;
    
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    doc.text(`Session Date: ${new Date().toLocaleDateString()}`, margin, y);
    y += 6;
    doc.text(`Session Time: ${finalResult.timestamp}`, margin, y);
    y += 6;
    doc.text(`Duration: ${Math.floor(finalResult.duration / 60)}:${(finalResult.duration % 60).toString().padStart(2, '0')} minutes`, margin, y);
    y += 12;
    
    doc.setDrawColor(200);
    doc.line(margin, y, pageWidth - margin, y);
    y += 10;
    
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text("Dominant Mental State", margin, y);
    y += 8;
    doc.setFontSize(18);
    doc.setTextColor(34, 197, 94);
    doc.text(finalResult.dominantState || finalResult.prediction, margin, y);
    doc.setTextColor(0);
    y += 8;
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    if (finalResult.stateDurations) {
      const duration = finalResult.stateDurations[finalResult.dominantState || finalResult.prediction] || 0;
      doc.text(`Duration: ${duration.toFixed(0)}s (${((duration / RECORDING_DURATION) * 100).toFixed(1)}%)`, margin, y);
    }
    y += 15;
    
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text("Average Focus Level", margin, y);
    y += 8;
    doc.setFontSize(16);
    doc.text(`${finalResult.focusLevel.toFixed(1)}%`, margin, y);
    y += 15;
    
    if (finalResult.stateDurations) {
      doc.setFontSize(14);
      doc.setFont(undefined, 'bold');
      doc.text("Time Spent in Each State", margin, y);
      y += 8;
      doc.setFontSize(10);
      doc.setFont(undefined, 'normal');
      
      Object.entries(finalResult.stateDurations)
        .sort(([, a], [, b]) => b - a)
        .forEach(([state, duration]) => {
          const percentage = ((duration / RECORDING_DURATION) * 100).toFixed(1);
          const occurrences = finalResult.stateOccurrences[state] || 0;
          doc.text(`${state}: ${duration.toFixed(0)}s (${percentage}%) - ${occurrences} occurrences`, margin + 5, y);
          y += 6;
        });
      y += 10;
    }
    
    if (finalResult.probabilities) {
      doc.setFontSize(14);
      doc.setFont(undefined, 'bold');
      doc.text("Average Classification Confidence", margin, y);
      y += 8;
      doc.setFontSize(10);
      doc.setFont(undefined, 'normal');
      
      Object.entries(finalResult.probabilities)
        .sort(([, a], [, b]) => b - a)
        .forEach(([cls, prob]) => {
          doc.text(`${LABEL_NAMES[cls] || `State ${cls}`}: ${(prob * 100).toFixed(1)}%`, margin + 5, y);
          y += 6;
        });
      y += 10;
    }
    
    if (finalResult.bandPowers) {
      doc.setFontSize(14);
      doc.setFont(undefined, 'bold');
      doc.text("Average EEG Band Powers", margin, y);
      y += 8;
      doc.setFontSize(10);
      doc.setFont(undefined, 'normal');
      
      Object.entries(finalResult.bandPowers).forEach(([band, power]) => {
        doc.text(`${band.charAt(0).toUpperCase() + band.slice(1)}: ${power.toFixed(6)}`, margin + 5, y);
        y += 6;
      });
      y += 10;
    }
    
    doc.setFontSize(14);
    doc.setFont(undefined, 'bold');
    doc.text("Session Statistics", margin, y);
    y += 8;
    doc.setFontSize(10);
    doc.setFont(undefined, 'normal');
    doc.text(`Total Predictions: ${finalResult.totalPredictions || 'N/A'}`, margin + 5, y);
    y += 6;
    doc.text(`Total Samples: ${finalResult.sampleCount?.toLocaleString() || 'N/A'}`, margin + 5, y);
    y += 6;
    doc.text(`Sample Rate: ${fs} Hz`, margin + 5, y);
    
    y = doc.internal.pageSize.getHeight() - 20;
    doc.setFontSize(8);
    doc.setTextColor(128);
    doc.text("Generated by EEG Focus & Attention Tracker", pageWidth / 2, y, { align: 'center' });
    
    const filename = `EEG_Report_${new Date().toISOString().split('T')[0]}_${new Date().toTimeString().split(' ')[0].replace(/:/g, '-')}.pdf`;
    doc.save(filename);
  };

  const bandChartData = BANDS.map(({ key, label, color }) => ({
    name: label.split(" ")[0],
    value: parseFloat((bandPowers[key] || 0).toFixed(6)),
    color,
  }));

  const getStatusColor = (pred) => {
    if (!pred) return "gray";
    const p = pred.toLowerCase();
    if (p.includes("focus") || p.includes("concentrated") || p.includes("attention")) return "green";
    if (p.includes("distract")) return "red";
    if (p.includes("neutral") || p.includes("relax") || p.includes("baseline")) return "blue";
    if (p.includes("drowsy") || p.includes("tired")) return "yellow";
    return "blue";
  };

  const statusColor = getStatusColor(prediction);

  return (
    <div className="p-6 bg-gray-950 text-white min-h-screen">
      <div className="flex items-center gap-4 mb-8">
        <h1 className="text-3xl font-bold">🧠 EEG Focus & Attention Tracker</h1>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card title="📡 Device Setup & Connection">
            <div className="flex items-center gap-3 mb-4 flex-wrap">
              <button
                onClick={connectSerial}
                disabled={connected}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
              >
                {connected ? "✅ Connected" : "🔌 Connect Serial"}
              </button>
              <button
                onClick={disconnectSerial}
                disabled={!connected}
                className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Disconnect
              </button>
              
              <button
                onClick={testApi}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-semibold"
              >
                🔧 Test API
              </button>
              
              <label className="flex items-center gap-2 text-sm">
                <span className="text-gray-400">Sample Rate:</span>
                <select
                  value={fs}
                  onChange={(e) => setFs(Number(e.target.value))}
                  disabled={recording}
                  className="bg-gray-800 text-white px-3 py-1 rounded border border-gray-700 disabled:opacity-50"
                >
                  <option value={128}>128 Hz</option>
                  <option value={256}>256 Hz</option>
                  <option value={512}>512 Hz</option>
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
                  apiStatus === "success" ? "text-green-400 font-semibold" :
                  apiStatus === "error" ? "text-red-400 font-semibold" :
                  apiStatus === "fetching" ? "text-yellow-400 font-semibold" : "text-gray-500"
                }>
                  {apiStatus === "success" && "✅ Connected"}
                  {apiStatus === "error" && "❌ Error"}
                  {apiStatus === "fetching" && "⏳ Processing"}
                  {apiStatus === "idle" && "⚫ Idle"}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Total Samples:</span>{" "}
                <span className={`font-mono font-bold ${sampleCount > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {sampleCount.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Predictions:</span>{" "}
                <span className="font-mono font-bold text-blue-400">
                  {processingCount}
                </span>
              </div>
              {recording && (
                <div className="col-span-2">
                  <span className="text-gray-400">Buffer Status:</span>{" "}
                  <span className="text-cyan-400 font-mono">
                    {samples.length}/{windowSize} samples
                  </span>
                  <div className="w-full bg-gray-700 rounded-full h-2 mt-2 overflow-hidden">
                    <div
                      className="h-full bg-cyan-500 transition-all duration-300"
                      style={{ width: `${Math.min(100, (samples.length / windowSize) * 100)}%` }}
                    />
                  </div>
                </div>
              )}
            </div>
            
            {apiStatus === "error" && (
              <div className="mt-4 bg-red-900/20 border border-red-500/50 rounded-lg p-4 text-red-300 text-sm">
                <div className="font-semibold mb-2">⚠️ API Connection Error</div>
                <div>Make sure Flask server is running:</div>
                <code className="block bg-black/30 p-2 rounded mt-2">python app.py</code>
              </div>
            )}
          </Card>

          <Card title="🎬 Recording Controls">
            <div className="flex items-center gap-4 flex-wrap">
              <button
                onClick={startRecording}
                disabled={recording || !connected}
                className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg"
              >
                {recording ? "🔴 Recording..." : "▶️ Start Recording"}
              </button>
              <button
                onClick={stopRecording}
                disabled={!recording}
                className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg"
              >
                ⏹️ Stop
              </button>
              <div className="text-2xl font-mono font-bold">
                {Math.floor(elapsed / 60)}:{(elapsed % 60).toString().padStart(2, "0")}
                <span className="text-gray-500 text-lg"> / {Math.floor(RECORDING_DURATION / 60)}:{(RECORDING_DURATION % 60).toString().padStart(2, "0")}</span>
              </div>
              
              {recordingComplete && (
                <div className="ml-auto px-4 py-2 bg-green-600 text-white rounded-lg font-semibold animate-pulse">
                  ✅ Complete!
                </div>
              )}
            </div>
            
            {!connected && !recording && (
              <div className="mt-4 text-yellow-400 text-sm">
                ⚠️ Please connect your EEG device first
              </div>
            )}
          </Card>

          <Card>
            {recordingComplete && finalResult && (
              <div className="mb-6 bg-gradient-to-r from-green-900/30 to-blue-900/30 border-2 border-green-500/50 rounded-xl p-6 animate-pulse">
                <div className="text-center">
                  <div className="text-green-400 text-2xl font-bold mb-2">🎉 Recording Complete!</div>
                  <div className="text-gray-300 text-sm mb-4">2-minute session finished</div>
                  <div className="grid grid-cols-2 gap-4 text-left bg-gray-800/50 rounded-lg p-4">
                    <div>
                      <div className="text-gray-400 text-xs">Final State</div>
                      <div className="text-white text-lg font-bold">{finalResult.prediction}</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Focus Level</div>
                      <div className="text-blue-400 text-lg font-bold">{finalResult.focusLevel.toFixed(1)}%</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Duration</div>
                      <div className="text-white text-lg font-bold">
                        {Math.floor(finalResult.duration / 60)}:{(finalResult.duration % 60).toString().padStart(2, '0')}
                      </div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Samples</div>
                      <div className="text-white text-lg font-bold">{finalResult.sampleCount.toLocaleString()}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="text-center py-8">
              {prediction ? (
                <div>
                  <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Current State</div>
                  <div className={`text-6xl font-bold mb-6 ${
                    statusColor === "green" ? "text-green-400" :
                    statusColor === "red" ? "text-red-400" : 
                    statusColor === "yellow" ? "text-yellow-400" : "text-blue-400"
                  }`}>
                    {prediction}
                  </div>
                  
                  <div className="mt-8 mb-6">
                    <div className="text-sm text-gray-400 mb-3 uppercase tracking-wide">Focus Level</div>
                    <div className="flex items-center justify-center gap-6">
                      <div className="text-5xl font-bold text-blue-400">
                        {focusLevel.toFixed(1)}%
                      </div>
                      <div className="w-80 bg-gray-800 rounded-full h-8 overflow-hidden border-2 border-gray-700">
                        <div
                          className={`h-full transition-all duration-1000 ${
                            focusLevel > 60 ? "bg-gradient-to-r from-green-500 to-green-400" :
                            focusLevel > 30 ? "bg-gradient-to-r from-yellow-500 to-yellow-400" : 
                            "bg-gradient-to-r from-red-500 to-red-400"
                          }`}
                          style={{ width: `${focusLevel}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {probabilities && (
                    <div className="mt-8 bg-gray-800 p-6 rounded-lg inline-block border border-gray-700">
                      <div className="text-sm text-gray-400 mb-3 uppercase tracking-wide">Classification Confidence</div>
                      <div className="grid grid-cols-2 gap-4 text-left">
                        {Object.entries(probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([cls, prob]) => (
                            <div key={cls} className="flex items-center justify-between gap-4">
                              <span className="text-gray-300 font-medium">{LABEL_NAMES[cls] || cls}:</span>
                              <span className="text-white font-bold">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-gray-500 py-16">
                  <div className="text-8xl mb-6">🧠</div>
                  <div className="text-2xl font-semibold mb-2">
                    {recording 
                      ? (samples.length < windowSize 
                          ? `Collecting data... ${samples.length}/${windowSize}` 
                          : "Processing EEG signals...")
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

          <Card title="📊 EEG Band Powers">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={bandChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: "#1F2937", 
                    border: "1px solid #374151", 
                    borderRadius: "8px",
                    color: "#fff"
                  }}
                />
                <Bar dataKey="value" radius={[8, 8, 0, 0]}>
                  {bandChartData.map((entry, i) => (
                    <Cell key={`cell-${i}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            
            <div className="grid grid-cols-4 gap-3 mt-6 text-xs">
              {BANDS.map(({ key, label, color }) => (
                <div key={key} className="bg-gray-800 p-3 rounded-lg text-center border border-gray-700">
                  <div className="text-gray-400 mb-1">{label.split(" ")[0]}</div>
                  <div className="font-bold text-white text-base" style={{ color }}>
                    {(bandPowers[key] || 0).toFixed(6)}
                  </div>
                  <div className="text-gray-500 text-xs mt-1">{label.split(" ")[1]}</div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="〰️ Live EEG Signal (Last 2 seconds)">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={waveData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="t" 
                  stroke="#9CA3AF"
                  label={{ value: 'Time (s)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  label={{ value: 'Amplitude', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: "#1F2937", 
                    border: "1px solid #374151", 
                    borderRadius: "8px",
                    color: "#fff"
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="v"
                  stroke="#10B981"
                  dot={false}
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </div>

        <div className="space-y-6">
           (
            <Card title="📊 Session Summary Report">
  <div className="bg-gradient-to-br from-green-900/20 to-blue-900/20 rounded-lg p-6 border-2 border-green-500/30 space-y-4">
    <div className="text-center mb-4">
      <div className="text-green-400 text-xl font-bold mb-2">✅ 2-Minute Session Complete!</div>
      <div className="text-gray-400 text-sm mb-4">Comprehensive Analysis Report</div>

      {/* Always show the button */}
      <button
        onClick={downloadPDF}
        disabled={!finalResult} // disable until finalResult exists
        className={`mt-2 px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-semibold transition-all shadow-lg hover:shadow-xl flex items-center gap-2 mx-auto
          ${!finalResult ? "opacity-50 cursor-not-allowed" : ""}`}
      >
        <span className="text-xl">📄</span>
        <span>Download PDF Report</span>
      </button>
    </div>

    {/* Optional: Show a notice if finalResult not ready */}
    {!finalResult && (
      <div className="text-yellow-400 text-sm mt-2">
        PDF will be available after recording completes
      </div>
    )}
  </div>
</Card>

            
          )
          
          <Card title="📜 Session History">
            {history.length > 0 ? (
              <div className="space-y-3 max-h-[900px] overflow-y-auto pr-2">
                {history.map((h, i) => (
                  <div 
                    key={i} 
                    className="bg-gray-800 p-4 rounded-lg border-l-4 hover:bg-gray-750 transition" 
                    style={{
                      borderColor: getStatusColor(h.prediction) === "green" ? "#10B981" :
                                  getStatusColor(h.prediction) === "red" ? "#EF4444" : 
                                  getStatusColor(h.prediction) === "yellow" ? "#F59E0B" : "#3B82F6"
                    }}
                  >
                    <div className="font-bold text-lg mb-2">{h.prediction}</div>
                    <div className="text-sm text-gray-300 mb-1">
                      Focus: <span className="text-white font-semibold">{h.focusLevel.toFixed(1)}%</span>
                    </div>
                    <div className="text-xs text-gray-400 mb-2">
                      Samples: {h.sampleCount?.toLocaleString() || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">{h.timestamp}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500 text-center py-12">
                <div className="text-4xl mb-3">📝</div>
                <div>No predictions yet</div>
                <div className="text-sm mt-2">Start recording to see results</div>
              </div>
            )}
          </Card>
          
          <Card title="ℹ️ Quick Guide">
            <div className="text-sm space-y-3 text-gray-300">
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">1. Connect Device</div>
                <div>Click "Connect Serial" and select your EEG headset port</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">2. Test API</div>
                <div>Verify the Flask server is running and connected</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">3. Start Recording</div>
                <div>Click "Start Recording" to begin collecting EEG data</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">4. View Results</div>
                <div>Real-time predictions appear every 3 seconds</div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function Card({ title, children, className = "" }) {
  return (
    <div className={`bg-gray-900 border border-gray-800 rounded-xl p-6 shadow-xl ${className}`}>
      {title && <h2 className="text-xl font-bold text-white mb-4 border-b border-gray-800 pb-3">{title}</h2>}
      <div>{children}</div>
    </div>
  );
}