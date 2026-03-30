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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";

// Label mapping - converts numeric labels to readable names
const LABEL_NAMES = {
  1: "Happiness",
  2: "Anger",
  3: "Sadness",
  4: "Fear",
  5: "Neutral"
};

// Emotion colors for visualization
const EMOTION_COLORS = {
  1: "#FBBF24", // Happiness - Yellow
  2: "#EF4444", // Anger - Red
  3: "#3B82F6", // Sadness - Blue
  4: "#A855F7", // Fear - Purple
  5: "#6B7280"  // Neutral - Gray
};

const SAMPLE_RATE_DEFAULT = 256;
const WINDOW_SEC = 4;
const WINDOW_SIZE_DEFAULT = SAMPLE_RATE_DEFAULT * WINDOW_SEC;

const BANDS = [
  { key: "delta", label: "Delta (0.5-4 Hz)", lo: 0.5, hi: 4, color: "#8B5CF6" },
  { key: "theta", label: "Theta (4-8 Hz)", lo: 4, hi: 8, color: "#3B82F6" },
  { key: "alpha", label: "Alpha (8-12 Hz)", lo: 8, hi: 12, color: "#10B981" },
  { key: "beta", label: "Beta (12-30 Hz)", lo: 12, hi: 30, color: "#F59E0B" },
  { key: "gamma", label: "Gamma (30-50 Hz)", lo: 30, hi: 50, color: "#EC4899" },
];

// Compute band powers using FFT
function computeBandPowers(samples, fs) {
  const n = samples.length;
  if (n < 64) return { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };
  
  const windowed = samples.map((val, i) => 
    val * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)))
  );
  
  const bandPowers = { delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 };
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
    
    if (freq >= 0.5 && freq < 4) bandPowers.delta += power;
    else if (freq >= 4 && freq < 8) bandPowers.theta += power;
    else if (freq >= 8 && freq < 12) bandPowers.alpha += power;
    else if (freq >= 12 && freq < 30) bandPowers.beta += power;
    else if (freq >= 30 && freq < 50) bandPowers.gamma += power;
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

export default function MoodEmotionPage() {
  const [connected, setConnected] = useState(false);
  const [statusMsg, setStatusMsg] = useState("Not connected");
  const [fs, setFs] = useState(SAMPLE_RATE_DEFAULT);
  const [windowSize, setWindowSize] = useState(WINDOW_SIZE_DEFAULT);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [samples, setSamples] = useState([]);
  const [sampleCount, setSampleCount] = useState(0);
  const [waveData, setWaveData] = useState([]);
  const [bandPowers, setBandPowers] = useState({ delta: 0, theta: 0, alpha: 0, beta: 0, gamma: 0 });
  const [arousal, setArousal] = useState(0);
  const [valence, setValence] = useState(0);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState(null);
  const [history, setHistory] = useState([]);
  const [apiStatus, setApiStatus] = useState("idle");
  const [processingCount, setProcessingCount] = useState(0);
  const [recordingComplete, setRecordingComplete] = useState(false);
  const [finalResult, setFinalResult] = useState(null);
  const sessionTrackingRef = useRef([]);
  const [useCSV, setUseCSV] = useState(false);
  const [csvFile, setCsvFile] = useState(null);
  const [csvProcessing, setCsvProcessing] = useState(false);
  
  const timerRef = useRef(null);
  const portRef = useRef(null);
  const readerRef = useRef(null);
  const recordingRef = useRef(false);
  const samplesBufferRef = useRef([]);
  const lastApiCallRef = useRef(0);

  const RECORDING_DURATION = 60;
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
          beta: bp.beta,
          gamma: bp.gamma
        });
        
        if (apiRes && !apiRes.error) {
          const predictionNum = apiRes.prediction;
          const predictionName = LABEL_NAMES[predictionNum] || `Emotion ${predictionNum}`;
          
          setPrediction(predictionName);
          setProbabilities(apiRes.probabilities || null);
          setArousal(apiRes.arousal_index || 0);
          setValence(apiRes.valence_proxy || 0);
          setApiStatus("success");
          setProcessingCount(c => c + 1);
          
          const trackingEntry = {
            timestamp: Date.now(),
            elapsed: elapsed,
            emotion: predictionName,
            emotionNum: predictionNum,
            arousal: apiRes.arousal_index || 0,
            valence: apiRes.valence_proxy || 0,
            probabilities: apiRes.probabilities,
            bandPowers: { ...bp }
          };
          sessionTrackingRef.current.push(trackingEntry);
          
          console.log("✅ Emotion:", predictionName, `(${predictionNum})`);
          console.log("📊 Arousal:", apiRes.arousal_index?.toFixed(1), "Valence:", apiRes.valence_proxy?.toFixed(1));
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
      alert("❌ Cannot connect to Flask API!\n\nMake sure the server is running:\npython app_emotion.py");
      setApiStatus("error");
    }
  };

  const handleCSVUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setCsvFile(file);
      console.log("📁 CSV file selected:", file.name);
    }
  };

  const processCSVData = async () => {
    if (!csvFile) {
      alert("⚠️ Please select a CSV file first");
      return;
    }

    setCsvProcessing(true);
    setApiStatus("fetching");

    try {
      const text = await csvFile.text();
      const lines = text.split('\n').filter(line => line.trim());
      
      // Skip header row
      const dataLines = lines.slice(1);
      
      // Extract EEG values (assuming column index 1 is eeg_raw_value)
      const eegValues = [];
      for (const line of dataLines) {
        const parts = line.split(',');
        if (parts.length > 1) {
          const value = parseFloat(parts[1]);
          if (!isNaN(value) && isFinite(value)) {
            eegValues.push(value);
          }
        }
      }

      if (eegValues.length < windowSize) {
        alert(`⚠️ CSV has insufficient data. Need at least ${windowSize} samples, got ${eegValues.length}`);
        setCsvProcessing(false);
        setApiStatus("idle");
        return;
      }

      console.log(`📊 Processing ${eegValues.length} samples from CSV...`);
      setSampleCount(eegValues.length);
      
      // Process in windows
      const predictions = [];
      const step = Math.floor(windowSize / 2); // 50% overlap
      
      for (let i = 0; i <= eegValues.length - windowSize; i += step) {
        const window = eegValues.slice(i, i + windowSize);
        
        const bp = computeBandPowers(window, fs);
        
        const apiRes = await sendToApi({
          signal: window,
          fs: fs,
          delta: bp.delta,
          theta: bp.theta,
          alpha: bp.alpha,
          beta: bp.beta,
          gamma: bp.gamma
        });

        if (apiRes && !apiRes.error) {
          const predictionNum = apiRes.prediction;
          const predictionName = LABEL_NAMES[predictionNum] || `Emotion ${predictionNum}`;
          
          predictions.push({
            emotion: predictionName,
            emotionNum: predictionNum,
            arousal: apiRes.arousal_index || 0,
            valence: apiRes.valence_proxy || 0,
            probabilities: apiRes.probabilities,
            bandPowers: { ...bp }
          });

          // Log first few predictions for debugging
          if (predictions.length <= 3) {
            console.log(`✅ Window ${predictions.length}:`, predictionName, `(${predictionNum})`);
            console.log(`   Probabilities:`, apiRes.probabilities);
          } else if (predictions.length % 10 === 0) {
            console.log(`✅ Processed ${predictions.length} windows...`);
          }
        }

        // Update progress
        const progress = Math.floor((i / (eegValues.length - windowSize)) * 100);
        setProcessingCount(predictions.length);
      }

      if (predictions.length > 0) {
        // Calculate aggregate results - count occurrences by emotion NUMBER
        const emotionCounts = {};
        predictions.forEach(p => {
          const num = p.emotionNum;
          emotionCounts[num] = (emotionCounts[num] || 0) + 1;
        });

        // Find dominant emotion by highest count
        const dominantEmotionNum = Object.entries(emotionCounts)
          .sort(([, a], [, b]) => b - a)[0][0];
        const dominantEmotion = LABEL_NAMES[dominantEmotionNum];

        console.log("📊 Emotion counts by number:", emotionCounts);
        console.log("🎯 Dominant emotion:", dominantEmotion, `(${dominantEmotionNum})`);

        const avgArousal = predictions.reduce((sum, p) => sum + p.arousal, 0) / predictions.length;
        const avgValence = predictions.reduce((sum, p) => sum + p.valence, 0) / predictions.length;

        const avgBandPowers = {
          delta: predictions.reduce((sum, p) => sum + p.bandPowers.delta, 0) / predictions.length,
          theta: predictions.reduce((sum, p) => sum + p.bandPowers.theta, 0) / predictions.length,
          alpha: predictions.reduce((sum, p) => sum + p.bandPowers.alpha, 0) / predictions.length,
          beta: predictions.reduce((sum, p) => sum + p.bandPowers.beta, 0) / predictions.length,
          gamma: predictions.reduce((sum, p) => sum + p.bandPowers.gamma, 0) / predictions.length,
        };

        // Calculate average probabilities properly
        const avgProbabilities = {};
        Object.keys(LABEL_NAMES).forEach(emotionNum => {
          const allProbs = predictions.map(p => p.probabilities?.[emotionNum] || 0);
          avgProbabilities[emotionNum] = allProbs.reduce((sum, p) => sum + p, 0) / allProbs.length;
        });

        console.log("📊 Average probabilities:", avgProbabilities);

        const result = {
          prediction: dominantEmotion,
          probabilities: avgProbabilities,
          arousal: avgArousal,
          valence: avgValence,
          bandPowers: avgBandPowers,
          timestamp: new Date().toLocaleTimeString(),
          sampleCount: eegValues.length,
          duration: Math.floor(eegValues.length / fs),
          totalPredictions: predictions.length,
          emotionCounts: Object.fromEntries(
            Object.entries(emotionCounts).map(([num, count]) => [LABEL_NAMES[num], count])
          ),
          source: 'CSV'
        };

        setPrediction(dominantEmotion);
        setProbabilities(avgProbabilities);
        setArousal(avgArousal);
        setValence(avgValence);
        setBandPowers(avgBandPowers);
        setFinalResult(result);
        setHistory((prev) => [result, ...prev].slice(0, 50));
        setApiStatus("success");

        console.log("✅ CSV Processing Complete:", result);
      }

    } catch (error) {
      console.error("❌ CSV Processing Error:", error);
      alert("❌ Error processing CSV file: " + error.message);
      setApiStatus("error");
    } finally {
      setCsvProcessing(false);
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
          console.log("⏰ Timer reached 1:00, stopping recording...");
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
      const emotionDurations = {};
      const emotionOccurrences = {};
      
      trackingData.forEach((entry, index) => {
        const emotion = entry.emotion;
        const duration = index < trackingData.length - 1 
          ? (trackingData[index + 1].elapsed - entry.elapsed)
          : (RECORDING_DURATION - entry.elapsed);
        
        emotionDurations[emotion] = (emotionDurations[emotion] || 0) + duration;
        emotionOccurrences[emotion] = (emotionOccurrences[emotion] || 0) + 1;
      });
      
      const avgArousal = trackingData.reduce((sum, entry) => sum + entry.arousal, 0) / trackingData.length;
      const avgValence = trackingData.reduce((sum, entry) => sum + entry.valence, 0) / trackingData.length;
      
      const avgBandPowers = {
        delta: trackingData.reduce((sum, e) => sum + (e.bandPowers.delta || 0), 0) / trackingData.length,
        theta: trackingData.reduce((sum, e) => sum + (e.bandPowers.theta || 0), 0) / trackingData.length,
        alpha: trackingData.reduce((sum, e) => sum + (e.bandPowers.alpha || 0), 0) / trackingData.length,
        beta: trackingData.reduce((sum, e) => sum + (e.bandPowers.beta || 0), 0) / trackingData.length,
        gamma: trackingData.reduce((sum, e) => sum + (e.bandPowers.gamma || 0), 0) / trackingData.length,
      };
      
      const dominantEmotion = Object.entries(emotionDurations)
        .sort(([, a], [, b]) => b - a)[0][0];
      
      const avgProbabilities = {};
      Object.keys(LABEL_NAMES).forEach(emotionNum => {
        const probs = trackingData
          .map(e => e.probabilities?.[emotionNum] || 0)
          .filter(p => p > 0);
        avgProbabilities[emotionNum] = probs.length > 0
          ? probs.reduce((sum, p) => sum + p, 0) / probs.length
          : 0;
      });
      
      const report = {
        prediction: dominantEmotion,
        probabilities: avgProbabilities,
        arousal: avgArousal,
        valence: avgValence,
        bandPowers: avgBandPowers,
        timestamp: new Date().toLocaleTimeString(),
        sampleCount: samplesBufferRef.current.length,
        duration: elapsed,
        emotionDurations,
        emotionOccurrences,
        dominantEmotion,
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
        arousal,
        valence,
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

  const bandChartData = BANDS.map(({ key, label, color }) => ({
    name: label.split(" ")[0],
    value: parseFloat((bandPowers[key] || 0).toFixed(6)),
    color,
  }));

  const getEmotionColor = (emotion) => {
    if (!emotion) return "gray";
    const emotionNum = Object.entries(LABEL_NAMES).find(([, name]) => name === emotion)?.[0];
    return EMOTION_COLORS[emotionNum] || "#6B7280";
  };

  const emotionColor = getEmotionColor(prediction);

  const emotionRadarData = probabilities ? Object.entries(LABEL_NAMES).map(([num, name]) => ({
    emotion: name,
    probability: (probabilities[num] || 0) * 100
  })) : [];

  return (
    <div className="p-6 bg-gray-950 text-white min-h-screen">
      <div className="flex items-center gap-4 mb-8">
        <h1 className="text-3xl font-bold">😊 EEG Mood & Emotion Tracker</h1>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card title="📡 Device Setup & Connection">
            <div className="mb-4 flex items-center gap-4 bg-gray-800 p-3 rounded-lg">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={useCSV}
                  onChange={(e) => setUseCSV(e.target.checked)}
                  className="w-5 h-5 cursor-pointer"
                />
                <span className="text-white font-semibold">Use CSV File (Testing Mode)</span>
              </label>
            </div>

            {useCSV ? (
              <div className="space-y-4">
                <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-4">
                  <div className="text-blue-300 font-semibold mb-2">📁 CSV Upload Mode</div>
                  <div className="text-sm text-gray-300 mb-3">
                    Upload a CSV file with collected EEG data for testing
                  </div>
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleCSVUpload}
                    className="block w-full text-sm text-gray-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-600 file:text-white hover:file:bg-blue-700 file:cursor-pointer"
                  />
                  {csvFile && (
                    <div className="mt-3 text-sm text-green-400">
                      ✅ Selected: {csvFile.name} ({(csvFile.size / 1024).toFixed(2)} KB)
                    </div>
                  )}
                </div>

                <button
                  onClick={processCSVData}
                  disabled={!csvFile || csvProcessing}
                  className="w-full px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed font-semibold text-lg"
                >
                  {csvProcessing ? "🔄 Processing CSV..." : "▶️ Process CSV Data"}
                </button>

                <button
                  onClick={testApi}
                  className="w-full px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition font-semibold"
                >
                  🔧 Test API
                </button>
              </div>
            ) : (
              <>
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
                <code className="block bg-black/30 p-2 rounded mt-2">python app_emotion.py</code>
              </div>
            )}
              </>
            )}
          </Card>

          <Card title="🎬 Recording Controls">
            {!useCSV && (
              <>
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
              </>
            )}
            
            {useCSV && (
              <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-6 text-center">
                <div className="text-blue-300 text-lg font-semibold mb-2">📊 CSV Mode Active</div>
                <div className="text-gray-300 text-sm">
                  Upload and process your CSV file in the "Device Setup" section above
                </div>
              </div>
            )}
          </Card>

          <Card>
            {recordingComplete && finalResult && (
              <div className="mb-6 bg-gradient-to-r from-purple-900/30 to-pink-900/30 border-2 border-purple-500/50 rounded-xl p-6 animate-pulse">
                <div className="text-center">
                  <div className="text-purple-400 text-2xl font-bold mb-2">🎉 Recording Complete!</div>
                  <div className="text-gray-300 text-sm mb-4">1-minute session finished</div>
                  <div className="grid grid-cols-2 gap-4 text-left bg-gray-800/50 rounded-lg p-4">
                    <div>
                      <div className="text-gray-400 text-xs">Dominant Emotion</div>
                      <div className="text-white text-lg font-bold">{finalResult.prediction}</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Arousal Level</div>
                      <div className="text-pink-400 text-lg font-bold">{finalResult.arousal?.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Valence Level</div>
                      <div className="text-yellow-400 text-lg font-bold">{finalResult.valence?.toFixed(1)}</div>
                    </div>
                    <div>
                      <div className="text-gray-400 text-xs">Samples</div>
                      <div className="text-white text-lg font-bold">{finalResult.sampleCount?.toLocaleString()}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div className="text-center py-8">
              {prediction ? (
                <div>
                  <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Current Emotion</div>
                  <div className="text-6xl font-bold mb-6" style={{ color: emotionColor }}>
                    {prediction}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-6 max-w-2xl mx-auto mb-8">
                    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                      <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Arousal</div>
                      <div className="text-4xl font-bold text-pink-400 mb-2">
                        {arousal.toFixed(1)}
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-pink-500 to-pink-400 transition-all duration-1000"
                          style={{ width: `${Math.min(100, Math.max(0, arousal))}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                      <div className="text-sm text-gray-400 mb-2 uppercase tracking-wide">Valence</div>
                      <div className="text-4xl font-bold text-yellow-400 mb-2">
                        {valence.toFixed(1)}
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-1000"
                          style={{ width: `${Math.min(100, Math.max(0, valence))}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {probabilities && emotionRadarData.length > 0 && (
                    <div className="mt-8">
                      <div className="text-sm text-gray-400 mb-4 uppercase tracking-wide">Emotion Distribution</div>
                      <ResponsiveContainer width="100%" height={300}>
                        <RadarChart data={emotionRadarData}>
                          <PolarGrid stroke="#374151" />
                          <PolarAngleAxis dataKey="emotion" stroke="#9CA3AF" />
                          <PolarRadiusAxis angle={90} domain={[0, 100]} stroke="#9CA3AF" />
                          <Radar name="Probability" dataKey="probability" stroke="#8B5CF6" fill="#8B5CF6" fillOpacity={0.6} />
                          <Tooltip
                            contentStyle={{ 
                              backgroundColor: "#1F2937", 
                              border: "1px solid #374151", 
                              borderRadius: "8px",
                              color: "#fff"
                            }}
                          />
                        </RadarChart>
                      </ResponsiveContainer>
                      
                      <div className="grid grid-cols-5 gap-2 mt-6">
                        {Object.entries(LABEL_NAMES).map(([num, name]) => (
                          <div key={num} className="bg-gray-800 p-3 rounded-lg text-center border border-gray-700">
                            <div className="text-gray-400 text-xs mb-1">{name}</div>
                            <div className="text-white font-bold text-lg">
                              {((probabilities[num] || 0) * 100).toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-gray-500 py-16">
                  <div className="text-8xl mb-6">😊</div>
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
            
            <div className="grid grid-cols-5 gap-3 mt-6 text-xs">
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
          <Card title="📜 Session History">
            {history.length > 0 ? (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2">
                {history.map((h, i) => {
                  const histColor = getEmotionColor(h.prediction);
                  return (
                    <div 
                      key={i} 
                      className="bg-gray-800 p-4 rounded-lg border-l-4 hover:bg-gray-750 transition" 
                      style={{ borderColor: histColor }}
                    >
                      <div className="font-bold text-lg mb-2" style={{ color: histColor }}>
                        {h.prediction}
                      </div>
                      <div className="text-sm text-gray-300 mb-1">
                        Arousal: <span className="text-pink-400 font-semibold">{h.arousal?.toFixed(1)}</span>
                        {" | "}
                        Valence: <span className="text-yellow-400 font-semibold">{h.valence?.toFixed(1)}</span>
                      </div>
                      <div className="text-xs text-gray-400 mb-2">
                        Samples: {h.sampleCount?.toLocaleString() || 'N/A'}
                      </div>
                      <div className="text-xs text-gray-500">{h.timestamp}</div>
                    </div>
                  );
                })}
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
                <div>Verify the Flask server is running on port 8001</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">3. Start Recording</div>
                <div>Record for 1 minute to detect your emotional state</div>
              </div>
              <div className="bg-gray-800 p-3 rounded">
                <div className="font-semibold text-white mb-1">4. View Results</div>
                <div>Real-time emotion predictions appear every 3 seconds</div>
              </div>
            </div>
          </Card>
          
          <Card title="🎭 Emotion Guide">
            <div className="space-y-2 text-sm">
              {Object.entries(LABEL_NAMES).map(([num, name]) => (
                <div key={num} className="flex items-center gap-3 bg-gray-800 p-3 rounded-lg">
                  <div 
                    className="w-4 h-4 rounded-full" 
                    style={{ backgroundColor: EMOTION_COLORS[num] }}
                  />
                  <div className="font-semibold text-white">{name}</div>
                </div>
              ))}
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