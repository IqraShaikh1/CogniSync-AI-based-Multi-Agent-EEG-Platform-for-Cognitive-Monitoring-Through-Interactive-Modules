// pages/ModuleManager.jsx
import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  HeartPulse, Focus, BatteryCharging, Moon, Brain,
  Gamepad2, Smile, NotebookText, AlertTriangle,
  Play, Square, RotateCw, Activity, Server,
  CheckCircle2, XCircle, Loader2, AlertCircle, FileX,
  ChevronRight, Radio, Terminal, X,
} from "lucide-react";

const MANAGER_URL = "http://127.0.0.1:9000";
const POLL_MS     = 3000;
const LOG_POLL_MS = 2000;

const MODULE_META = {
  "mental-health":    { icon: HeartPulse,     color: "#EC4899", path: "/mental-health"    },
  "focus-tracking":   { icon: Focus,           color: "#3B82F6", path: "/focus-tracking"   },
  "fatigue":          { icon: BatteryCharging, color: "#F59E0B", path: "/fatigue"           },
  "sleep-monitoring": { icon: Moon,            color: "#6366F1", path: "/sleep-monitoring"  },
  "meditation":       { icon: Brain,           color: "#10B981", path: "/meditation"        },
  "brain-games":      { icon: Gamepad2,        color: "#8B5CF6", path: "/brain-games"       },
  "mood-emotion":     { icon: Smile,           color: "#F97316", path: "/mood-emotion"      },
  "brain-journal":    { icon: NotebookText,    color: "#14B8A6", path: "/brain-journal"     },
  "seizure-alerts":   { icon: AlertTriangle,   color: "#EF4444", path: "/seizure-alerts"    },
};

// ─── API helper ─────────────────────────────────────────────
async function managerApi(method, path) {
  try {
    const res = await fetch(`${MANAGER_URL}${path}`, {
      method,
      headers: { "Content-Type": "application/json" },
    });
    return await res.json();
  } catch (e) {
    return { error: e.message };
  }
}

// ─── State chip ─────────────────────────────────────────────
function StateChip({ state }) {
  const map = {
    running:  { bg: "bg-green-900/50",  border: "border-green-500/70",  text: "text-green-300",
                icon: <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse inline-block" />, label: "Running"   },
    starting: { bg: "bg-yellow-900/40", border: "border-yellow-500/60", text: "text-yellow-300",
                icon: <Loader2 className="w-3 h-3 animate-spin" />,                                      label: "Starting…" },
    stopped:  { bg: "bg-gray-800",      border: "border-gray-700",      text: "text-gray-400",
                icon: <XCircle className="w-3 h-3" />,                                                   label: "Stopped"   },
    missing:  { bg: "bg-red-900/30",    border: "border-red-700/60",    text: "text-red-400",
                icon: <FileX className="w-3 h-3" />,                                                     label: "Missing"   },
    error:    { bg: "bg-red-900/50",    border: "border-red-500/70",    text: "text-red-300",
                icon: <AlertCircle className="w-3 h-3" />,                                               label: "Error"     },
  };
  const cfg = map[state] || map.stopped;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border
      text-xs font-semibold ${cfg.bg} ${cfg.border} ${cfg.text}`}>
      {cfg.icon} {cfg.label}
    </span>
  );
}

// ─── Module Card ─────────────────────────────────────────────
function ModuleCard({ modKey, data, activeKey, lastError, onStart, onStop, onShowLogs, busy }) {
  const meta     = MODULE_META[modKey] || {};
  const Icon     = meta.icon || Activity;
  const color    = meta.color || "#6B7280";
  const state    = data?.state || "stopped";
  const name     = data?.name  || modKey;
  const desc     = data?.description || "";
  const pid      = data?.pid;
  const missing  = state === "missing";
  const running  = state === "running";
  const starting = state === "starting";
  const isActive = modKey === activeKey;
  const isBusy   = busy === modKey;
  const hasError = lastError?.key === modKey;

  return (
    <div className={`relative bg-gray-900 rounded-xl p-5 border transition-all duration-200 overflow-hidden
      ${isActive && (running || starting)
        ? "border-2 shadow-xl"
        : hasError
          ? "border border-red-700/60"
          : "border-gray-800 hover:border-gray-700"}
    `}
      style={isActive && (running || starting)
        ? { borderColor: color, boxShadow: `0 0 24px ${color}30` }
        : {}
      }
    >
      {/* Top glow strip when active */}
      {isActive && running && (
        <div className="absolute top-0 left-0 right-0 h-0.5"
          style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }} />
      )}

      {/* ACTIVE badge */}
      {isActive && (running || starting) && (
        <div className="absolute top-3 right-3 flex items-center gap-1 px-2 py-0.5
          rounded-full bg-black/50 border text-xs font-bold"
          style={{ borderColor: color + "99", color }}>
          <Radio className="w-2.5 h-2.5" /> ACTIVE
        </div>
      )}

      {/* Header */}
      <div className="flex items-start gap-3 mb-3 pr-20">
        <div className="p-2 rounded-lg flex-shrink-0" style={{ backgroundColor: color + "22" }}>
          <Icon className="w-5 h-5" style={{ color }} />
        </div>
        <div className="min-w-0">
          <div className="font-semibold text-sm text-white leading-tight">{name}</div>
          <div className="text-xs text-gray-500 mt-0.5 font-mono">
            :8000{pid ? ` · PID ${pid}` : ""}
          </div>
        </div>
      </div>

      {/* State chip */}
      {(!isActive || (!running && !starting)) && (
        <div className="mb-3">
          <StateChip state={hasError ? "error" : state} />
        </div>
      )}

      {/* Description */}
      <p className="text-xs text-gray-500 mb-4 leading-relaxed min-h-[2.2rem]">{desc}</p>

      {/* Error message */}
      {hasError && (
        <div className="mb-3 bg-red-900/20 border border-red-700/50 rounded-lg p-3 text-xs">
          <div className="text-red-400 font-semibold mb-1">❌ Startup failed</div>
          <div className="text-red-300/80 leading-relaxed break-words">
            {lastError.message}
          </div>
          {lastError.crash_output && (
            <button onClick={() => onShowLogs(modKey, lastError.crash_output)}
              className="mt-2 flex items-center gap-1 text-red-400 hover:text-red-300
                underline underline-offset-2 text-xs">
              <Terminal className="w-3 h-3" /> View crash output
            </button>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex gap-2">
        {!running && !starting ? (
          <button
            onClick={() => onStart(modKey)}
            disabled={missing || isBusy}
            className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg
              text-xs font-semibold bg-green-700/80 hover:bg-green-600 text-white
              transition disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {isBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
            {isBusy ? "Starting…" : "Start"}
          </button>
        ) : (
          <>
            <button
              onClick={() => onStop(modKey)}
              disabled={isBusy}
              className="flex-1 flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg
                text-xs font-semibold bg-red-700/80 hover:bg-red-600 text-white
                transition disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Square className="w-3.5 h-3.5" />}
              {isBusy ? "Stopping…" : "Stop"}
            </button>
            <button onClick={() => onStart(modKey)} disabled={isBusy} title="Restart"
              className="px-3 py-2 rounded-lg text-xs bg-gray-700 hover:bg-gray-600 text-white
                transition disabled:opacity-40">
              <RotateCw className="w-3.5 h-3.5" />
            </button>
          </>
        )}
        {running && meta.path && (
          <a href={meta.path}
            className="px-3 py-2 rounded-lg text-xs font-semibold bg-blue-700/70
              hover:bg-blue-600 text-white transition flex items-center gap-1">
            Open <ChevronRight className="w-3 h-3" />
          </a>
        )}
        {(running || starting) && (
          <button onClick={() => onShowLogs(modKey, null)} title="View logs"
            className="px-3 py-2 rounded-lg text-xs bg-gray-700 hover:bg-gray-600
              text-white transition">
            <Terminal className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      {/* Missing script */}
      {missing && (
        <div className="mt-3 text-xs text-red-400/80 bg-red-900/20 border border-red-800/40
          rounded px-2 py-1.5 font-mono">
          {data?.script} — not found in manager folder
        </div>
      )}
    </div>
  );
}

// ─── Log Modal ───────────────────────────────────────────────
function LogModal({ title, lines, onClose, liveKey }) {
  const bottomRef  = useRef(null);
  const [logs, setLogs] = useState(lines || []);

  // Poll live logs if a live key is given
  useEffect(() => {
    if (!liveKey) return;
    const id = setInterval(async () => {
      const data = await managerApi("GET", "/manager/logs");
      if (data.logs) setLogs(data.logs);
    }, LOG_POLL_MS);
    return () => clearInterval(id);
  }, [liveKey]);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70">
      <div className="bg-gray-900 border border-gray-700 rounded-xl w-full max-w-3xl
        max-h-[80vh] flex flex-col shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-800">
          <div className="flex items-center gap-2 font-semibold text-white">
            <Terminal className="w-4 h-4 text-green-400" />
            {title}
            {liveKey && (
              <span className="flex items-center gap-1 text-xs text-green-400
                bg-green-900/30 border border-green-700/50 px-2 py-0.5 rounded-full ml-2">
                <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
                live
              </span>
            )}
          </div>
          <button onClick={onClose}
            className="p-1 rounded hover:bg-gray-700 text-gray-400 hover:text-white transition">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="overflow-y-auto flex-1 p-4 font-mono text-xs leading-relaxed">
          {logs.length === 0 ? (
            <div className="text-gray-600 text-center py-8">No output yet</div>
          ) : (
            logs.map((line, i) => (
              <div key={i}
                className={`${line.includes("Error") || line.includes("Traceback") || line.includes("❌")
                  ? "text-red-400"
                  : line.includes("✅") || line.includes("Running")
                    ? "text-green-400"
                    : line.includes("▶️") || line.includes("Starting")
                      ? "text-yellow-400"
                      : "text-gray-300"}`}>
                {line}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
        <div className="px-5 py-3 border-t border-gray-800 flex items-center justify-between">
          <span className="text-xs text-gray-600">{logs.length} lines</span>
          <button onClick={onClose}
            className="px-4 py-1.5 bg-gray-700 hover:bg-gray-600 text-white
              rounded-lg text-xs font-semibold transition">
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Main page ───────────────────────────────────────────────
export default function ModuleManager() {
  const [modules,       setModules]       = useState({});
  const [activeInfo,    setActiveInfo]    = useState(null);
  const [managerOnline, setManagerOnline] = useState(false);
  const [lastRefresh,   setLastRefresh]   = useState(null);
  const [busy,          setBusy]          = useState(null);
  const [toast,         setToast]         = useState(null);
  const [filter,        setFilter]        = useState("all");
  const [lastErrors,    setLastErrors]    = useState({});  // key → {key,message,crash_output}
  const [logModal,      setLogModal]      = useState(null);// {title, lines, liveKey}
  const pollRef = useRef(null);

  const showToast = (msg, type = "info") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 5000);
  };

  const fetchStatus = useCallback(async () => {
    const data = await managerApi("GET", "/manager/status");
    if (data.error) {
      setManagerOnline(false);
    } else {
      setManagerOnline(true);
      setModules(data.modules || {});
      setActiveInfo(data.active || null);
      setLastRefresh(new Date().toLocaleTimeString());
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    pollRef.current = setInterval(fetchStatus, POLL_MS);
    return () => clearInterval(pollRef.current);
  }, [fetchStatus]);

  const handleStart = async (key) => {
    setBusy(key);
    // Clear any previous error for this key
    setLastErrors(prev => { const n = { ...prev }; delete n[key]; return n; });

    const activeKey = activeInfo?.key;
    if (activeKey && activeKey !== key) {
      showToast(`⏹ Stopping ${modules[activeKey]?.name || activeKey}…`, "info");
    }
    showToast(`▶️ Starting ${modules[key]?.name || key}…`, "info");

    const res = await managerApi("POST", `/manager/start/${key}`);

    if (res.error) {
      const errMsg = res.error || "Unknown error";
      setLastErrors(prev => ({
        ...prev,
        [key]: { key, message: errMsg, crash_output: res.crash_output || null },
      }));
      showToast(`❌ ${errMsg}`, "error");

      // Auto-open log modal with crash output if available
      if (res.crash_output) {
        setLogModal({
          title:   `Crash output — ${modules[key]?.name || key}`,
          lines:   res.crash_output.split("\n"),
          liveKey: null,
        });
      }
    } else {
      showToast(`✅ ${res.message}`, "success");
    }

    await fetchStatus();
    setBusy(null);
  };

  const handleStop = async (key) => {
    setBusy(key);
    const res = await managerApi("POST", `/manager/stop/${key}`);
    if (res.error) showToast(`❌ ${res.error}`, "error");
    else showToast(`⏹ ${res.message}`, "info");
    await fetchStatus();
    setBusy(null);
  };

  const handleStopActive = async () => {
    if (!activeInfo) return;
    setBusy(activeInfo.key);
    const res = await managerApi("POST", "/manager/stop-active");
    showToast(`⏹ ${res.message || "Stopped"}`, "info");
    await fetchStatus();
    setBusy(null);
  };

  const handleShowLogs = async (key, staticLines) => {
    if (staticLines) {
      setLogModal({
        title:   `Crash output — ${modules[key]?.name || key}`,
        lines:   staticLines.split ? staticLines.split("\n") : staticLines,
        liveKey: null,
      });
    } else {
      // Fetch current logs then open live modal
      const data = await managerApi("GET", "/manager/logs");
      setLogModal({
        title:   `Live logs — ${modules[key]?.name || key}`,
        lines:   data.logs || [],
        liveKey: key,
      });
    }
  };

  const allKeys     = Object.keys(modules);
  const runningKeys = allKeys.filter(k => modules[k]?.state === "running");
  const missingKeys = allKeys.filter(k => modules[k]?.state === "missing");
  const errorKeys   = Object.keys(lastErrors);
  const activeKey   = activeInfo?.key || null;

  const filteredKeys = allKeys.filter(k => {
    if (filter === "running") return ["running", "starting"].includes(modules[k]?.state);
    if (filter === "stopped") return ["stopped", "missing", "error"].includes(modules[k]?.state);
    return true;
  });

  return (
    <div className="min-h-screen bg-gray-950 text-white p-6">

      {/* Log Modal */}
      {logModal && (
        <LogModal
          title={logModal.title}
          lines={logModal.lines}
          liveKey={logModal.liveKey}
          onClose={() => setLogModal(null)}
        />
      )}

      {/* Toast */}
      {toast && (
        <div className={`fixed top-5 right-5 z-40 px-5 py-3 rounded-xl shadow-2xl text-sm
          font-semibold border max-w-sm leading-relaxed
          ${toast.type === "error"   ? "bg-red-900/95 border-red-500/60 text-red-200"      :
            toast.type === "success" ? "bg-green-900/95 border-green-500/60 text-green-200" :
                                       "bg-gray-800 border-gray-600 text-white"}`}>
          {toast.msg}
        </div>
      )}

      {/* Header */}
      <div className="mb-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <div className="p-2 bg-blue-600/20 rounded-lg border border-blue-500/30">
              <Server className="w-6 h-6 text-blue-400" />
            </div>
            <h1 className="text-2xl font-bold">Module Manager</h1>
          </div>
          <p className="text-sm text-gray-500 ml-14">
            One module at a time · All share port 8000
          </p>
        </div>
        <div className="flex items-center gap-3 flex-wrap">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold
            ${managerOnline
              ? "bg-green-900/30 border-green-500/40 text-green-400"
              : "bg-red-900/30 border-red-500/40 text-red-400"}`}>
            <span className={`w-2 h-2 rounded-full ${managerOnline ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
            {managerOnline ? "Manager Online" : "Manager Offline"}
          </div>
          {lastRefresh && <span className="text-xs text-gray-600">Updated {lastRefresh}</span>}
          <button onClick={() => handleShowLogs(activeKey, null)}
            disabled={!managerOnline}
            title="View live logs"
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition disabled:opacity-40">
            <Terminal className="w-4 h-4 text-gray-400" />
          </button>
          <button onClick={fetchStatus} title="Refresh"
            className="p-2 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
            <RotateCw className="w-4 h-4 text-gray-400" />
          </button>
        </div>
      </div>

      {/* Manager offline */}
      {!managerOnline && (
        <div className="mb-6 bg-red-900/20 border border-red-500/40 rounded-xl p-5 flex items-start gap-4">
          <AlertCircle className="w-6 h-6 text-red-400 flex-shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold text-red-300 mb-1">Module Manager is offline</div>
            <div className="text-sm text-red-400/80 mb-3">Start it first:</div>
            <code className="block bg-black/40 border border-red-900/60 rounded-lg
              px-4 py-2 text-xs text-green-400 font-mono">
              python module_manager.py
            </code>
          </div>
        </div>
      )}

      {/* Active module banner */}
      {managerOnline && activeInfo && (
        <div className="mb-6 rounded-xl border-2 p-4 flex items-center justify-between gap-4"
          style={{
            borderColor: MODULE_META[activeInfo.key]?.color || "#6B7280",
            background:  (MODULE_META[activeInfo.key]?.color || "#6B7280") + "11",
          }}>
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 text-xs font-bold px-2 py-1
              rounded-full bg-black/30 border"
              style={{ borderColor: MODULE_META[activeInfo.key]?.color, color: MODULE_META[activeInfo.key]?.color }}>
              <Radio className="w-3 h-3" /> ACTIVE
            </div>
            <div>
              <div className="font-bold text-white">{activeInfo.name}</div>
              <div className="text-xs text-gray-400 font-mono">
                http://127.0.0.1:8000 · {activeInfo.state}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => handleShowLogs(activeInfo.key, null)}
              className="px-3 py-2 rounded-lg text-xs bg-gray-700 hover:bg-gray-600
                text-white transition flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5" /> Logs
            </button>
            {MODULE_META[activeInfo.key]?.path && (
              <a href={MODULE_META[activeInfo.key].path}
                className="px-4 py-2 rounded-lg text-xs font-semibold bg-blue-600
                  hover:bg-blue-500 text-white transition flex items-center gap-1">
                Open Module <ChevronRight className="w-3 h-3" />
              </a>
            )}
            <button onClick={handleStopActive} disabled={busy === activeInfo.key}
              className="px-4 py-2 rounded-lg text-xs font-semibold bg-red-700
                hover:bg-red-600 text-white transition disabled:opacity-50 flex items-center gap-1.5">
              {busy === activeInfo.key
                ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                : <Square className="w-3.5 h-3.5" />}
              Stop
            </button>
          </div>
        </div>
      )}

      {/* No active module */}
      {managerOnline && !activeInfo && (
        <div className="mb-6 rounded-xl border border-dashed border-gray-700
          p-4 text-center text-gray-500 text-sm">
          No module running ·
          Click <strong className="text-gray-300">Start</strong> on any module below to activate it
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
        {[
          { label: "Total",   value: allKeys.length,     color: "text-white",     bg: "bg-gray-800/80" },
          { label: "Running", value: runningKeys.length, color: "text-green-400", bg: "bg-green-900/20 border border-green-900/50" },
          { label: "Errors",  value: errorKeys.length,   color: errorKeys.length > 0 ? "text-red-400" : "text-gray-400",
            bg: errorKeys.length > 0 ? "bg-red-900/20 border border-red-900/50" : "bg-gray-800/80" },
          { label: "Missing", value: missingKeys.length, color: missingKeys.length > 0 ? "text-yellow-400" : "text-gray-400",
            bg: "bg-gray-800/80" },
        ].map(s => (
          <div key={s.label} className={`rounded-xl p-4 ${s.bg}`}>
            <div className={`text-2xl font-bold ${s.color}`}>{s.value}</div>
            <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <button onClick={handleStopActive}
          disabled={!managerOnline || !activeInfo || !!busy}
          className="flex items-center gap-2 px-4 py-2 bg-red-700/80 hover:bg-red-700
            text-white rounded-lg text-sm font-semibold transition
            disabled:opacity-40 disabled:cursor-not-allowed">
          <Square className="w-4 h-4" /> Stop Active
        </button>
        {errorKeys.length > 0 && (
          <button onClick={() => setLastErrors({})}
            className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600
              text-white rounded-lg text-sm transition">
            <X className="w-4 h-4" /> Clear Errors
          </button>
        )}
        <div className="ml-auto flex items-center gap-1 bg-gray-800 rounded-lg p-1">
          {["all", "running", "stopped"].map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`px-3 py-1 rounded-md text-xs font-semibold capitalize transition
                ${filter === f ? "bg-gray-600 text-white" : "text-gray-400 hover:text-white"}`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Module grid */}
      {!managerOnline ? (
        <div className="text-center text-gray-600 py-24">
          <Server className="w-12 h-12 mx-auto mb-4 opacity-30" />
          <div className="text-lg font-semibold mb-1">Waiting for Module Manager</div>
          <div className="text-sm">
            Run <code className="text-green-400 font-mono">python module_manager.py</code>
          </div>
        </div>
      ) : filteredKeys.length === 0 ? (
        <div className="text-center text-gray-600 py-20 text-sm">No modules match this filter.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredKeys.map(key => (
            <ModuleCard
              key={key}
              modKey={key}
              data={modules[key]}
              activeKey={activeKey}
              lastError={lastErrors[key] || null}
              onStart={handleStart}
              onStop={handleStop}
              onShowLogs={handleShowLogs}
              busy={busy}
            />
          ))}
        </div>
      )}

      {/* Footer — required app.py change */}
      {managerOnline && (
        <div className="mt-10 p-4 bg-gray-900 border border-gray-800 rounded-xl text-xs text-gray-500 space-y-2">
          <div className="font-semibold text-gray-400 mb-1">
            ⚠️ Required one-time change in every app.py
          </div>
          <div>Replace the last <code className="text-green-400">app.run()</code> line with:</div>
          <pre className="bg-black/50 rounded-lg p-3 text-green-400 font-mono overflow-x-auto">{
`port = int(os.environ.get('PORT', <your_default_port>))
app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)`
          }</pre>
          <div className="text-gray-600 text-xs mt-1">
            The manager passes <code className="text-yellow-400">PORT=8000</code> at launch.
            Without <code>use_reloader=False</code>, Flask forks a child process and the manager
            loses track of it — causing the "process died" error.
          </div>
        </div>
      )}
    </div>
  );
}