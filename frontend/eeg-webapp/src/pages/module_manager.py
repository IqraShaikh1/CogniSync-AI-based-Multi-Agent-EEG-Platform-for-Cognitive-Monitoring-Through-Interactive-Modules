# module_manager.py
# ============================================================
# CogniSync Module Manager  –  port 9000
# ONE module at a time, all modules bind to port 8000
# ============================================================
#
# FIXES vs previous version
# --------------------------
# 1. stdout/stderr now go to a background reader thread → no
#    pipe-buffer deadlock that was killing child processes
# 2. All child output is printed live to this terminal so you
#    can see exactly why a module fails to start
# 3. /manager/logs/<key> endpoint returns last 200 log lines
# 4. Start response includes last log lines when crash happens
# 5. _current_key is cleared atomically when process dies
#
# REQUIRED CHANGE IN EVERY app.py
# --------------------------------
# Replace the last line of every app.py:
#
#   port = int(os.environ.get('PORT', <your_default_port>))
#   app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)
#
# use_reloader=False stops Flask from forking a second child.
# ============================================================

import subprocess
import sys
import os
import time
import threading
import collections
import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODULE_PORT = 8000   # every module shares this single port

# ─── Module registry ────────────────────────────────────────
MODULES = {
    "mental-health": {
        "name":        "Mental Health Detection",
        "script":      "mental_health_app.py",
        "description": "Detects mental health states from EEG band powers",
        "color":       "#EC4899",
    },
    "focus-tracking": {
        "name":        "Focus and Attention Tracking",
        "script":      "app.py",
        "description": "Tracks focus, distraction and baseline states",
        "color":       "#3B82F6",
    },
    "fatigue": {
        "name":        "Fatigue Detection",
        "script":      "fatigue_app.py",
        "description": "Detects fatigue and drowsiness from EEG signals",
        "color":       "#F59E0B",
    },
    "sleep-monitoring": {
        "name":        "Sleep Stage Monitoring",
        "script":      "sleep_app.py",
        "description": "Classifies sleep stages (Wake, N1, N2, N3, REM)",
        "color":       "#6366F1",
    },
    "meditation": {
        "name":        "Meditation Assistant",
        "script":      "meditation_app.py",
        "description": "Guides and measures depth of meditation sessions",
        "color":       "#10B981",
    },
    "brain-games": {
        "name":        "Brain-Controlled Games",
        "script":      "brain_games_app.py",
        "description": "Real-time EEG-based game control interface",
        "color":       "#8B5CF6",
    },
    "mood-emotion": {
        "name":        "Mood and Emotion Recognition",
        "script":      "app_emotion.py",
        "description": "Recognises emotional states from EEG patterns",
        "color":       "#F97316",
    },
    "brain-journal": {
        "name":        "Daily Brain Journal",
        "script":      "journal_app.py",
        "description": "Logs and trends daily cognitive and mood states",
        "color":       "#14B8A6",
    },
    "seizure-alerts": {
        "name":        "Seizure and Abnormal Activity Alerts",
        "script":      "app_seizure.py",
        "description": "Detects seizure activity and raises alerts",
        "color":       "#EF4444",
    },
}

# ─── Runtime state ──────────────────────────────────────────
_state_lock    = threading.Lock()
_current_key   = None                          # which module is active
_current_proc  = None                          # subprocess.Popen object
_log_buffer    = collections.deque(maxlen=200) # last 200 log lines
_log_lock      = threading.Lock()
_reader_thread = None                          # background stdout reader


# ─── Log helpers ────────────────────────────────────────────

def _log(line: str):
    """Print to terminal AND store in ring buffer."""
    print(line, flush=True)
    with _log_lock:
        _log_buffer.append(line)


def _get_logs() -> list:
    with _log_lock:
        return list(_log_buffer)


def _start_reader(proc: subprocess.Popen, module_name: str):
    """Background thread: drain stdout+stderr so the pipe never fills."""
    def _read():
        try:
            for line in proc.stdout:
                _log(f"[{module_name}] {line.rstrip()}")
        except Exception:
            pass
    t = threading.Thread(target=_read, daemon=True)
    t.start()
    return t


# ─── Helpers ────────────────────────────────────────────────

def _script_path(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def _proc_alive() -> bool:
    with _state_lock:
        p = _current_proc
    return p is not None and p.poll() is None


def _ping(timeout: float = 1.5) -> bool:
    for path in ("/health", "/test"):
        try:
            r = requests.get(
                f"http://127.0.0.1:{MODULE_PORT}{path}", timeout=timeout
            )
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False


def _wait_for_ready(retries: int = 30, delay: float = 0.7) -> bool:
    """
    Poll port 8000 until the server responds.
    Max wait ≈ 30 × 0.7 = 21 seconds.
    Returns (ready: bool, crashed: bool).
    """
    for i in range(retries):
        if _ping():
            _log(f"   ✅ Module ready on port {MODULE_PORT} (poll #{i + 1})")
            return True
        if not _proc_alive():
            _log("   ❌ Process exited before becoming ready")
            return False
        time.sleep(delay)
    _log(f"   ⚠️  Module still not responding after {retries} polls — treating as 'starting'")
    return False


def _kill_current():
    """Kill the active module process and clean up state."""
    global _current_key, _current_proc, _reader_thread

    with _state_lock:
        proc = _current_proc
        key  = _current_key

    if proc is None or proc.poll() is not None:
        with _state_lock:
            _current_key  = None
            _current_proc = None
        return

    name = MODULES.get(key, {}).get("name", key or "?")
    _log(f"⏹️  Stopping {name} (PID {proc.pid}) ...")

    proc.terminate()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    _log(f"   ✅ {name} stopped")
    with _state_lock:
        _current_key   = None
        _current_proc  = None
        _reader_thread = None


def _build_module_list() -> dict:
    with _state_lock:
        active_key = _current_key
    alive  = _proc_alive()
    online = _ping(timeout=0.4) if alive else False

    result = {}
    for key, cfg in MODULES.items():
        script_exists = os.path.isfile(_script_path(cfg["script"]))

        if not script_exists:
            state = "missing"
        elif key == active_key and alive and online:
            state = "running"
        elif key == active_key and alive and not online:
            state = "starting"
        else:
            state = "stopped"

        with _state_lock:
            proc = _current_proc if (key == active_key and alive) else None

        result[key] = {
            "key":           key,
            "name":          cfg["name"],
            "port":          MODULE_PORT,
            "script":        cfg["script"],
            "description":   cfg["description"],
            "color":         cfg["color"],
            "state":         state,
            "script_exists": script_exists,
            "pid":           proc.pid if proc else None,
        }
    return result


# ─── Core start / stop ──────────────────────────────────────

def _start(key: str) -> dict:
    global _current_key, _current_proc, _reader_thread

    if key not in MODULES:
        return {"error": f"Unknown module key: '{key}'"}

    cfg    = MODULES[key]
    script = _script_path(cfg["script"])

    if not os.path.isfile(script):
        return {
            "error":  (
                f"Script '{cfg['script']}' not found. "
                f"Make sure it is in the same folder as module_manager.py"
            ),
            "state":  "missing",
            "module": key,
        }

    # Already running the same module?
    with _state_lock:
        same = (_current_key == key)
    if same and _proc_alive() and _ping():
        return {
            "message": f"{cfg['name']} is already running on port {MODULE_PORT}",
            "state":   "running",
            "port":    MODULE_PORT,
            "module":  key,
        }

    # Stop whatever is currently running
    _kill_current()
    time.sleep(0.8)   # give the OS a moment to free port 8000

    _log(f"\n▶️  Starting {cfg['name']} on port {MODULE_PORT} ...")
    _log(f"   Script : {script}")

    # ── Clear log buffer so new module starts fresh ──
    with _log_lock:
        _log_buffer.clear()

    # Build environment
    env = os.environ.copy()
    env["PORT"]               = str(MODULE_PORT)
    env["FLASK_DEBUG"]        = "0"
    env["FLASK_ENV"]          = "production"
    # Fix Windows cp1252 UnicodeEncodeError when app.py prints Unicode (✓ ✅ etc.)
    env["PYTHONIOENCODING"]   = "utf-8"
    env["PYTHONUTF8"]         = "1"

    try:
        proc = subprocess.Popen(
            [sys.executable, script],
            env=env,
            # Merge stdout+stderr into one stream; reader thread drains it
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",   # Windows fix: force UTF-8 pipe decoding
            errors="replace",   # replace any remaining bad chars instead of crashing
            bufsize=1,           # line-buffered
        )
    except Exception as e:
        _log(f"   ❌ Failed to launch process: {e}")
        return {"error": str(e), "module": key}

    with _state_lock:
        _current_key   = key
        _current_proc  = proc
        _reader_thread = _start_reader(proc, cfg["name"])

    ready = _wait_for_ready()

    # Check if process is still alive after polling
    if not _proc_alive():
        # Collect whatever the reader captured so far
        time.sleep(0.3)   # let reader drain a bit more
        logs = _get_logs()
        crash_output = "\n".join(logs[-30:]) if logs else "(no output captured)"
        _log(f"\n   ❌ {cfg['name']} crashed. Last output:\n{crash_output}")
        with _state_lock:
            _current_key  = None
            _current_proc = None
        return {
            "error":       f"{cfg['name']} crashed on startup. Check the terminal for the full traceback.",
            "crash_output": crash_output,
            "state":       "stopped",
            "module":      key,
        }

    state = "running" if ready else "starting"
    return {
        "message": f"{cfg['name']} started (PID {proc.pid})",
        "state":   state,
        "port":    MODULE_PORT,
        "pid":     proc.pid,
        "module":  key,
    }


def _stop(key: str) -> dict:
    with _state_lock:
        active = _current_key

    if key not in MODULES:
        return {"error": f"Unknown module: '{key}'"}

    if active != key or not _proc_alive():
        return {
            "message": f"{MODULES[key]['name']} is not currently running",
            "state":   "stopped",
            "module":  key,
        }

    _kill_current()
    return {
        "message": f"{MODULES[key]['name']} stopped",
        "state":   "stopped",
        "module":  key,
    }


# ─── REST API ───────────────────────────────────────────────

@app.route("/manager/health", methods=["GET"])
def manager_health():
    return jsonify({
        "status":        "healthy",
        "manager_port":  9000,
        "module_port":   MODULE_PORT,
        "active_module": _current_key,
    })


@app.route("/manager/status", methods=["GET"])
def all_status():
    with _state_lock:
        active_key = _current_key
    alive  = _proc_alive()
    online = _ping(timeout=0.4) if alive else False

    active_info = None
    if active_key and alive:
        active_info = {
            "key":   active_key,
            "name":  MODULES[active_key]["name"],
            "state": "running" if online else "starting",
            "port":  MODULE_PORT,
        }

    return jsonify({
        "modules":     _build_module_list(),
        "active":      active_info,
        "module_port": MODULE_PORT,
        "status":      "success",
    })


@app.route("/manager/status/<key>", methods=["GET"])
def single_status(key):
    if key not in MODULES:
        return jsonify({"error": f"Unknown module '{key}'"}), 404
    return jsonify(_build_module_list()[key])


@app.route("/manager/active", methods=["GET"])
def active_module():
    with _state_lock:
        key = _current_key
    alive  = _proc_alive()
    online = _ping(timeout=0.4) if alive else False
    if not key or not alive:
        return jsonify({"active_module": None, "state": "idle", "port": MODULE_PORT})
    return jsonify({
        "active_module": key,
        "name":          MODULES[key]["name"],
        "state":         "running" if online else "starting",
        "port":          MODULE_PORT,
    })


@app.route("/manager/logs", methods=["GET"])
def get_logs():
    """Return last 200 log lines from the active (or most recent) module."""
    return jsonify({"logs": _get_logs(), "status": "success"})


@app.route("/manager/start/<key>", methods=["POST"])
def start_module(key):
    result = _start(key)
    if result.get("state") == "missing" or "not found" in result.get("error", ""):
        return jsonify(result), 404
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200


@app.route("/manager/stop/<key>", methods=["POST"])
def stop_module(key):
    result = _stop(key)
    if "Unknown module" in result.get("error", ""):
        return jsonify(result), 404
    return jsonify(result), 200


@app.route("/manager/stop-active", methods=["POST"])
def stop_active():
    with _state_lock:
        key = _current_key
    if not key:
        return jsonify({"message": "No module is running", "state": "idle"})
    return jsonify(_stop(key))


@app.route("/manager/restart/<key>", methods=["POST"])
def restart_module(key):
    _stop(key)
    time.sleep(0.5)
    result = _start(key)
    return jsonify(result), (500 if "error" in result else 200)


@app.route("/manager/stop-all", methods=["POST"])
def stop_all():
    with _state_lock:
        key = _current_key
    result = _stop(key) if key else {"message": "Nothing was running", "state": "idle"}
    return jsonify({"result": result, "status": "success"})


# ─── Proxy ──────────────────────────────────────────────────

@app.route("/proxy/", defaults={"path": ""}, methods=["GET", "POST", "PUT", "DELETE"])
@app.route("/proxy/<path:path>",              methods=["GET", "POST", "PUT", "DELETE"])
def proxy(path):
    if not _proc_alive():
        return jsonify({
            "error": "No module is running. Start one from the Module Manager.",
            "state": "idle",
        }), 503

    url     = f"http://127.0.0.1:{MODULE_PORT}/{path}"
    headers = {k: v for k, v in request.headers if k.lower() != "host"}

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            params=request.args,
            files={k: (f.filename, f.stream, f.content_type)
                   for k, f in request.files.items()},
            timeout=30,
        )
        return Response(resp.content, status=resp.status_code, headers=dict(resp.headers))
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot reach active module on port {MODULE_PORT}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─── Cleanup ────────────────────────────────────────────────

import atexit

@atexit.register
def _cleanup():
    _log("\n🔴 Manager exiting — stopping active module ...")
    _kill_current()


# ─── Entry point ────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("🧠  CogniSync Module Manager")
    print("=" * 65)
    print(f"  Manager port  :  9000")
    print(f"  Module port   :  {MODULE_PORT}  ← all modules share this")
    print(f"  Mode          :  ONE module active at a time")
    print()
    print("  Modules:")
    for key, cfg in MODULES.items():
        found = "✅" if os.path.isfile(_script_path(cfg["script"])) else "❌ MISSING"
        print(f"    {key:<22}  →  {cfg['script']:<30} {found}")
    print()
    print("  API:")
    print("    GET  /manager/status          all module states")
    print("    GET  /manager/active          currently active module")
    print("    GET  /manager/logs            live output from active module")
    print("    POST /manager/start/<key>     start a module (auto-stops current)")
    print("    POST /manager/stop/<key>      stop a module")
    print("    POST /manager/stop-active     stop whatever is running")
    print("    POST /manager/restart/<key>   restart")
    print("    POST /manager/stop-all        stop all")
    print("    ANY  /proxy/<path>            proxy to active module on :8000")
    print()
    print("  ⚠️  REQUIRED in every app.py — change the last line to:")
    print("     port = int(os.environ.get('PORT', <your_default_port>))")
    print("     app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False)")
    print("=" * 65 + "\n")

    app.run(host="127.0.0.1", port=9000, debug=False, use_reloader=False)