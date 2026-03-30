"""
api.py
------
Flask REST API for the Daily Brain Journal.
NO demo/seed data. All data comes strictly from real module predictions.
"""

import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from database import (
    init_db,
    insert_focus_session,
    insert_emotion_session,
    insert_seizure_session,
    get_combined_summary,
    get_focus_summary,
    get_emotion_summary,
    get_seizure_summary,
    get_table_rows,
    delete_row,
)

app = Flask(__name__)
CORS(app)

init_db()


# ─────────────────────────────────────────────
#  HEALTH
# ─────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# ─────────────────────────────────────────────
#  AUTO-INSERT (called by EEG modules)
# ─────────────────────────────────────────────

@app.route("/api/sessions/focus", methods=["POST"])
def add_focus():
    data = request.get_json()
    try:
        row_id = insert_focus_session(
            duration_s  = float(data.get("duration_s", 0)),
            focus_score = float(data.get("focus_score", 0)),
            focus_label = str(data.get("focus_label", "")),
            notes       = str(data.get("notes", ""))
        )
        return jsonify({"success": True, "id": row_id}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/emotion", methods=["POST"])
def add_emotion():
    data = request.get_json()
    try:
        scores = data.get("emotion_scores", {})
        row_id = insert_emotion_session(
            duration_s       = float(data.get("duration_s", 0)),
            dominant_emotion = str(data.get("dominant_emotion", "")),
            emotion_scores   = json.dumps(scores),
            notes            = str(data.get("notes", ""))
        )
        return jsonify({"success": True, "id": row_id}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sessions/seizure", methods=["POST"])
def add_seizure():
    data = request.get_json()
    try:
        row_id = insert_seizure_session(
            duration_s   = float(data.get("duration_s", 0)),
            detected     = bool(data.get("detected", False)),
            confidence   = float(data.get("confidence", 0)),
            seizure_type = str(data.get("seizure_type", "")),
            notes        = str(data.get("notes", ""))
        )
        return jsonify({"success": True, "id": row_id}), 201
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ─────────────────────────────────────────────
#  JOURNAL SUMMARY ENDPOINTS
# ─────────────────────────────────────────────

def _period_params():
    period    = request.args.get("period", "daily")
    date_from = request.args.get("date_from", None)
    date_to   = request.args.get("date_to", None)
    return period, date_from, date_to


@app.route("/api/journal/combined", methods=["GET"])
def journal_combined():
    period, date_from, date_to = _period_params()
    return jsonify(get_combined_summary(period, date_from, date_to))


@app.route("/api/journal/focus", methods=["GET"])
def journal_focus():
    period, date_from, date_to = _period_params()
    return jsonify(get_focus_summary(period, date_from, date_to))


@app.route("/api/journal/emotion", methods=["GET"])
def journal_emotion():
    period, date_from, date_to = _period_params()
    return jsonify(get_emotion_summary(period, date_from, date_to))


@app.route("/api/journal/seizure", methods=["GET"])
def journal_seizure():
    period, date_from, date_to = _period_params()
    return jsonify(get_seizure_summary(period, date_from, date_to))


# ─────────────────────────────────────────────
#  DB VIEWER ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/db/rows/<table>", methods=["GET"])
def db_rows(table):
    """Return raw rows for a table. Used by DB Viewer UI."""
    try:
        limit  = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        result = get_table_rows(table, limit, offset)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/db/rows/<table>/<int:row_id>", methods=["DELETE"])
def db_delete_row(table, row_id):
    """Delete a row by id. Used by DB Viewer UI."""
    try:
        delete_row(table, row_id)
        return jsonify({"success": True})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)