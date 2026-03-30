import { useState, useEffect, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  Tooltip, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Legend
} from "recharts";

// ─── TOKENS ──────────────────────────────────────────────────────────────────
const C = {
  bg:        "#04080F",
  card:      "#080F1C",
  panel:     "#0C1525",
  border:    "rgba(56,189,248,0.10)",
  borderHi:  "rgba(56,189,248,0.30)",
  focus:     "#38BDF8",
  focusGlow: "rgba(56,189,248,0.18)",
  emotion:   "#A78BFA",
  seizure:   "#F87171",
  safe:      "#34D399",
  warn:      "#FBBF24",
  text:      "#DDE8F5",
  dim:       "#5A7A9A",
  faint:     "rgba(90,122,154,0.45)",
  high:      "#34D399",
  medium:    "#FBBF24",
  low:       "#F87171",
};

const EMOTION_COLORS = {
  Happy:"#FFD700", Calm:"#38BDF8", Focused:"#A78BFA",
  Stressed:"#F97316", Anxious:"#F87171", Neutral:"#94A3B8",
};

const FONT  = "'DM Mono','Fira Code',monospace";
const FONTD = "'DM Sans','Outfit',system-ui,sans-serif";

// ─── API ─────────────────────────────────────────────────────────────────────
const API = "http://localhost:5000/api";

async function apiFetch(path) {
  const res = await fetch(API + path);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function apiDelete(path) {
  const res = await fetch(API + path, { method: "DELETE" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ─── SMALL SHARED COMPONENTS ─────────────────────────────────────────────────
function Card({ children, style, glow }) {
  return (
    <div style={{
      background: C.card, borderRadius: 18,
      border: `1px solid ${C.border}`,
      padding: "22px 26px",
      boxShadow: glow ? `0 0 32px ${glow}` : "0 4px 28px rgba(0,0,0,0.5)",
      ...style
    }}>
      {children}
    </div>
  );
}

function SectionLabel({ children, color }) {
  return (
    <div style={{ display:"flex", alignItems:"center", gap:8, marginBottom:14 }}>
      <div style={{ width:3, height:16, borderRadius:2, background: color || C.focus }} />
      <span style={{ fontFamily:FONTD, fontSize:13, fontWeight:700, color:C.text, letterSpacing:"0.02em" }}>
        {children}
      </span>
    </div>
  );
}

function StatTile({ icon, value, unit, label, color }) {
  return (
    <div style={{
      flex:"1 1 110px", minWidth:100, padding:"14px 16px",
      background:C.panel, borderRadius:12, border:`1px solid ${C.border}`,
      display:"flex", flexDirection:"column", gap:5
    }}>
      <span style={{ fontSize:18 }}>{icon}</span>
      <div style={{ fontFamily:FONT, fontSize:24, fontWeight:700, color:color||C.text, lineHeight:1 }}>
        {value ?? "—"}
        {unit && <span style={{ fontSize:12, color:C.dim, marginLeft:2 }}>{unit}</span>}
      </div>
      <div style={{ fontFamily:FONTD, fontSize:10, color:C.dim, textTransform:"uppercase", letterSpacing:"0.08em" }}>
        {label}
      </div>
    </div>
  );
}

function ChartTip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background:"#0C1525", border:`1px solid ${C.borderHi}`,
      borderRadius:10, padding:"10px 14px", fontFamily:FONT, fontSize:11, color:C.text
    }}>
      <div style={{ color:C.dim, marginBottom:4, fontSize:10 }}>{label}</div>
      {payload.map((p,i) => (
        <div key={i} style={{ color:p.color||C.text }}>
          {p.name}: <b>{typeof p.value==="number" ? p.value.toFixed(1) : p.value}</b>
        </div>
      ))}
    </div>
  );
}

// ─── EMPTY STATE ─────────────────────────────────────────────────────────────
function EmptyState({ module, icon, color }) {
  return (
    <div style={{
      display:"flex", flexDirection:"column", alignItems:"center",
      justifyContent:"center", gap:14, padding:"44px 20px",
      border:`1px dashed ${color}33`, borderRadius:16,
      background:`${color}06`
    }}>
      <div style={{
        width:56, height:56, borderRadius:16, fontSize:28,
        background:`${color}14`, border:`1px solid ${color}33`,
        display:"flex", alignItems:"center", justifyContent:"center"
      }}>{icon}</div>
      <div style={{ textAlign:"center" }}>
        <div style={{ fontFamily:FONTD, fontSize:15, fontWeight:700, color, marginBottom:6 }}>
          No {module} data yet
        </div>
        <div style={{ fontFamily:FONT, fontSize:11, color:C.dim, lineHeight:1.7, maxWidth:260 }}>
          Data will appear here automatically once you run a session in the <b style={{color}}>{module} Module</b>.
          No save button needed — it stores instantly.
        </div>
      </div>
    </div>
  );
}

// ─── BACKEND OFFLINE BANNER ───────────────────────────────────────────────────
function OfflineBanner() {
  return (
    <div style={{
      padding:"14px 22px", borderRadius:12, marginBottom:20,
      background:"rgba(248,113,113,0.08)", border:`1px solid ${C.seizure}44`,
      display:"flex", alignItems:"center", gap:12, fontFamily:FONT, fontSize:12
    }}>
      <span style={{ fontSize:20 }}>🔌</span>
      <div>
        <div style={{ color:C.seizure, fontWeight:700, marginBottom:2 }}>Backend not connected</div>
        <div style={{ color:C.dim }}>
          Start the Flask server: <code style={{ color:C.warn, background:"rgba(0,0,0,0.3)", padding:"1px 6px", borderRadius:4 }}>
            python api.py
          </code> — then refresh this page.
        </div>
      </div>
    </div>
  );
}

// ─── FOCUS PANEL ─────────────────────────────────────────────────────────────
function FocusPanel({ data }) {
  if (!data) return null;
  const { stats: s, timeline, label_distribution, has_data } = data;
  const labelColor = { High:C.high, Medium:C.medium, Low:C.low };

  return (
    <Card glow={has_data ? C.focusGlow : undefined}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:18 }}>
        <div style={{
          width:38, height:38, borderRadius:11,
          background:"linear-gradient(135deg,#0369A1,#38BDF8)",
          display:"flex", alignItems:"center", justifyContent:"center", fontSize:20
        }}>🧠</div>
        <div>
          <div style={{ fontFamily:FONTD, fontSize:15, fontWeight:700, color:C.focus }}>Focus Module</div>
          <div style={{ fontFamily:FONT, fontSize:10, color:C.dim }}>Attention & Concentration</div>
        </div>
      </div>

      {!has_data ? (
        <EmptyState module="Focus" icon="🧠" color={C.focus} />
      ) : (
        <>
          <div style={{ display:"flex", gap:10, flexWrap:"wrap", marginBottom:18 }}>
            <StatTile icon="📊" label="Sessions"   value={s.total_sessions} color={C.focus} />
            <StatTile icon="🎯" label="Avg Score"  value={s.avg_score_pct}  unit="%" color={C.focus} />
            <StatTile icon="⚡" label="Peak Score" value={s.peak_score_pct} unit="%" color={C.high} />
            <StatTile icon="⏱️" label="Total Time" value={s.total_minutes}  unit="min" color={C.dim} />
          </div>

          <SectionLabel color={C.focus}>Score Timeline</SectionLabel>
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={timeline} margin={{ top:4, right:4, left:-22, bottom:0 }}>
              <defs>
                <linearGradient id="fgGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.focus} stopOpacity={0.28} />
                  <stop offset="95%" stopColor={C.focus} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(56,189,248,0.05)" vertical={false} />
              <XAxis dataKey="timestamp" tick={{ fill:C.faint, fontSize:8, fontFamily:FONT }}
                tickLine={false} axisLine={false} interval="preserveStartEnd"
                tickFormatter={v => v?.slice(5,16) || v} />
              <YAxis domain={[0,100]} tick={{ fill:C.faint, fontSize:9, fontFamily:FONT }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTip />} />
              <Area type="monotone" dataKey="score_pct" name="Score %"
                stroke={C.focus} fill="url(#fgGrad)" strokeWidth={2}
                dot={false} activeDot={{ r:4, fill:C.focus }} />
            </AreaChart>
          </ResponsiveContainer>

          {label_distribution?.length > 0 && (
            <>
              <SectionLabel color={C.focus} style={{ marginTop:18 }}>Label Breakdown</SectionLabel>
              <div style={{ display:"flex", gap:10, flexWrap:"wrap" }}>
                {label_distribution.map((r,i) => (
                  <div key={i} style={{
                    flex:"1 1 80px", padding:"10px 14px", borderRadius:10,
                    background:`${labelColor[r.focus_label] || C.dim}12`,
                    border:`1px solid ${labelColor[r.focus_label] || C.dim}33`,
                    textAlign:"center"
                  }}>
                    <div style={{ fontFamily:FONT, fontSize:20, fontWeight:700, color:labelColor[r.focus_label]||C.dim }}>
                      {r.count}
                    </div>
                    <div style={{ fontFamily:FONTD, fontSize:10, color:C.dim, marginTop:3 }}>
                      {r.focus_label}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </Card>
  );
}

// ─── EMOTION PANEL ────────────────────────────────────────────────────────────
function EmotionPanel({ data }) {
  if (!data) return null;
  const { stats: s, distribution, timeline, has_data } = data;

  return (
    <Card glow={has_data ? "rgba(167,139,250,0.15)" : undefined}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:18 }}>
        <div style={{
          width:38, height:38, borderRadius:11,
          background:"linear-gradient(135deg,#6D28D9,#A78BFA)",
          display:"flex", alignItems:"center", justifyContent:"center", fontSize:20
        }}>💜</div>
        <div>
          <div style={{ fontFamily:FONTD, fontSize:15, fontWeight:700, color:C.emotion }}>Emotion Module</div>
          <div style={{ fontFamily:FONT, fontSize:10, color:C.dim }}>Affective State Recognition</div>
        </div>
      </div>

      {!has_data ? (
        <EmptyState module="Emotion" icon="💜" color={C.emotion} />
      ) : (
        <>
          <div style={{ display:"flex", gap:10, flexWrap:"wrap", marginBottom:18 }}>
            <StatTile icon="💡" label="Sessions"   value={s.total_sessions} color={C.emotion} />
            <StatTile icon="😊" label="Top Emotion" value={distribution?.[0]?.dominant_emotion} color={EMOTION_COLORS[distribution?.[0]?.dominant_emotion]||C.emotion} />
            <StatTile icon="⏱️" label="Total Time" value={s.total_minutes} unit="min" color={C.dim} />
          </div>

          <SectionLabel color={C.emotion}>Emotion Frequency</SectionLabel>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={distribution} margin={{ top:4, right:4, left:-22, bottom:20 }}>
              <CartesianGrid stroke="rgba(167,139,250,0.05)" vertical={false} />
              <XAxis dataKey="dominant_emotion"
                tick={{ fill:C.faint, fontSize:9, fontFamily:FONT }}
                tickLine={false} axisLine={false} angle={-20} textAnchor="end" />
              <YAxis tick={{ fill:C.faint, fontSize:9, fontFamily:FONT }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTip />} />
              <Bar dataKey="count" name="Count" radius={[4,4,0,0]}>
                {distribution.map((r,i) => (
                  <Cell key={i} fill={EMOTION_COLORS[r.dominant_emotion]||C.emotion} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <SectionLabel color={C.emotion} style={{ marginTop:18 }}>Recent Sessions</SectionLabel>
          <div style={{ display:"flex", flexWrap:"wrap", gap:6 }}>
            {timeline.slice(-20).map((r,i) => {
              const col = EMOTION_COLORS[r.dominant_emotion] || C.dim;
              return (
                <div key={i} style={{
                  padding:"4px 10px", borderRadius:20,
                  background:`${col}14`, border:`1px solid ${col}33`,
                  fontFamily:FONT, fontSize:10, color:col
                }}>
                  {r.dominant_emotion}
                </div>
              );
            })}
          </div>
        </>
      )}
    </Card>
  );
}

// ─── SEIZURE PANEL ────────────────────────────────────────────────────────────
function SeizurePanel({ data }) {
  if (!data) return null;
  const { stats: s, timeline, has_data } = data;
  const safeRate = s.total_scans > 0
    ? +((s.total_clear / s.total_scans) * 100).toFixed(1) : null;
  const alerting = (s.total_detected || 0) > 0;

  return (
    <Card glow={alerting ? "rgba(248,113,113,0.15)" : undefined}>
      <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:18 }}>
        <div style={{
          width:38, height:38, borderRadius:11,
          background: alerting
            ? "linear-gradient(135deg,#991B1B,#F87171)"
            : "linear-gradient(135deg,#065F46,#34D399)",
          display:"flex", alignItems:"center", justifyContent:"center", fontSize:20
        }}>{alerting ? "⚠️" : "🛡️"}</div>
        <div>
          <div style={{ fontFamily:FONTD, fontSize:15, fontWeight:700, color:alerting?C.seizure:C.safe }}>
            Seizure Module
          </div>
          <div style={{ fontFamily:FONT, fontSize:10, color:C.dim }}>Neural Activity Monitoring</div>
        </div>
      </div>

      {!has_data ? (
        <EmptyState module="Seizure" icon="🔬" color={C.seizure} />
      ) : (
        <>
          <div style={{ display:"flex", gap:10, flexWrap:"wrap", marginBottom:18 }}>
            <StatTile icon="🔬" label="Scans"      value={s.total_scans}    color={C.text} />
            <StatTile icon="✅" label="Clear"       value={s.total_clear}    color={C.safe} />
            <StatTile icon="🚨" label="Detected"   value={s.total_detected} color={alerting?C.seizure:C.dim} />
            {safeRate !== null && (
              <StatTile icon="🛡️" label="Safe Rate" value={safeRate} unit="%" color={C.safe} />
            )}
          </div>

          {/* Safe vs Detected donut */}
          <div style={{ display:"flex", alignItems:"center", gap:20, marginBottom:18 }}>
            <ResponsiveContainer width={110} height={110}>
              <PieChart>
                <Pie
                  data={[
                    { name:"Clear",    value: s.total_clear    || 0 },
                    { name:"Detected", value: s.total_detected || 0 },
                  ]}
                  cx="50%" cy="50%" innerRadius={32} outerRadius={48}
                  paddingAngle={3} dataKey="value" startAngle={90} endAngle={-270}
                >
                  <Cell fill={C.safe}    stroke="none" />
                  <Cell fill={C.seizure} stroke="none" />
                </Pie>
                <Tooltip content={<ChartTip />} />
              </PieChart>
            </ResponsiveContainer>
            <div>
              <div style={{ fontFamily:FONT, fontSize:32, fontWeight:700, color:C.safe, lineHeight:1 }}>
                {safeRate}<span style={{ fontSize:14, color:C.dim }}>%</span>
              </div>
              <div style={{ fontFamily:FONTD, fontSize:11, color:C.dim, marginTop:4 }}>Safe Scan Rate</div>
              {s.avg_conf_pct != null && alerting && (
                <div style={{ fontFamily:FONT, fontSize:11, color:C.seizure, marginTop:6 }}>
                  Avg detect conf: {s.avg_conf_pct}%
                </div>
              )}
            </div>
          </div>

          {alerting && timeline.filter(r=>r.detected===1).length > 0 && (
            <>
              <SectionLabel color={C.seizure}>Detection Events</SectionLabel>
              <ResponsiveContainer width="100%" height={100}>
                <BarChart
                  data={timeline.filter(r=>r.detected===1).map(r=>({
                    ts: r.timestamp?.slice(5,16)||r.timestamp,
                    conf: +(r.confidence*100).toFixed(1),
                    type: r.seizure_type
                  }))}
                  margin={{ top:4, right:4, left:-22, bottom:0 }}
                >
                  <CartesianGrid stroke="rgba(248,113,113,0.05)" vertical={false} />
                  <XAxis dataKey="ts" tick={{ fill:C.faint, fontSize:8, fontFamily:FONT }} tickLine={false} axisLine={false} />
                  <YAxis domain={[0,100]} tick={{ fill:C.faint, fontSize:9, fontFamily:FONT }} tickLine={false} axisLine={false} />
                  <Tooltip content={<ChartTip />} />
                  <Bar dataKey="conf" name="Confidence %" fill={C.seizure} radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </>
          )}

          {!alerting && (
            <div style={{
              textAlign:"center", padding:"14px",
              background:`${C.safe}0C`, borderRadius:10,
              border:`1px solid ${C.safe}22`,
              fontFamily:FONTD, fontSize:13, color:C.safe
            }}>
              ✨ No seizure activity detected in this period
            </div>
          )}
        </>
      )}
    </Card>
  );
}

// ─── OVERVIEW CARDS ───────────────────────────────────────────────────────────
function OverviewRow({ data }) {
  if (!data) return null;
  const f = data.focus?.stats  || {};
  const e = data.emotion?.stats || {};
  const s = data.seizure?.stats || {};
  const totalSessions = (f.total_sessions||0) + (e.total_sessions||0) + (s.total_scans||0);
  const anyData = totalSessions > 0;

  return (
    <div style={{
      display:"grid",
      gridTemplateColumns:"repeat(auto-fit,minmax(140px,1fr))",
      gap:12, marginBottom:22
    }}>
      {[
        { icon:"📈", label:"Total Sessions",  value: anyData ? totalSessions : "—",    color:C.text },
        { icon:"🧠", label:"Focus Sessions",  value: f.total_sessions || "—",          color:C.focus },
        { icon:"💜", label:"Emotion Sessions",value: e.total_sessions || "—",          color:C.emotion },
        { icon:"🔬", label:"Seizure Scans",   value: s.total_scans    || "—",          color:C.seizure },
        { icon:"🎯", label:"Avg Focus Score", value: f.avg_score_pct ? `${f.avg_score_pct}%` : "—", color:C.focus },
        { icon:"🛡️", label:"Seizure Free",   value: s.total_detected!=null ? (s.total_detected===0?"Yes":"No") : "—", color:s.total_detected===0?C.safe:C.seizure },
      ].map((item,i) => (
        <div key={i} style={{
          background:C.card, borderRadius:14, border:`1px solid ${C.border}`,
          padding:"16px 18px", display:"flex", flexDirection:"column", gap:5
        }}>
          <span style={{ fontSize:19 }}>{item.icon}</span>
          <span style={{ fontFamily:FONT, fontSize:22, fontWeight:700, color:item.color, lineHeight:1 }}>
            {item.value}
          </span>
          <span style={{ fontFamily:FONTD, fontSize:10, color:C.dim, textTransform:"uppercase", letterSpacing:"0.07em" }}>
            {item.label}
          </span>
        </div>
      ))}
    </div>
  );
}

// ─── FILTER BAR ───────────────────────────────────────────────────────────────
function FilterBar({ period, dateFrom, dateTo, onPeriod, onDateFrom, onDateTo, onApply, loading }) {
  const tabs = [
    { key:"daily",   label:"Today"  },
    { key:"weekly",  label:"7 Days" },
    { key:"monthly", label:"30 Days"},
    { key:"custom",  label:"Custom" },
  ];
  return (
    <div style={{
      display:"flex", alignItems:"center", gap:12, flexWrap:"wrap",
      background:C.card, border:`1px solid ${C.border}`,
      borderRadius:14, padding:"12px 18px", marginBottom:22
    }}>
      <div style={{ display:"flex", gap:4, background:C.panel, borderRadius:10, padding:4 }}>
        {tabs.map(t => (
          <button key={t.key} onClick={() => onPeriod(t.key)} style={{
            padding:"7px 18px", borderRadius:8, border:"none", cursor:"pointer",
            fontFamily:FONTD, fontSize:13, fontWeight:600,
            background: period===t.key ? "linear-gradient(135deg,#0369A1,#38BDF8)" : "transparent",
            color: period===t.key ? "#fff" : C.dim,
            transition:"all 0.2s"
          }}>{t.label}</button>
        ))}
      </div>

      {period === "custom" && (
        <div style={{ display:"flex", alignItems:"center", gap:8 }}>
          <input type="date" value={dateFrom} onChange={e=>onDateFrom(e.target.value)} style={{
            background:C.panel, border:`1px solid ${C.border}`, color:C.text,
            borderRadius:8, padding:"6px 10px", fontFamily:FONT, fontSize:12, outline:"none"
          }}/>
          <span style={{ color:C.dim, fontFamily:FONT, fontSize:12 }}>→</span>
          <input type="date" value={dateTo} onChange={e=>onDateTo(e.target.value)} style={{
            background:C.panel, border:`1px solid ${C.border}`, color:C.text,
            borderRadius:8, padding:"6px 10px", fontFamily:FONT, fontSize:12, outline:"none"
          }}/>
          <button onClick={onApply} style={{
            padding:"7px 16px", borderRadius:8, border:"none", cursor:"pointer",
            background:"linear-gradient(135deg,#6D28D9,#A78BFA)",
            color:"#fff", fontFamily:FONTD, fontSize:13, fontWeight:600
          }}>Apply</button>
        </div>
      )}

      <div style={{ marginLeft:"auto", display:"flex", alignItems:"center", gap:8 }}>
        {loading && (
          <div style={{
            width:14, height:14, borderRadius:"50%",
            border:`2px solid ${C.focus}`, borderTopColor:"transparent",
            animation:"spin 0.8s linear infinite"
          }}/>
        )}
        <span style={{ fontFamily:FONT, fontSize:10, color:C.dim }}>
          {loading ? "Fetching…" : "Live data"}
        </span>
      </div>
    </div>
  );
}

// ─── DB VIEWER ────────────────────────────────────────────────────────────────
const TABLES = [
  { key:"focus_sessions",   label:"Focus Sessions",   color:C.focus   },
  { key:"emotion_sessions", label:"Emotion Sessions", color:C.emotion },
  { key:"seizure_sessions", label:"Seizure Sessions", color:C.seizure },
];

function DBViewer() {
  const [activeTable, setActiveTable] = useState("focus_sessions");
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deleting, setDeleting] = useState(null);
  const limit = 25;

  const load = useCallback(async (table, pg) => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiFetch(`/db/rows/${table}?limit=${limit}&offset=${pg*limit}`);
      setRows(data.rows || []);
      setTotal(data.total || 0);
    } catch(e) {
      setError("Backend not reachable. Start Flask server first.");
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { setPage(0); load(activeTable, 0); }, [activeTable]);
  useEffect(() => { load(activeTable, page); }, [page]);

  const handleDelete = async (id) => {
    if (!window.confirm(`Delete row #${id}?`)) return;
    setDeleting(id);
    try {
      await apiDelete(`/db/rows/${activeTable}/${id}`);
      load(activeTable, page);
    } catch(e) {
      alert("Delete failed: " + e.message);
    } finally {
      setDeleting(null);
    }
  };

  const tbl = TABLES.find(t => t.key === activeTable);
  const totalPages = Math.ceil(total / limit);
  const cols = rows[0] ? Object.keys(rows[0]) : [];

  return (
    <div>
      {/* Table Selector */}
      <div style={{ display:"flex", gap:8, marginBottom:20 }}>
        {TABLES.map(t => (
          <button key={t.key} onClick={() => setActiveTable(t.key)} style={{
            padding:"9px 20px", borderRadius:10, border:`1px solid ${activeTable===t.key ? t.color : C.border}`,
            cursor:"pointer", fontFamily:FONTD, fontSize:13, fontWeight:600,
            background: activeTable===t.key ? `${t.color}18` : C.panel,
            color: activeTable===t.key ? t.color : C.dim,
            transition:"all 0.2s"
          }}>{t.label}</button>
        ))}
        <div style={{ marginLeft:"auto", display:"flex", alignItems:"center", gap:6 }}>
          <span style={{ fontFamily:FONT, fontSize:11, color:C.dim }}>
            {total} total rows
          </span>
          <button onClick={() => load(activeTable, page)} style={{
            padding:"7px 14px", borderRadius:8, border:`1px solid ${C.border}`,
            background:C.panel, color:C.dim, fontFamily:FONT, fontSize:11, cursor:"pointer"
          }}>↻ Refresh</button>
        </div>
      </div>

      {error && (
        <div style={{
          padding:"14px 18px", borderRadius:10, marginBottom:16,
          background:"rgba(248,113,113,0.08)", border:`1px solid ${C.seizure}44`,
          fontFamily:FONT, fontSize:12, color:C.seizure
        }}>🔌 {error}</div>
      )}

      {loading ? (
        <div style={{ textAlign:"center", padding:"40px", color:C.dim, fontFamily:FONT, fontSize:13 }}>
          Loading…
        </div>
      ) : rows.length === 0 ? (
        <div style={{
          textAlign:"center", padding:"48px",
          border:`1px dashed ${tbl?.color}33`, borderRadius:16,
          background:`${tbl?.color}06`
        }}>
          <div style={{ fontSize:36, marginBottom:12 }}>📭</div>
          <div style={{ fontFamily:FONTD, fontSize:15, fontWeight:700, color:tbl?.color, marginBottom:6 }}>
            No records in {tbl?.label}
          </div>
          <div style={{ fontFamily:FONT, fontSize:11, color:C.dim }}>
            Records appear here automatically after running EEG module predictions.
          </div>
        </div>
      ) : (
        <>
          {/* Table */}
          <div style={{ overflowX:"auto", borderRadius:12, border:`1px solid ${C.border}` }}>
            <table style={{ width:"100%", borderCollapse:"collapse" }}>
              <thead>
                <tr style={{ background:C.panel }}>
                  {cols.map(col => (
                    <th key={col} style={{
                      padding:"10px 14px", fontFamily:FONT, fontSize:10,
                      color:C.dim, textAlign:"left", borderBottom:`1px solid ${C.border}`,
                      textTransform:"uppercase", letterSpacing:"0.07em", whiteSpace:"nowrap"
                    }}>{col}</th>
                  ))}
                  <th style={{
                    padding:"10px 14px", fontFamily:FONT, fontSize:10,
                    color:C.dim, textAlign:"center", borderBottom:`1px solid ${C.border}`,
                    textTransform:"uppercase", letterSpacing:"0.07em"
                  }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr key={row.id} style={{
                    background: ri%2===0 ? C.card : `${C.panel}80`,
                    transition:"background 0.15s"
                  }}>
                    {cols.map(col => {
                      let val = row[col];
                      let color = C.text;
                      // Color-code specific columns
                      if (col==="detected") { val = val===1?"🔴 Yes":"🟢 No"; color=val.includes("Yes")?C.seizure:C.safe; }
                      if (col==="focus_label") color = {High:C.high,Medium:C.medium,Low:C.low}[val]||C.text;
                      if (col==="dominant_emotion") color = EMOTION_COLORS[val]||C.text;
                      if (col==="id") color = C.dim;
                      if (col==="emotion_scores") {
                        try { val = Object.entries(JSON.parse(val)).map(([k,v])=>`${k}:${(+v*100).toFixed(0)}%`).join(" · "); } catch{}
                      }
                      return (
                        <td key={col} style={{
                          padding:"9px 14px", fontFamily:FONT, fontSize:11,
                          color, borderBottom:`1px solid ${C.border}22`,
                          maxWidth:200, overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap"
                        }} title={String(row[col])}>
                          {String(val ?? "")}
                        </td>
                      );
                    })}
                    <td style={{ padding:"9px 14px", textAlign:"center", borderBottom:`1px solid ${C.border}22` }}>
                      <button
                        onClick={() => handleDelete(row.id)}
                        disabled={deleting===row.id}
                        style={{
                          padding:"4px 10px", borderRadius:6,
                          border:`1px solid ${C.seizure}44`,
                          background:`${C.seizure}0E`, color:C.seizure,
                          fontFamily:FONT, fontSize:10, cursor:"pointer",
                          opacity: deleting===row.id ? 0.5 : 1
                        }}
                      >
                        {deleting===row.id ? "…" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {totalPages > 1 && (
            <div style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:10, marginTop:14 }}>
              <button onClick={() => setPage(p => Math.max(0,p-1))} disabled={page===0} style={{
                padding:"6px 16px", borderRadius:8, border:`1px solid ${C.border}`,
                background:C.panel, color:page===0?C.faint:C.text,
                fontFamily:FONT, fontSize:12, cursor:page===0?"default":"pointer"
              }}>← Prev</button>
              <span style={{ fontFamily:FONT, fontSize:11, color:C.dim }}>
                Page {page+1} of {totalPages}
              </span>
              <button onClick={() => setPage(p => Math.min(totalPages-1,p+1))} disabled={page===totalPages-1} style={{
                padding:"6px 16px", borderRadius:8, border:`1px solid ${C.border}`,
                background:C.panel, color:page===totalPages-1?C.faint:C.text,
                fontFamily:FONT, fontSize:12, cursor:page===totalPages-1?"default":"pointer"
              }}>Next →</button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── JOURNAL TAB ─────────────────────────────────────────────────────────────
function JournalTab() {
  const [period, setPeriod]   = useState("daily");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo]   = useState("");
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const load = useCallback(async (p, df, dt) => {
    setLoading(true);
    setError(null);
    try {
      let url = `/journal/combined?period=${p}`;
      if (p === "custom" && df && dt) url += `&date_from=${df}&date_to=${dt}`;
      const d = await apiFetch(url);
      setData(d);
    } catch(e) {
      setError("Cannot reach backend. Make sure Flask is running on port 5000.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(period, dateFrom, dateTo); }, [period]);

  // Auto-refresh every 30s
  useEffect(() => {
    const id = setInterval(() => load(period, dateFrom, dateTo), 30000);
    return () => clearInterval(id);
  }, [period, dateFrom, dateTo, load]);

  return (
    <>
      {error && <OfflineBanner />}

      <FilterBar
        period={period} dateFrom={dateFrom} dateTo={dateTo}
        onPeriod={setPeriod}
        onDateFrom={setDateFrom}
        onDateTo={setDateTo}
        onApply={() => load("custom", dateFrom, dateTo)}
        loading={loading}
      />

      {data && <OverviewRow data={data} />}

      <div style={{
        display:"grid",
        gridTemplateColumns:"repeat(auto-fit,minmax(340px,1fr))",
        gap:20,
        animation:"fadeUp 0.35s ease"
      }}>
        <FocusPanel   data={data?.focus} />
        <EmotionPanel data={data?.emotion} />
        <SeizurePanel data={data?.seizure} />
      </div>

      {/* How it works note */}
      {data && !data.focus?.has_data && !data.emotion?.has_data && !data.seizure?.has_data && (
        <div style={{
          marginTop:24, padding:"20px 24px", borderRadius:14,
          background:`${C.warn}08`, border:`1px solid ${C.warn}22`
        }}>
          <div style={{ fontFamily:FONTD, fontSize:14, fontWeight:700, color:C.warn, marginBottom:8 }}>
            ℹ️ How data appears here
          </div>
          <div style={{ fontFamily:FONT, fontSize:12, color:C.dim, lineHeight:1.9 }}>
            1. Run a session in the <span style={{color:C.focus}}>Focus Module</span>, <span style={{color:C.emotion}}>Emotion Module</span>, or <span style={{color:C.seizure}}>Seizure Module</span>.<br/>
            2. When each module finishes its prediction, it <b>automatically</b> calls the API and inserts a row.<br/>
            3. Come back here — the journal updates in real time. No save button ever needed.
          </div>
        </div>
      )}
    </>
  );
}

// ─── ROOT APP ─────────────────────────────────────────────────────────────────
export default function BrainJournal() {
  const [tab, setTab] = useState("journal");

  return (
    <div style={{ minHeight:"100vh", background:C.bg, color:C.text, fontFamily:FONTD }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
        * { box-sizing:border-box; }
        body { margin:0; }
        @keyframes spin    { to { transform:rotate(360deg); } }
        @keyframes fadeUp  { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
        button:hover { opacity:0.88; }
        ::-webkit-scrollbar { width:6px; height:6px; }
        ::-webkit-scrollbar-track { background:#04080F; }
        ::-webkit-scrollbar-thumb { background:#1A3050; border-radius:3px; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom:`1px solid ${C.border}`,
        background:"linear-gradient(180deg,#070D1A 0%,#04080F 100%)",
        padding:"0 32px",
      }}>
        <div style={{
          maxWidth:1300, margin:"0 auto", padding:"18px 0",
          display:"flex", alignItems:"center", justifyContent:"space-between", flexWrap:"wrap", gap:12
        }}>
          {/* Logo */}
          <div style={{ display:"flex", alignItems:"center", gap:12 }}>
            <div style={{
              width:42, height:42, borderRadius:13,
              background:"linear-gradient(135deg,#0C4A6E,#0EA5E9)",
              display:"flex", alignItems:"center", justifyContent:"center",
              fontSize:21, boxShadow:`0 0 18px ${C.focusGlow}`
            }}>🧬</div>
            <div>
              <div style={{ fontFamily:FONTD, fontSize:19, fontWeight:700, color:C.text, letterSpacing:"-0.01em" }}>
                Daily Brain Journal
              </div>
              <div style={{ fontFamily:FONT, fontSize:10, color:C.dim, marginTop:1 }}>
                EEG Neural Activity Tracker · Auto-sync · No save button
              </div>
            </div>
          </div>

          {/* Nav Tabs */}
          <div style={{ display:"flex", gap:6, background:C.panel, borderRadius:12, padding:5 }}>
            {[
              { key:"journal", icon:"📓", label:"Brain Journal" },
              { key:"db",      icon:"🗄️",  label:"Database Viewer" },
            ].map(t => (
              <button key={t.key} onClick={() => setTab(t.key)} style={{
                padding:"8px 20px", borderRadius:9, border:"none", cursor:"pointer",
                fontFamily:FONTD, fontSize:13, fontWeight:600,
                background: tab===t.key ? "linear-gradient(135deg,#0369A1,#0EA5E9)" : "transparent",
                color: tab===t.key ? "#fff" : C.dim,
                display:"flex", alignItems:"center", gap:6,
                transition:"all 0.2s"
              }}>
                <span>{t.icon}</span> {t.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Page Content */}
      <div style={{ maxWidth:1300, margin:"0 auto", padding:"26px 32px" }}>
        {tab === "journal" ? <JournalTab /> : <DBViewer />}
      </div>
    </div>
  );
}