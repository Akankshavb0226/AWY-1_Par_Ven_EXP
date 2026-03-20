"""
AWY-1  ·  Interactive Workspace Clearance Controller
=====================================================
Run with:  python -m streamlit run streamlit_app.py

How it works
------------
• Drag the slider to move the object anywhere in the corridor
• Every slider movement adds sensor readings to a rolling buffer
• Once 300 readings are collected the REAL trained RF model predicts:
    – Motion class  (Moving Away / Stationary / Approaching)
    – Time-to-Clear (seconds until zone ≥ 0.90 m)
• The AGV reacts to the ML prediction automatically
• Use Quick Scenario buttons to instantly load a preset trajectory
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import joblib
import streamlit as st
import plotly.graph_objects as go

from simulation import (
    MEAS_RATE, TIME_PER_READING,
    LABEL_AWAY, LABEL_APPROACHING, LABEL_STATIONARY,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS  (must match awy1_project.py training exactly)
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE    = 300
ZONE_THRESHOLD = 0.90        # metres  — matches real model training
READINGS_PER_STEP = 30       # readings added per slider interaction
MAX_BUFFER     = 700         # rolling buffer cap

LABEL_NAMES  = {LABEL_APPROACHING: 'APPROACHING',
                LABEL_AWAY:        'MOVING AWAY',
                LABEL_STATIONARY:  'STATIONARY'}
LABEL_COLORS = {LABEL_APPROACHING: '#e74c3c',
                LABEL_AWAY:        '#2ecc71',
                LABEL_STATIONARY:  '#3498db'}

# Corridor layout
CX_MIN, CX_MAX   = -0.35, 3.20
CY_MIN, CY_MAX   =  0.00, 2.80
CY_MID           =  1.40
WALL_BOT, WALL_TOP = 0.18, 2.62

# AGV
AGV_W, AGV_H  = 0.20, 0.30
AGV_WAIT_X    = -0.26
AGV_READY_X   = -0.12
AGV_MOVE_X    =  0.80    # where AGV sits when moving through

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'results', 'models')


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION — 16 features, identical to awy1_project.py
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(window: np.ndarray) -> np.ndarray:
    w  = window.astype(float)
    t  = np.arange(len(w), dtype=float)
    sl, ic = np.polyfit(t, w, 1)
    vel    = sl * MEAS_RATE
    h      = len(w) // 2
    v1     = float(np.polyfit(t[:h], w[:h], 1)[0])
    v2     = float(np.polyfit(t[h:], w[h:], 1)[0])
    accel  = (v2 - v1) / (h / MEAS_RATE)
    tn     = float(np.std(w - (sl * t + ic)))
    diffs  = np.diff(w)
    mean_d = float(np.mean(w))
    mn, mx = float(np.min(w)), float(np.max(w))
    return np.array([
        mean_d, float(np.std(w)), mn, mx, mx - mn,
        vel, accel, tn,
        float(w[0]), float(w[-1]), float(w[h]),
        float(w[-1] - w[0]),
        vel / max(mean_d, 0.1),
        float((diffs > 0).mean()),
        float((diffs < 0).mean()),
        1.0 if mean_d < ZONE_THRESHOLD else 0.0,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# LOAD REAL TRAINED MODEL (cached once per session)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=" Loading real RF model trained on FIUS data…")
def load_models():
    clf_path = os.path.join(MODEL_DIR, 'rf_classifier.joblib')
    reg_path = os.path.join(MODEL_DIR, 'rf_regressor.joblib')
    if not os.path.exists(clf_path):
        st.error("❌ Run `python awy1_project.py` first to train and save the model.")
        st.stop()
    clf = joblib.load(clf_path)
    reg = joblib.load(reg_path) if os.path.exists(reg_path) else None
    return clf, reg


# ─────────────────────────────────────────────────────────────────────────────
# CORRIDOR FIGURE
# ─────────────────────────────────────────────────────────────────────────────
def make_corridor(dist: float, label: int | None,
                  ttc_val, agv_x: float,
                  zone_clear: bool, ready: bool) -> go.Figure:

    obj_color = LABEL_COLORS.get(label, '#888888') if label is not None else '#888888'
    icons     = {LABEL_AWAY: '➡', LABEL_APPROACHING: '⬅', LABEL_STATIONARY: '■'}

    agv_fill  = ('rgba(39,174,96,0.85)'  if zone_clear else
                 'rgba(243,156,18,0.75)' if (label == LABEL_AWAY and ttc_val is not None and ttc_val < 4.0) else
                 'rgba(68,85,102,0.80)')

    fig = go.Figure()

    # Zone fills
    fig.add_shape(type='rect', x0=0, x1=ZONE_THRESHOLD,
                  y0=WALL_BOT, y1=WALL_TOP,
                  fillcolor='rgba(180,20,20,0.22)', line_width=0)
    fig.add_shape(type='rect', x0=ZONE_THRESHOLD, x1=CX_MAX,
                  y0=WALL_BOT, y1=WALL_TOP,
                  fillcolor='rgba(20,140,20,0.18)', line_width=0)

    # Walls
    for y0, y1 in [(0, WALL_BOT), (WALL_TOP, CY_MAX)]:
        fig.add_shape(type='rect', x0=CX_MIN, x1=CX_MAX, y0=y0, y1=y1,
                      fillcolor='#1a1f27', line_width=0)
    for yw in [WALL_BOT, WALL_TOP]:
        fig.add_shape(type='line', x0=CX_MIN, x1=CX_MAX, y0=yw, y1=yw,
                      line=dict(color='#364048', width=4))

    # Zone boundary
    fig.add_shape(type='line',
                  x0=ZONE_THRESHOLD, x1=ZONE_THRESHOLD,
                  y0=WALL_BOT, y1=WALL_TOP,
                  line=dict(color='#f39c12', width=2.5, dash='dash'))

    # Sensor beam
    spread = max(0.04, 0.22 * dist)
    alpha  = 0.20 if dist < ZONE_THRESHOLD else 0.07
    fig.add_trace(go.Scatter(
        x=[0, dist, dist, 0, 0],
        y=[CY_MID, CY_MID+spread, CY_MID-spread, CY_MID, CY_MID],
        fill='toself',
        fillcolor=f'rgba(241,196,15,{alpha})',
        line=dict(width=0.5, color='rgba(241,196,15,0.3)'),
        showlegend=False, hoverinfo='skip', mode='lines',
    ))

    # Dotted measurement line
    fig.add_shape(type='line', x0=0, x1=dist, y0=CY_MID, y1=CY_MID,
                  line=dict(color='rgba(241,196,15,0.5)', width=1.5, dash='dot'))

    # Distance ruler
    for xm in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        if xm <= CX_MAX:
            fig.add_shape(type='line', x0=xm, x1=xm,
                          y0=WALL_BOT, y1=WALL_BOT-0.06,
                          line=dict(color='#555f6b', width=1))
            fig.add_annotation(x=xm, y=WALL_BOT-0.25, text=f'{xm}m',
                               showarrow=False, font=dict(color='#6e7681', size=9))

    # AGV
    fig.add_shape(type='rect',
                  x0=agv_x, x1=agv_x+AGV_W,
                  y0=CY_MID-AGV_H/2, y1=CY_MID+AGV_H/2,
                  fillcolor=agv_fill,
                  line=dict(color='#8899aa', width=1.5))
    if zone_clear:
        agv_txt, agv_col = '🚗 MOVING ➡', '#68d391'
    elif label == LABEL_AWAY and ttc_val is not None and ttc_val < 4.0:
        agv_txt, agv_col = '🚗 READY ⚙', '#f6c90e'
    else:
        agv_txt, agv_col = '🚗 WAITING ⏸', '#8899aa'
    fig.add_annotation(x=agv_x+AGV_W/2, y=CY_MID+AGV_H/2+0.22,
                       text=agv_txt, showarrow=False,
                       font=dict(color=agv_col, size=10))

    # Object circle
    obj_x = float(np.clip(dist, 0.08, CX_MAX-0.1))
    obj_size = 30 if (label == LABEL_STATIONARY) else 26
    fig.add_trace(go.Scatter(
        x=[obj_x], y=[CY_MID],
        mode='markers',
        marker=dict(size=obj_size, color=obj_color,
                    line=dict(color='white', width=2.5)),
        showlegend=False, hoverinfo='skip',
    ))

    if ready and label is not None:
        fig.add_annotation(x=obj_x, y=CY_MID+0.52,
                           text=f'{icons[label]}  {LABEL_NAMES[label]}',
                           showarrow=False,
                           font=dict(color=obj_color, size=12, family='Arial Black'))
    else:
        fig.add_annotation(x=obj_x, y=CY_MID+0.52,
                           text='?  COLLECTING DATA…',
                           showarrow=False,
                           font=dict(color='#4a5568', size=11))

    fig.add_annotation(x=obj_x, y=CY_MID-0.54,
                       text=f'd = {dist:.3f} m',
                       showarrow=False, font=dict(color='#aaaaaa', size=11))

    # Zone status near boundary
    if zone_clear:
        fig.add_annotation(x=ZONE_THRESHOLD, y=WALL_BOT-0.38,
                           text='✓  ZONE CLEAR',
                           showarrow=False,
                           font=dict(color='#68d391', size=12, family='Arial Black'))
    elif ready and ttc_val is not None:
        fig.add_annotation(x=ZONE_THRESHOLD, y=WALL_BOT-0.38,
                           text=f'TtC ≈ {ttc_val:.1f} s',
                           showarrow=False,
                           font=dict(color='#f39c12', size=12, family='Arial Black'))

    # Zone labels
    fig.add_annotation(x=ZONE_THRESHOLD*0.5, y=WALL_TOP+0.18,
                       text=' ZONE OCCUPIED  (< 0.90 m)',
                       showarrow=False,
                       font=dict(color='#fc8181', size=11, family='Arial Black'))
    fig.add_annotation(x=ZONE_THRESHOLD+(CX_MAX-ZONE_THRESHOLD)*0.5, y=WALL_TOP+0.18,
                       text=' ZONE CLEAR  (≥ 0.90 m)',
                       showarrow=False,
                       font=dict(color='#68d391', size=11, family='Arial Black'))

    # Sensor
    fig.add_shape(type='rect', x0=-0.22, x1=0.0,
                  y0=CY_MID-0.20, y1=CY_MID+0.20,
                  fillcolor='#21262d', line=dict(color='#f1c40f', width=1.5))
    fig.add_annotation(x=-0.11, y=CY_MID, text='📡',
                       showarrow=False, font=dict(size=15))
    fig.add_annotation(x=-0.11, y=CY_MID-0.48, text='FIUS',
                       showarrow=False, font=dict(color='#f1c40f', size=8))

    fig.update_layout(
        plot_bgcolor='#161b22', paper_bgcolor='#0d1117',
        margin=dict(l=5, r=5, t=42, b=10),
        title=dict(
            text='WAREHOUSE CORRIDOR — TOP-DOWN VIEW'
                 '<span style="font-size:10px;color:#4a5568;"> '
                 ' Real RF · 500 trees · trained on FIUS data</span>',
            font=dict(color='white', size=13), x=0.5),
        xaxis=dict(range=[CX_MIN, CX_MAX], showgrid=False, zeroline=False,
                   tickfont=dict(color='#6e7681', size=9),
                   title=dict(text='Distance from sensor (metres)',
                              font=dict(color='#8b949e', size=11))),
        yaxis=dict(range=[CY_MIN-0.52, CY_MAX+0.25],
                   showgrid=False, zeroline=False, showticklabels=False),
        height=310,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SENSOR HISTORY CHART
# ─────────────────────────────────────────────────────────────────────────────
def make_history_chart(readings: list, label: int | None) -> go.Figure:
    dot_color = LABEL_COLORS.get(label, '#00d4ff') if label is not None else '#00d4ff'
    fig = go.Figure()
    fig.add_hline(y=ZONE_THRESHOLD,
                  line=dict(color='#f39c12', width=1.5, dash='dash'))
    fig.add_annotation(xref='paper', x=0.01, y=ZONE_THRESHOLD+0.08,
                       text=f'Zone boundary ({ZONE_THRESHOLD} m)',
                       showarrow=False,
                       font=dict(color='#f39c12', size=9), xanchor='left')
    if readings:
        t = [i * TIME_PER_READING for i in range(len(readings))]
        fig.add_trace(go.Scatter(
            x=t, y=readings,
            mode='lines',
            line=dict(color='#00d4ff', width=1.5),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[t[-1]], y=[readings[-1]],
            mode='markers',
            marker=dict(size=9, color=dot_color,
                        line=dict(color='white', width=2)),
            showlegend=False, hoverinfo='skip',
        ))
        # Highlight last WINDOW_SIZE readings (what model sees)
        if len(readings) >= WINDOW_SIZE:
            ws = len(readings) - WINDOW_SIZE
            fig.add_vrect(x0=t[ws], x1=t[-1],
                          fillcolor='rgba(255,255,255,0.04)',
                          line_width=0,
                          annotation_text='ML window',
                          annotation_font=dict(color='#4a5568', size=9),
                          annotation_position='top left')

    fig.update_layout(
        plot_bgcolor='#0d1117', paper_bgcolor='#0d1117',
        margin=dict(l=50, r=15, t=30, b=35),
        title=dict(text='SENSOR HISTORY  (your movements)',
                   font=dict(color='#718096', size=11), x=0.5),
        xaxis=dict(showgrid=True, gridcolor='#1e2733', zeroline=False,
                   tickfont=dict(color='#6e7681', size=9),
                   title=dict(text='Time (s)', font=dict(color='#8b949e', size=10))),
        yaxis=dict(range=[0, 3.2], showgrid=True, gridcolor='#1e2733',
                   zeroline=False, tickfont=dict(color='#6e7681', size=9),
                   title=dict(text='Distance (m)', font=dict(color='#8b949e', size=10))),
        height=180,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# METRIC CARDS
# ─────────────────────────────────────────────────────────────────────────────
def card_class(label: int) -> str:
    color = LABEL_COLORS[label]
    icons = {LABEL_AWAY: '➡', LABEL_APPROACHING: '⬅', LABEL_STATIONARY: '■'}
    return f"""
    <div style="background:#111827;border-radius:12px;padding:14px 10px;
                border:2px solid {color};text-align:center;margin-bottom:10px;">
      <div style="color:#718096;font-size:10px;font-weight:bold;
                  text-transform:uppercase;letter-spacing:1px;"> ML PREDICTION</div>
      <div style="color:{color};font-size:22px;font-weight:bold;
                  margin:8px 0;font-family:monospace;">
        {icons[label]}&nbsp;&nbsp;{LABEL_NAMES[label]}</div>
      <div style="color:#4a5568;font-size:9px;">Real RF · 500 trees · FIUS training data</div>
    </div>"""


def card_ttc(ttc_val, zone_clear: bool) -> str:
    if zone_clear:
        val, color, note = '0.0 s', '#2ecc71', '✓  Zone is CLEAR — AGV may proceed'
    elif ttc_val is not None:
        color = '#f39c12' if ttc_val > 2 else '#e67e22'
        val   = f'{ttc_val:.1f} s'
        note  = 'RF-predicted time until zone clears'
    else:
        val, color, note = '—', '#4a5568', 'N/A (object not Moving Away in zone)'
    return f"""
    <div style="background:#111827;border-radius:12px;padding:14px 10px;
                border:1px solid #2d3748;text-align:center;margin-bottom:10px;">
      <div style="color:#718096;font-size:10px;font-weight:bold;
                  text-transform:uppercase;letter-spacing:1px;">⏱ TIME-TO-CLEAR</div>
      <div style="color:{color};font-size:40px;font-weight:bold;
                  margin:8px 0;font-family:monospace;">{val}</div>
      <div style="color:#718096;font-size:10px;">{note}</div>
    </div>"""


def card_agv(label: int, ttc_val, zone_clear: bool) -> str:
    if zone_clear:
        bg, border, color = '#0a2e14', '#2ecc71', '#2ecc71'
        decision, note = '  RESUME OPERATIONS', 'Zone confirmed clear — AGV moving'
    elif label == LABEL_AWAY and ttc_val is not None and ttc_val < 4.0:
        bg, border, color = '#2d2800', '#f39c12', '#f6c90e'
        decision, note = f'⚙  PREPARE  (~{ttc_val:.0f}s)', 'ML early warning — pre-positioning'
    elif label == LABEL_AWAY:
        bg, border, color = '#1a1400', '#d4921c', '#f39c12'
        decision, note = '⏳  WAIT — CLEARANCE PREDICTED', 'Object moving away'
    elif label == LABEL_APPROACHING:
        bg, border, color = '#2e0000', '#e74c3c', '#fc8181'
        decision, note = '🚨  EMERGENCY HOLD', 'Object approaching — danger!'
    else:
        bg, border, color = '#00101a', '#3498db', '#63b3ed'
        decision, note = '⏸  HOLD POSITION', 'Object stationary — wait'
    return f"""
    <div style="background:{bg};border-radius:12px;padding:14px 10px;
                border:2px solid {border};text-align:center;margin-bottom:10px;">
      <div style="color:#718096;font-size:10px;font-weight:bold;
                  text-transform:uppercase;letter-spacing:1px;">🚗 AGV DECISION</div>
      <div style="color:{color};font-size:17px;font-weight:bold;
                  margin:8px 0;">{decision}</div>
      <div style="color:#a0aec0;font-size:10px;">{note}</div>
    </div>"""


def card_waiting() -> str:
    return """
    <div style="background:#111827;border-radius:12px;padding:14px 10px;
                border:1px solid #2d3748;text-align:center;margin-bottom:10px;">
      <div style="color:#4a5568;font-size:13px;font-weight:bold;margin:8px 0;">
        Move the object to build sensor history
      </div>
      <div style="color:#374151;font-size:10px;">ML predicts after 300 readings</div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────────
if 'readings' not in st.session_state:
    st.session_state.readings  = []
if 'last_dist' not in st.session_state:
    st.session_state.last_dist = 0.50


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AWY-1 · Interactive Clearance Controller',
    layout='wide',
)
st.markdown("""
<style>
  .stApp { background-color: #0d1117; }
  #MainMenu, footer { visibility: hidden; }
  section[data-testid="stSidebar"] { background-color: #111827; }
  section[data-testid="stSidebar"] label { color: #cbd5e0 !important; }
  h1,h2,h3 { color: white !important; }
  .stMarkdown p { color: #a0aec0; }
  hr { border-color: #2d3748; }
  div[data-testid="stSlider"] label { color: white !important; font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)


rng = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
clf, reg = load_models()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# AWY-1  ·  Interactive Workspace Clearance Controller")
st.markdown(
    "**Drag the slider** to move the object · "
    "The real RF model predicts motion class and Time-to-Clear in real-time"
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SLIDER  (big and prominent)
# ─────────────────────────────────────────────────────────────────────────────
dist = st.slider(
    " Object Distance from Sensor  —  drag me to move the object",
    min_value=0.05,
    max_value=3.00,
    value=float(st.session_state.last_dist),
    step=0.01,
    format="%.2f m",
)


# ─────────────────────────────────────────────────────────────────────────────
# ADD READINGS TO BUFFER on every slider interaction
# ─────────────────────────────────────────────────────────────────────────────
noise_std = 0.012
buf_rng   = np.random.default_rng(len(st.session_state.readings))
new_vals  = dist + buf_rng.normal(0, noise_std, READINGS_PER_STEP)
new_vals  = np.clip(new_vals, 0.05, 3.0).tolist()
st.session_state.readings.extend(new_vals)
st.session_state.readings  = st.session_state.readings[-MAX_BUFFER:]
st.session_state.last_dist = dist


# ─────────────────────────────────────────────────────────────────────────────
# ML INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
readings   = st.session_state.readings
buf_len    = len(readings)
ready      = buf_len >= WINDOW_SIZE
zone_clear = dist >= ZONE_THRESHOLD

label   = None
ttc_val = None

if ready:
    window = np.array(readings[-WINDOW_SIZE:])
    feats  = extract_features(window).reshape(1, -1)
    label  = int(clf.predict(feats)[0])
    if label == LABEL_AWAY and dist < ZONE_THRESHOLD and reg is not None:
        ttc_val = max(0.0, float(reg.predict(feats)[0]))

# AGV position
if zone_clear:
    agv_x = AGV_MOVE_X
elif ready and label == LABEL_AWAY and ttc_val is not None and ttc_val < 4.0:
    agv_x = AGV_READY_X
else:
    agv_x = AGV_WAIT_X


# ─────────────────────────────────────────────────────────────────────────────
# BUFFER PROGRESS BAR
# ─────────────────────────────────────────────────────────────────────────────
if not ready:
    pct = buf_len / WINDOW_SIZE
    st.progress(pct,
                text=f"  Building sensor history: {buf_len} / {WINDOW_SIZE} readings "
                     f"— keep moving the slider  ({int(pct*100)}%)")
else:
    st.success(f" ML Active — analysing last {WINDOW_SIZE} readings  "
               f"({buf_len} total collected)")


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT: corridor  |  metrics
# ─────────────────────────────────────────────────────────────────────────────
col_view, col_metrics = st.columns([2.2, 1], gap="medium")

with col_view:
    st.plotly_chart(
        make_corridor(dist, label, ttc_val, agv_x, zone_clear, ready),
        use_container_width=True,
        config={'displayModeBar': False},
    )
    st.plotly_chart(
        make_history_chart(readings, label),
        use_container_width=True,
        config={'displayModeBar': False},
    )

with col_metrics:
    st.markdown("### Live ML Dashboard")

    # Current distance card
    in_zone_str = 'IN ZONE' if dist < ZONE_THRESHOLD else '○ CLEAR'
    st.markdown(
        f"""<div style="background:#0d1117;border-radius:8px;padding:10px;
                       border:1px solid #1e2733;text-align:center;margin-bottom:10px;">
              <span style="color:#6e7681;font-size:10px;">SENSOR READING</span><br>
              <span style="color:#00d4ff;font-size:34px;font-weight:bold;
                           font-family:monospace;">{dist:.3f} m</span><br>
              <span style="color:{'#fc8181' if dist < ZONE_THRESHOLD else '#68d391'};
                           font-size:11px;">{in_zone_str}</span>
            </div>""",
        unsafe_allow_html=True,
    )

    if ready and label is not None:
        st.markdown(card_class(label),         unsafe_allow_html=True)
        st.markdown(card_ttc(ttc_val, zone_clear), unsafe_allow_html=True)
        st.markdown(card_agv(label, ttc_val, zone_clear), unsafe_allow_html=True)
    else:
        st.markdown(card_waiting(), unsafe_allow_html=True)
        st.markdown(card_waiting(), unsafe_allow_html=True)
        st.markdown(card_waiting(), unsafe_allow_html=True)
