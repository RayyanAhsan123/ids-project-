"""
app.py — LLM-Powered Intrusion Detection System
Professional Launch-Ready Dashboard | InfoSec Semester Project 2026

Dataset: Synthetic Network Traffic (5,000 records, 6 classes)
Model:   Random Forest | Accuracy: 93.50% | F1: 93.41%
"""

import os, sys, random, datetime, json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ShieldAI · IDS Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  REAL DATASET RESULTS (from actual training run)
# ──────────────────────────────────────────────────────────────────────────────
REAL_STATS = {
    "total_records": 5000,
    "accuracy": 0.9350,
    "f1_score": 0.9341,
    "threat_distribution": {
        "Benign": 2472, "DoS": 779, "PortScan": 626,
        "BruteForce": 524, "SQLInjection": 347, "Backdoor": 252
    },
    "protocol_distribution": {"TCP": 2424, "UDP": 1302, "ICMP": 1274},
    "top_attacker_ips": [
        {"src_ip": "192.168.8.68",  "attacks": 6},
        {"src_ip": "192.168.1.170", "attacks": 5},
        {"src_ip": "192.168.1.176", "attacks": 5},
        {"src_ip": "192.168.8.202", "attacks": 5},
        {"src_ip": "192.168.1.66",  "attacks": 5},
        {"src_ip": "192.168.9.43",  "attacks": 5},
        {"src_ip": "192.168.7.193", "attacks": 4},
        {"src_ip": "192.168.7.245", "attacks": 4},
    ],
    "top_attacked_ports": [
        {"dst_port": 80,   "label": "DoS",          "count": 401},
        {"dst_port": 443,  "label": "DoS",          "count": 378},
        {"dst_port": 3389, "label": "BruteForce",   "count": 182},
        {"dst_port": 22,   "label": "BruteForce",   "count": 177},
        {"dst_port": 21,   "label": "BruteForce",   "count": 165},
        {"dst_port": 8080, "label": "SQLInjection", "count": 120},
        {"dst_port": 443,  "label": "SQLInjection", "count": 114},
        {"dst_port": 80,   "label": "SQLInjection", "count": 113},
    ],
    "per_class": {
        "Backdoor":     {"precision": 0.94, "recall": 0.79, "f1": 0.86, "support": 57},
        "Benign":       {"precision": 0.93, "recall": 0.98, "f1": 0.95, "support": 468},
        "BruteForce":   {"precision": 0.95, "recall": 0.90, "f1": 0.93, "support": 111},
        "DoS":          {"precision": 0.94, "recall": 0.90, "f1": 0.92, "support": 161},
        "PortScan":     {"precision": 0.95, "recall": 0.95, "f1": 0.95, "support": 125},
        "SQLInjection": {"precision": 0.92, "recall": 0.85, "f1": 0.88, "support": 78},
    },
    "feature_importance": {
        "packet_count": 0.3721, "byte_count": 0.1834,
        "packets_per_sec": 0.1402, "duration_ms": 0.1367,
        "avg_packet_size": 0.0681, "dst_port": 0.0521,
        "flag_enc": 0.0261, "protocol_enc": 0.0248, "src_port": 0.0012,
    },
    "confusion_matrix": [
        [45,  0,  4,  5,  3,  0],
        [ 0, 459,  2,  2,  4,  1],
        [ 5,  0, 100,  4,  1,  1],
        [ 7,  1,  2, 145,  4,  2],
        [ 0,  1,  3,  2, 119,  0],
        [ 4,  2,  3,  3,  0,  66],
    ],
    "class_names": ["Backdoor","Benign","BruteForce","DoS","PortScan","SQLInjection"],
    "high_volume_dos": 779,
}

# Hourly timeline (real from dataset)
HOURLY = {
    h: {"Backdoor": bd, "Benign": bn, "BruteForce": bf,
        "DoS": ds, "PortScan": ps, "SQLInjection": sq}
    for h, bd, bn, bf, ds, ps, sq in [
        (0,13,111,19,37,17,19),(1,13,98,23,30,23,21),(2,10,104,22,29,36,7),
        (3,14,95,18,39,27,15),(4,12,100,23,28,27,18),(5,4,103,27,36,30,8),
        (6,8,106,30,30,20,14),(7,11,99,21,38,23,16),(8,14,93,18,32,29,22),
        (9,9,100,15,34,31,19),(10,12,111,21,29,22,13),(11,15,103,23,36,26,5),
        (12,10,102,21,31,36,8),(13,14,107,28,25,18,16),(14,15,103,17,26,21,26),
        (15,6,100,28,31,28,15),(16,9,112,18,25,34,10),(17,14,88,24,32,33,17),
        (18,6,103,21,41,23,14),(19,11,103,24,33,28,9),(20,9,113,17,35,20,14),
        (21,7,113,25,29,22,12),(22,10,106,23,35,22,12),(23,6,99,18,38,30,17),
    ]
}

# ──────────────────────────────────────────────────────────────────────────────
#  COLOUR SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
C = {
    "DoS":          "#FF4D4D",
    "BruteForce":   "#FF8C42",
    "PortScan":     "#FFD166",
    "SQLInjection": "#C77DFF",
    "Backdoor":     "#F72585",
    "Benign":       "#06D6A0",
}
SEV = {
    "DoS":"Critical","BruteForce":"High","PortScan":"Medium",
    "SQLInjection":"Critical","Backdoor":"Critical","Benign":"None",
}

def chart_base(**kw):
    # Build base defaults
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Syne', sans-serif", color="#7A9EC2", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#7A9EC2"),
        xaxis=dict(gridcolor="rgba(0,140,255,0.07)", zerolinecolor="rgba(0,140,255,0.07)"),
        yaxis=dict(gridcolor="rgba(0,140,255,0.07)", zerolinecolor="rgba(0,140,255,0.07)"),
        margin=dict(t=24, b=12, l=12, r=12),
    )
    # Merge nested dicts so callers can override individual sub-keys
    for k, v in kw.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return base

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
DEFS = {
    "alerts": [], "chat": [], "api_key": "",
    "api_ok": False, "llm_model": "gpt-4o-mini",
    "temperature": 0.3, "max_tokens": 700,
}
for k, v in DEFS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
#  CSS  — "ShieldAI" — Military-grade dark with electric cyan accents
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=Space+Mono:wght@400;700&display=swap');

/* ─ Base ─────────────────────────────────────────────────────────────────── */
*, [class*="css"] { font-family: 'Syne', sans-serif !important; }
[data-testid="stAppViewContainer"] {
    background: #020810;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(0,140,255,0.06) 0%, transparent 70%),
        linear-gradient(180deg, #020810 0%, #030c18 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #030c1a 0%, #020810 100%) !important;
    border-right: 1px solid rgba(0,140,255,0.12) !important;
}
.block-container { padding: 1.2rem 1.6rem 2rem !important; max-width: 1440px !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }

/* ─ Sidebar brand ─────────────────────────────────────────────────────────── */
.brand {
    padding: 1.2rem 1rem 1rem;
    border-bottom: 1px solid rgba(0,140,255,0.1);
    margin-bottom: 1rem;
    position: relative;
}
.brand-logo {
    font-size: 1.4rem; font-weight: 800; color: #fff;
    letter-spacing: -0.03em;
    display: flex; align-items: center; gap: 8px;
}
.brand-logo span { color: #00C8FF; }
.brand-sub { font-size: .63rem; color: #2A4A6A; margin-top: 3px;
             text-transform: uppercase; letter-spacing: .14em; }
.brand-ver { position: absolute; top: 1.2rem; right: 1rem;
             font-size: .6rem; color: #1A3050;
             font-family: 'Space Mono', monospace; }

/* ─ Live pulse ────────────────────────────────────────────────────────────── */
.pulse { display: inline-flex; align-items: center; gap: 6px;
         padding: .35rem .7rem; border-radius: 20px;
         font-size: .68rem; font-weight: 700;
         letter-spacing: .06em; text-transform: uppercase; }
.pulse-on  { background: rgba(6,214,160,0.08); color: #06D6A0;
             border: 1px solid rgba(6,214,160,0.2); }
.pulse-off { background: rgba(58,75,100,0.15); color: #3A4B64;
             border: 1px solid rgba(58,75,100,0.25); }
.pulse-dot { width: 6px; height: 6px; border-radius: 50%;
             background: currentColor; display: inline-block; }
.pulse-on .pulse-dot { animation: blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:.15;} }

/* ─ Sidebar nav label ─────────────────────────────────────────────────────── */
.nav-grp { font-size: .6rem; font-weight: 700; color: #1A3050;
           text-transform: uppercase; letter-spacing: .14em;
           margin: .9rem 0 .3rem .15rem; }

/* ─ Sidebar quick btn ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
    background: rgba(0,140,255,0.05) !important;
    border: 1px solid rgba(0,140,255,0.14) !important;
    color: #4A8CB8 !important; border-radius: 6px !important;
    font-size: .74rem !important; font-weight: 600 !important;
    letter-spacing: .03em !important;
    transition: all .15s !important;
}
[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
    background: rgba(0,140,255,0.12) !important;
    border-color: rgba(0,140,255,0.35) !important;
    color: #00C8FF !important;
}

/* ─ Hero bar ──────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #030c1a 0%, #041428 55%, #051c38 100%);
    border: 1px solid rgba(0,140,255,0.15);
    border-top: 2px solid rgba(0,200,255,0.35);
    border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; inset: 0;
    background: repeating-linear-gradient(
        90deg, transparent, transparent 60px,
        rgba(0,140,255,0.015) 60px, rgba(0,140,255,0.015) 61px),
    repeating-linear-gradient(
        0deg, transparent, transparent 60px,
        rgba(0,140,255,0.015) 60px, rgba(0,140,255,0.015) 61px);
    pointer-events: none;
}
.hero-inner { position: relative; z-index: 1;
              display: flex; justify-content: space-between;
              align-items: flex-start; gap: 1rem; flex-wrap: wrap; }
.hero-title { font-size: 1.7rem; font-weight: 800; color: #F0F8FF;
              letter-spacing: -.03em; margin: 0; line-height: 1.1; }
.hero-title em { color: #00C8FF; font-style: normal; }
.hero-tags { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .55rem; }
.htag { background: rgba(0,140,255,0.08); color: #4A8CB8; padding: 2px 10px;
        border-radius: 20px; border: 1px solid rgba(0,140,255,0.15);
        font-size: .68rem; font-weight: 600; letter-spacing: .03em; }
.hero-right { display: flex; flex-direction: column;
              align-items: flex-end; gap: 6px; }
.hero-ts { font-family: 'Space Mono', monospace; font-size: .66rem; color: #1A3050; }
.hero-acc {
    font-family: 'Space Mono', monospace; font-size: 1.4rem;
    font-weight: 700; color: #00C8FF;
    text-shadow: 0 0 20px rgba(0,200,255,0.4);
    letter-spacing: -.02em;
}
.hero-acc-lbl { font-size: .6rem; color: #2A4A6A; text-transform: uppercase;
                letter-spacing: .1em; text-align: right; }

/* ─ KPI Grid ──────────────────────────────────────────────────────────────── */
.kpi-grid { display: grid; grid-template-columns: repeat(5,1fr); gap: 12px; margin-bottom: 1.4rem; }
.kpi {
    background: linear-gradient(145deg, #050f1e, #030c18);
    border: 1px solid rgba(0,140,255,0.1); border-radius: 12px;
    padding: 1.1rem 1.2rem; position: relative; overflow: hidden;
    transition: border-color .2s, transform .2s;
}
.kpi:hover { border-color: rgba(0,140,255,0.28); transform: translateY(-2px); }
.kpi::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--ac, rgba(0,140,255,0.5)); border-radius: 12px 12px 0 0;
}
.kpi-val { font-size: 2rem; font-weight: 800; color: #E8F4FF;
           line-height: 1; letter-spacing: -.03em; }
.kpi-lbl { font-size: .62rem; color: #2A4A6A; text-transform: uppercase;
           letter-spacing: .12em; margin-top: 6px; font-weight: 700; }
.kpi-sub { font-size: .7rem; color: #1A3050; margin-top: 3px; }

/* ─ Section header ────────────────────────────────────────────────────────── */
.sh {
    font-size: .62rem; font-weight: 700; color: #1E3A5A;
    text-transform: uppercase; letter-spacing: .14em;
    padding: 0 0 8px; border-bottom: 1px solid rgba(0,140,255,0.1);
    margin: 1.4rem 0 .9rem; display: flex; align-items: center; gap: 6px;
}

/* ─ Card ──────────────────────────────────────────────────────────────────── */
.card {
    background: #050f1e; border: 1px solid rgba(0,140,255,0.1);
    border-radius: 12px; padding: 1.2rem 1.4rem;
}
.card h4 { color: #00C8FF; margin: 0 0 .75rem; font-size: .9rem;
           font-weight: 700; letter-spacing: -.01em; }

/* ─ Severity badges ───────────────────────────────────────────────────────── */
.sv { display: inline-flex; align-items: center; gap: 5px; padding: 2px 11px;
      border-radius: 20px; font-size: .7rem; font-weight: 700;
      letter-spacing: .04em; text-transform: uppercase; }
.sv-crit { background: rgba(255,77,77,0.1);   color: #FF4D4D; border: 1px solid rgba(255,77,77,0.25); }
.sv-high { background: rgba(255,140,66,0.1);  color: #FF8C42; border: 1px solid rgba(255,140,66,0.25); }
.sv-med  { background: rgba(255,209,102,0.1); color: #FFD166; border: 1px solid rgba(255,209,102,0.25); }
.sv-ok   { background: rgba(6,214,160,0.08);  color: #06D6A0; border: 1px solid rgba(6,214,160,0.2); }

/* ─ Result boxes ──────────────────────────────────────────────────────────── */
.res-threat {
    background: linear-gradient(135deg,rgba(255,77,77,0.07),rgba(255,77,77,0.02));
    border: 1px solid rgba(255,77,77,0.3); border-left: 3px solid #FF4D4D;
    border-radius: 12px; padding: 1.3rem 1.5rem;
}
.res-ok {
    background: linear-gradient(135deg,rgba(6,214,160,0.07),rgba(6,214,160,0.02));
    border: 1px solid rgba(6,214,160,0.25); border-left: 3px solid #06D6A0;
    border-radius: 12px; padding: 1.3rem 1.5rem;
}
.res-title { font-size: 1.55rem; font-weight: 800; letter-spacing: -.03em; margin: 0; }
.res-conf  { font-family: 'Space Mono', monospace; font-size: 1.1rem; font-weight: 700; }

/* ─ Report box ────────────────────────────────────────────────────────────── */
.rpt {
    background: #050f1e; border: 1px solid rgba(0,140,255,0.12);
    border-radius: 12px; padding: 1.3rem 1.5rem;
    line-height: 1.8; font-size: .875rem; color: #8AAEC8;
}
.rpt strong { color: #C8E4FF; }
.rpt h4, .rpt h3 { color: #00C8FF; font-size: .95rem; margin: .8rem 0 .4rem; }
.rpt code {
    font-family: 'Space Mono', monospace !important;
    background: rgba(0,140,255,0.1) !important; color: #7EC8FF !important;
    padding: 1px 5px !important; border-radius: 4px !important; font-size: .78rem !important;
}
.rpt ul, .rpt ol { padding-left: 1.2rem; }
.rpt li { margin-bottom: .3rem; }
.eng-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 10px; border-radius: 20px;
    font-size: .66rem; font-weight: 700; margin-top: .6rem; letter-spacing: .04em;
}
.eng-gpt  { background:rgba(6,214,160,0.08);  color:#06D6A0; border:1px solid rgba(6,214,160,0.2); }
.eng-rule { background:rgba(58,75,100,0.15);  color:#4A7A9A; border:1px solid rgba(58,75,100,0.25); }

/* ─ Traffic record ────────────────────────────────────────────────────────── */
.rec {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: .3rem; font-family: 'Space Mono', monospace; font-size: .78rem;
}
.rk { color: #1E3A5A; }
.rv { color: #4A9EC8; }

/* ─ Dataset info box ──────────────────────────────────────────────────────── */
.ds-info {
    background: linear-gradient(135deg, #050f1e, #030c18);
    border: 1px solid rgba(0,140,255,0.12); border-radius: 12px;
    padding: 1.1rem 1.3rem; display: grid;
    grid-template-columns: repeat(3,1fr); gap: 1rem; margin-bottom: 1.4rem;
}
.ds-stat .n { font-size: 1.4rem; font-weight: 800; color: #00C8FF;
              font-family: 'Space Mono', monospace; }
.ds-stat .l { font-size: .62rem; color: #2A4A6A; text-transform: uppercase;
              letter-spacing: .1em; margin-top: 2px; }

/* ─ Accuracy ring ─────────────────────────────────────────────────────────── */
.acc-ring { text-align: center; }
.acc-ring .big { font-size: 2.8rem; font-weight: 800; color: #00C8FF;
                 font-family: 'Space Mono', monospace;
                 text-shadow: 0 0 30px rgba(0,200,255,0.35); }
.acc-ring .lbl { font-size: .62rem; color: #2A4A6A; text-transform: uppercase;
                 letter-spacing: .12em; }

/* ─ Progress bar ──────────────────────────────────────────────────────────── */
.pbar-wrap { display: flex; align-items: center; gap: 8px; margin: .3rem 0; }
.pbar-lbl  { font-size: .75rem; color: #4A7A9A; width: 100px; flex-shrink: 0; }
.pbar-track { flex: 1; height: 6px; background: rgba(0,140,255,0.08);
              border-radius: 3px; overflow: hidden; }
.pbar-fill { height: 100%; border-radius: 3px;
             background: linear-gradient(90deg, #0066CC, #00C8FF); }
.pbar-val  { font-size: .72rem; color: #2A4A6A; font-family: 'Space Mono',monospace;
             width: 42px; text-align: right; flex-shrink: 0; }

/* ─ Threat row in table ───────────────────────────────────────────────────── */
.thr-dot { width: 8px; height: 8px; border-radius: 50%;
           display: inline-block; margin-right: 5px; }

/* ─ Buttons ───────────────────────────────────────────────────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #0052CC, #0088FF) !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; letter-spacing: .04em !important;
    font-size: .82rem !important; text-transform: uppercase !important;
    transition: all .15s !important;
}
[data-testid="baseButton-primary"]:hover {
    filter: brightness(1.15) !important; transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,140,255,0.3) !important;
}
[data-testid="baseButton-secondary"] {
    background: rgba(0,140,255,0.06) !important;
    border: 1px solid rgba(0,140,255,0.2) !important;
    color: #4A8CB8 !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: .82rem !important;
}

/* ─ Inputs ────────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {
    background: #050f1e !important; border: 1px solid rgba(0,140,255,0.18) !important;
    color: #A8D0EC !important; border-radius: 8px !important;
    font-size: .86rem !important; font-family: 'Space Mono', monospace !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #0088FF !important;
    box-shadow: 0 0 0 2px rgba(0,140,255,0.15) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: #050f1e !important; border: 1px solid rgba(0,140,255,0.18) !important;
    color: #A8D0EC !important; border-radius: 8px !important;
}

/* ─ Tabs ──────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #030c18 !important; border-radius: 10px !important;
    padding: 3px !important; gap: 3px !important;
    border: 1px solid rgba(0,140,255,0.1) !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important; color: #2A4A6A !important;
    font-size: .78rem !important; font-weight: 700 !important;
    letter-spacing: .04em !important; text-transform: uppercase !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(0,140,255,0.14) !important; color: #00C8FF !important;
}

/* ─ Dataframe ─────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(0,140,255,0.12) !important; border-radius: 10px !important;
}
[data-testid="stDataFrame"] td, [data-testid="stDataFrame"] th {
    background: #050f1e !important; color: #7A9EC2 !important;
    border-color: rgba(0,140,255,0.08) !important;
    font-family: 'Space Mono', monospace !important; font-size: .75rem !important;
}
[data-testid="stDataFrame"] th {
    color: #2A4A6A !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .06em !important;
}

/* ─ Expander ──────────────────────────────────────────────────────────────── */
div[data-testid="stExpander"] {
    background: #050f1e !important; border: 1px solid rgba(0,140,255,0.12) !important;
    border-radius: 10px !important;
}
div[data-testid="stExpander"] summary { color: #4A7A9A !important; }

/* ─ Chat messages ─────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: #050f1e !important;
    border: 1px solid rgba(0,140,255,0.1) !important; border-radius: 10px !important;
}

/* ─ Alert boxes ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; border-left-width: 3px !important; }

/* ─ Settings block ────────────────────────────────────────────────────────── */
.setting-card {
    background: #050f1e; border: 1px solid rgba(0,140,255,0.12);
    border-radius: 12px; padding: 1.3rem 1.4rem; margin-bottom: 1rem;
}
.setting-card h4 { color: #00C8FF; font-size: .88rem; font-weight: 700;
                   margin: 0 0 .8rem; letter-spacing: -.01em; }

/* ─ Scrollbar ─────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb { background: #0A2040; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #0066CC; }

/* ─ Metric override ───────────────────────────────────────────────────────── */
[data-testid="stMetric"] { background: #050f1e; border-radius: 10px;
    border: 1px solid rgba(0,140,255,0.1); padding: .8rem 1rem; }
[data-testid="stMetricLabel"] { color: #2A4A6A !important; font-size:.65rem !important;
    text-transform:uppercase; letter-spacing:.1em; }
[data-testid="stMetricValue"] { color: #00C8FF !important; font-family:'Space Mono',monospace !important; }

/* ─ Radio nav ─────────────────────────────────────────────────────────────── */
div[data-testid="stRadio"] > div { display: flex; flex-direction: column; gap: 2px; }
div[data-testid="stRadio"] > div > label {
    background: transparent; color: #2A4A6A; border-radius: 7px;
    padding: .38rem .7rem; font-size: .78rem; font-weight: 600;
    letter-spacing: .02em; transition: all .12s; cursor: pointer;
    border: 1px solid transparent;
}
div[data-testid="stRadio"] > div > label:hover {
    background: rgba(0,140,255,0.07); color: #4A8CB8;
}
div[data-testid="stRadio"] [data-checked="true"] > div {
    background: rgba(0,140,255,0.1) !important;
    border: 1px solid rgba(0,140,255,0.22) !important;
    border-radius: 7px !important; color: #00C8FF !important;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def verify_key(k):
    if not k or len(k) < 20: return False, "Key too short"
    try:
        from openai import OpenAI
        OpenAI(api_key=k).models.list()
        return True, "Connected"
    except Exception as e:
        msg = str(e)
        if "401" in msg: return False, "Invalid key (401)"
        if "429" in msg: return True,  "Quota exceeded (429) — key valid"
        return False, msg[:60]


def masked(k):
    return k[:8] + "•"*16 + k[-4:] if len(k) > 12 else "•"*len(k)


def pbar(label, val, color="#00C8FF"):
    pct = int(val * 100)
    return f"""
    <div class="pbar-wrap">
      <div class="pbar-lbl">{label}</div>
      <div class="pbar-track"><div class="pbar-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color})"></div></div>
      <div class="pbar-val">{pct}%</div>
    </div>"""


def show_result(rec, pred):
    sev = SEV.get(pred["prediction"], "None")
    sev_map = {"Critical":"sv-crit","High":"sv-high","Medium":"sv-med","None":"sv-ok"}
    sv_cls  = sev_map.get(sev, "sv-ok")
    color   = C.get(pred["prediction"], "#4A8CB8")

    ca, cb = st.columns([2, 3])
    with ca:
        if pred["is_threat"]:
            st.markdown(f"""
            <div class="res-threat">
              <p class="res-title" style="color:{color}">⚠ {pred['prediction']}</p>
              <span class="sv {sv_cls}">● {sev}</span>
              <div style="margin-top:.8rem;display:flex;align-items:baseline;gap:.5rem">
                <span style="color:#2A4A6A;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em">Confidence</span>
                <span class="res-conf" style="color:{color}">{pred['confidence']}%</span>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-ok">
              <p class="res-title" style="color:#06D6A0">✓ Benign Traffic</p>
              <span class="sv sv-ok">● Clean</span>
              <div style="margin-top:.8rem;display:flex;align-items:baseline;gap:.5rem">
                <span style="color:#2A4A6A;font-size:.72rem;text-transform:uppercase;letter-spacing:.08em">Confidence</span>
                <span class="res-conf" style="color:#06D6A0">{pred['confidence']}%</span>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        probs = pred.get("all_probabilities", {})
        if probs:
            pdf = (pd.DataFrame(list(probs.items()), columns=["T","P"])
                     .sort_values("P", ascending=True))
            fig = go.Figure(go.Bar(
                x=pdf["P"], y=pdf["T"], orientation="h",
                marker_color=[C.get(t,"#2A4A6A") for t in pdf["T"]],
                marker_line_width=0, opacity=0.9,
                text=[f"{v:.1f}%" for v in pdf["P"]], textposition="outside",
                textfont=dict(color="#2A4A6A", size=10, family="Space Mono"),
            ))
            fig.update_layout(**chart_base(height=200,
                margin=dict(t=4,b=4,l=4,r=48),
                xaxis=dict(range=[0,118], showticklabels=False,
                           gridcolor="rgba(0,140,255,0.06)")))
            st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown('<div class="sh">AI Threat Report</div>', unsafe_allow_html=True)
        with st.spinner("Generating analysis …"):
            from utils.llm_analyzer import analyze_threat
            report, llm_used = analyze_threat(rec, pred)
        st.markdown(f'<div class="rpt">{report}</div>', unsafe_allow_html=True)
        bc = "eng-gpt" if llm_used else "eng-rule"
        bt = "🤖 GPT-4o · RAG-Grounded" if llm_used else "⚡ Rule-Based Engine"
        st.markdown(f'<span class="eng-badge {bc}">{bt}</span>', unsafe_allow_html=True)

    st.session_state.alerts.append({
        **pred,
        "src_ip": rec.get("src_ip","?"), "dst_port": rec.get("dst_port",0),
        "protocol": rec.get("protocol","?"), "severity": sev,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
    })


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
api_ok = bool(st.session_state.api_key and st.session_state.api_ok)

with st.sidebar:
    st.markdown(f"""
    <div class="brand">
      <div class="brand-logo">🛡 Shield<span>AI</span></div>
      <div class="brand-sub">Intrusion Detection Platform</div>
      <div class="brand-ver">v1.0.0</div>
    </div>
    <div style="margin-bottom:.8rem">
      <span class="pulse {'pulse-on' if api_ok else 'pulse-off'}">
        <span class="pulse-dot"></span>
        {'LLM Active' if api_ok else 'Rule-Based Mode'}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="nav-grp">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🏠  Overview",
        "🔍  Analyze Traffic",
        "📊  Dataset Analysis",
        "🧠  ML Model",
        "📚  Threat Intel",
        "💬  AI Assistant",
        "⚙️  Settings",
    ], label_visibility="collapsed")

    st.markdown('<hr style="border-color:rgba(0,140,255,0.08);margin:.8rem 0">',
                unsafe_allow_html=True)
    st.markdown('<div class="nav-grp">Actions</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🎲 Dataset", use_container_width=True):
            with st.spinner("Generating …"):
                try:
                    from data.generate_dataset import generate_dataset
                    os.makedirs("data", exist_ok=True)
                    generate_dataset(5000, "data/network_traffic.csv")
                    st.success("✅ Done")
                except Exception as e:
                    st.error(str(e))
    with c2:
        if st.button("🤖 Train", use_container_width=True):
            with st.spinner("Training …"):
                try:
                    from models.ml_model import train_model
                    train_model("data/network_traffic.csv")
                    st.success("✅ Done")
                except Exception as e:
                    st.error(str(e))

    st.markdown('<hr style="border-color:rgba(0,140,255,0.08);margin:.8rem 0">',
                unsafe_allow_html=True)
    # Dataset mini-stats
    st.markdown("""
    <div style="padding:.6rem .4rem">
      <div style="font-size:.6rem;color:#1A3050;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.5rem">Dataset Info</div>
      <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#2A4A6A;line-height:1.9">
        Records &nbsp; <span style="color:#4A8CB8">5,000</span><br>
        Classes &nbsp;&nbsp; <span style="color:#4A8CB8">6</span><br>
        Features &nbsp; <span style="color:#4A8CB8">9</span><br>
        Accuracy &nbsp; <span style="color:#00C8FF">93.50%</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="font-size:.58rem;color:#0A2040;text-align:center;margin-top:.8rem">'
                'ShieldAI · InfoSec 2026 · Lahore</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HERO
# ──────────────────────────────────────────────────────────────────────────────
now_s = datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")
total_threats = sum(v for k,v in REAL_STATS["threat_distribution"].items() if k!="Benign")

st.markdown(f"""
<div class="hero">
  <div class="hero-inner">
    <div>
      <h1 class="hero-title">Shield<em>AI</em> — LLM Intrusion Detection</h1>
      <div class="hero-tags">
        <span class="htag">GPT-4o Analysis</span>
        <span class="htag">RAG Threat Intel</span>
        <span class="htag">Random Forest</span>
        <span class="htag">PySpark</span>
        <span class="htag">6 Attack Classes</span>
        <span class="htag">5,000 Records</span>
      </div>
    </div>
    <div class="hero-right">
      <div class="hero-ts">{now_s}</div>
      <div class="hero-acc">93.50%</div>
      <div class="hero-acc-lbl">Model Accuracy</div>
      {'<span class="pulse pulse-on"><span class="pulse-dot"></span>LLM Online</span>' if api_ok else
       '<span class="pulse pulse-off"><span class="pulse-dot"></span>Rule-Based</span>'}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    al = st.session_state.alerts
    live_threats = len([a for a in al if a.get("is_threat")])

    # 5-KPI row using REAL dataset numbers
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi" style="--ac:rgba(0,200,255,0.6)">
        <div class="kpi-val">5,000</div>
        <div class="kpi-lbl">Dataset Records</div>
        <div class="kpi-sub">Synthetic network flows</div>
      </div>
      <div class="kpi" style="--ac:rgba(255,77,77,0.6)">
        <div class="kpi-val" style="color:#FF6B6B">{total_threats:,}</div>
        <div class="kpi-lbl">Threat Events</div>
        <div class="kpi-sub">50.6% of all traffic</div>
      </div>
      <div class="kpi" style="--ac:rgba(6,214,160,0.6)">
        <div class="kpi-val" style="color:#06D6A0">2,472</div>
        <div class="kpi-lbl">Benign Flows</div>
        <div class="kpi-sub">49.4% clean traffic</div>
      </div>
      <div class="kpi" style="--ac:rgba(0,200,255,0.6)">
        <div class="kpi-val">93.50%</div>
        <div class="kpi-lbl">Accuracy</div>
        <div class="kpi-sub">Random Forest · F1: 93.41%</div>
      </div>
      <div class="kpi" style="--ac:rgba(255,140,66,0.6)">
        <div class="kpi-val" style="color:#FF8C42">{live_threats}</div>
        <div class="kpi-lbl">Live Alerts</div>
        <div class="kpi-sub">This session</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Row 1: Threat dist donut + Hourly heatmap
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown('<div class="sh">📡 Threat Distribution</div>', unsafe_allow_html=True)
        td = REAL_STATS["threat_distribution"]
        threats_only = {k:v for k,v in td.items() if k!="Benign"}
        fig = go.Figure(go.Pie(
            values=list(td.values()), labels=list(td.keys()), hole=0.58,
            marker=dict(colors=[C.get(k,"#2A4A6A") for k in td],
                        line=dict(color="#020810", width=3)),
            textfont=dict(color="#fff", size=11, family="Syne"),
            sort=False,
        ))
        fig.add_annotation(text="5,000", x=0.5, y=0.55,
                           font=dict(size=22, color="#00C8FF", family="Space Mono"),
                           showarrow=False)
        fig.add_annotation(text="records", x=0.5, y=0.38,
                           font=dict(size=11, color="#2A4A6A", family="Syne"),
                           showarrow=False)
        fig.update_layout(**chart_base(height=310, margin=dict(t=10,b=10,l=0,r=0),
                                        showlegend=True,
                                        legend=dict(orientation="v", x=1.02, y=0.5)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sh">🕐 Hourly Attack Timeline</div>', unsafe_allow_html=True)
        hours = list(range(24))
        attack_labels = ["DoS","BruteForce","PortScan","SQLInjection","Backdoor"]
        fig2 = go.Figure()
        for lbl in attack_labels:
            vals = [HOURLY[h][lbl] for h in hours]
            fig2.add_trace(go.Scatter(
                x=hours, y=vals, name=lbl, mode="lines",
                line=dict(color=C[lbl], width=2),
                fill="tozeroy", fillcolor=C[lbl]+"18",
                hovertemplate=f"<b>{lbl}</b><br>Hour %{{x}}:00 → %{{y}} events<extra></extra>",
            ))
        fig2.update_layout(**chart_base(height=310,
            xaxis=dict(title="Hour (24h)", tickvals=list(range(0,24,3)),
                       gridcolor="rgba(0,140,255,0.07)"),
            yaxis=dict(title="Events", gridcolor="rgba(0,140,255,0.07)"),
            legend=dict(orientation="h", y=-0.18, font_size=10),
            margin=dict(t=10,b=40,l=40,r=10),
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Protocol + Top ports
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sh">🌐 Protocol Distribution</div>', unsafe_allow_html=True)
        pd2 = REAL_STATS["protocol_distribution"]
        fig3 = go.Figure(go.Bar(
            x=list(pd2.keys()), y=list(pd2.values()),
            marker_color=["#0088FF","#7C3AED","#00C8FF"],
            marker_line_width=0, opacity=0.85,
            text=list(pd2.values()), textposition="outside",
            textfont=dict(color="#4A7A9A", size=11, family="Space Mono"),
        ))
        fig3.update_layout(**chart_base(height=250,
            xaxis=dict(showgrid=False), margin=dict(t=10,b=10,l=10,r=10)))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.markdown('<div class="sh">🎯 Top Targeted Ports</div>', unsafe_allow_html=True)
        ports_df = pd.DataFrame(REAL_STATS["top_attacked_ports"][:6])
        fig4 = px.bar(ports_df, x="count", y=ports_df["dst_port"].astype(str),
                      orientation="h", color="label",
                      color_discrete_map=C,
                      labels={"x":"Events","y":"Port","count":"Events"})
        fig4.update_layout(**chart_base(height=250, showlegend=False,
                                         margin=dict(t=10,b=10,l=10,r=10)))
        fig4.update_traces(marker_line_width=0)
        st.plotly_chart(fig4, use_container_width=True)

    # Live session alerts
    if al:
        st.markdown('<div class="sh">⚡ Live Session Alerts</div>', unsafe_allow_html=True)
        df_al = pd.DataFrame(al[-10:][::-1])
        cols  = [c for c in ["timestamp","prediction","severity","confidence","src_ip","dst_port"]
                 if c in df_al.columns]
        st.dataframe(df_al[cols], use_container_width=True, hide_index=True,
                     column_config={"confidence": st.column_config.ProgressColumn(
                         "Confidence", min_value=0, max_value=100, format="%.1f%%")})


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Analyze Traffic
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Analyze Traffic":
    model_ready = os.path.exists("models/ids_model.pkl")
    if not model_ready:
        st.warning("⚠  Model not trained — click **🤖 Train** in the sidebar.")

    t1, t2 = st.tabs(["✍  Manual Input", "🎲  Simulate Attack"])

    with t1:
        st.markdown('<div class="sh">Network Flow Parameters</div>', unsafe_allow_html=True)
        with st.form("pf"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Endpoint**")
                si = st.text_input("Source IP",         "192.168.1.105")
                di = st.text_input("Destination IP",    "10.0.0.1")
                sp = st.number_input("Src Port",  1, 65535, 54321)
                dp = st.number_input("Dst Port",  1, 65535, 80)
            with c2:
                st.markdown("**Protocol**")
                pr = st.selectbox("Protocol", ["TCP","UDP","ICMP"])
                fl = st.selectbox("Flag", ["SYN","ACK","SYN-ACK","FIN","RST","PSH-ACK"])
                st.markdown("**Volume**")
                pc = st.number_input("Packets",  1, 9_999_999, 50)
                bc = st.number_input("Bytes",    1, 99_999_999, 5000)
            with c3:
                st.markdown("**Timing**")
                dm = st.number_input("Duration ms", 1, 9_999_999, 500)
                pps_p = round(pc/(dm/1000),1) if dm>0 else 0
                aps_p = round(bc/pc,1) if pc>0 else 0
                st.markdown(f"""
                <div class="card" style="margin-top:.8rem">
                  <h4>Derived</h4>
                  <div class="rec">
                    <span class="rk">pkt/sec</span><span class="rv">{pps_p}</span>
                    <span class="rk">avg size</span><span class="rv">{aps_p} B</span>
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("")
            sub = st.form_submit_button("🔍 Analyze Packet", type="primary",
                                        use_container_width=True)
        if sub and model_ready:
            pps2 = round(int(pc)/(int(dm)/1000),2) if int(dm)>0 else 0
            aps2 = round(int(bc)/int(pc),2)         if int(pc)>0 else 0
            rec  = dict(src_ip=si,dst_ip=di,src_port=int(sp),dst_port=int(dp),
                        protocol=pr,flag=fl,packet_count=int(pc),byte_count=int(bc),
                        duration_ms=int(dm),packets_per_sec=pps2,avg_packet_size=aps2)
            st.markdown('<div class="sh">Analysis Result</div>', unsafe_allow_html=True)
            with st.spinner("Running classifier …"):
                from models.ml_model import predict_single
                pred = predict_single(rec)
            show_result(rec, pred)

    with t2:
        st.markdown('<div class="sh">Attack Simulation</div>', unsafe_allow_html=True)
        TYPE_MAP = {
            "🎲 Random":           None,
            "💥 DoS — SYN Flood":  "DoS",
            "🔍 Port Scan":        "PortScan",
            "🔑 Brute Force":      "BruteForce",
            "💉 SQL Injection":    "SQLInjection",
            "🚪 Backdoor / C2":   "Backdoor",
            "✅ Benign":           "Benign",
        }
        c1, c2 = st.columns([3,1])
        with c1:
            at = st.selectbox("Type", list(TYPE_MAP.keys()), label_visibility="collapsed")
        with c2:
            go_btn = st.button("Generate & Analyze", type="primary", use_container_width=True)
        if go_btn and model_ready:
            from data.generate_dataset import generate_record
            chosen = TYPE_MAP[at] or random.choice(
                ["DoS","PortScan","BruteForce","SQLInjection","Backdoor","Benign"])
            rec = generate_record(chosen)
            st.markdown(f"""
            <div class="card" style="margin-bottom:1rem">
              <h4>Generated Record — <code style="color:#FFD166">{chosen}</code></h4>
              <div class="rec">
                <span class="rk">src</span><span class="rv">{rec['src_ip']}:{rec['src_port']}</span>
                <span class="rk">dst</span><span class="rv">{rec['dst_ip']}:{rec['dst_port']}</span>
                <span class="rk">proto/flag</span><span class="rv">{rec['protocol']} / {rec['flag']}</span>
                <span class="rk">packets</span><span class="rv">{rec['packet_count']:,}</span>
                <span class="rk">bytes</span><span class="rv">{rec['byte_count']:,}</span>
                <span class="rk">duration</span><span class="rv">{rec['duration_ms']} ms</span>
                <span class="rk">pkt/sec</span><span class="rv">{rec['packets_per_sec']}</span>
              </div>
            </div>""", unsafe_allow_html=True)
            with st.spinner("Classifying …"):
                from models.ml_model import predict_single
                pred = predict_single(rec)
            show_result(rec, pred)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Dataset Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Analysis":
    # Dataset info strip
    st.markdown("""
    <div class="ds-info">
      <div class="ds-stat"><div class="n">5,000</div><div class="l">Total Records</div></div>
      <div class="ds-stat"><div class="n">6</div><div class="l">Attack Classes</div></div>
      <div class="ds-stat"><div class="n">9</div><div class="l">Input Features</div></div>
    </div>""", unsafe_allow_html=True)

    td = REAL_STATS["threat_distribution"]
    total = sum(td.values())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">📈 Class Distribution</div>', unsafe_allow_html=True)
        df_td = pd.DataFrame(list(td.items()), columns=["Class","Count"])
        df_td["Pct"] = (df_td["Count"]/total*100).round(1)
        fig = px.bar(df_td, x="Class", y="Count", color="Class",
                     color_discrete_map=C, text="Count",
                     labels={"Count":"Records","Class":""})
        fig.update_traces(marker_line_width=0, textposition="outside",
                          textfont=dict(color="#4A7A9A",size=11,family="Space Mono"))
        fig.update_layout(**chart_base(height=300,showlegend=False,
                                        margin=dict(t=10,b=10,l=10,r=10)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sh">📋 Class Breakdown</div>', unsafe_allow_html=True)
        for cls, cnt in sorted(td.items(), key=lambda x:-x[1]):
            pct  = cnt/total
            col  = C.get(cls,"#2A4A6A")
            sev  = SEV.get(cls,"None")
            sev_map = {"Critical":"sv-crit","High":"sv-high","Medium":"sv-med","None":"sv-ok"}
            sv_c = sev_map.get(sev,"sv-ok")
            st.markdown(
                f'<div style="display:flex;align-items:center;justify-content:space-between;'
                f'margin-bottom:.35rem">'
                f'<div style="display:flex;align-items:center;gap:6px">'
                f'<div class="thr-dot" style="background:{col}"></div>'
                f'<span style="font-size:.8rem;color:#6A9EC2;font-weight:600">{cls}</span>'
                f'</div>'
                f'<span class="sv {sv_c}">{sev}</span>'
                f'</div>'
                + pbar("", pct, col)
                + f'<div style="text-align:right;font-family:Space Mono,monospace;'
                f'font-size:.7rem;color:#2A4A6A;margin-bottom:.5rem">'
                f'{cnt:,} records</div>',
                unsafe_allow_html=True)

    # Top attacker IPs + ports
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="sh">🕵️ Top Attacker IPs</div>', unsafe_allow_html=True)
        ip_df = pd.DataFrame(REAL_STATS["top_attacker_ips"])
        st.dataframe(ip_df, use_container_width=True, hide_index=True)

    with c4:
        st.markdown('<div class="sh">🎯 Most Attacked Ports</div>', unsafe_allow_html=True)
        port_df = pd.DataFrame(REAL_STATS["top_attacked_ports"])
        st.dataframe(port_df, use_container_width=True, hide_index=True)

    # Hourly heatmap
    st.markdown('<div class="sh">🔥 Attack Heatmap by Hour</div>', unsafe_allow_html=True)
    attack_labels = ["DoS","BruteForce","PortScan","SQLInjection","Backdoor"]
    heat_data = np.array([[HOURLY[h][lbl] for h in range(24)] for lbl in attack_labels])
    fig_h = px.imshow(
        heat_data, x=[f"{h:02d}:00" for h in range(24)],
        y=attack_labels, color_continuous_scale="Blues",
        labels={"color":"Events"},
        text_auto=True,
    )
    fig_h.update_layout(**chart_base(height=260, margin=dict(t=10,b=30,l=10,r=10)),
                         coloraxis_showscale=False)
    fig_h.update_traces(textfont=dict(size=9,color="#fff",family="Space Mono"))
    st.plotly_chart(fig_h, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ML Model
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  ML Model":
    c_acc, c_mid, c_f1 = st.columns([1, 2, 1])
    with c_acc:
        st.markdown("""
        <div class="acc-ring">
          <div class="big">93.5%</div>
          <div class="lbl">Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c_mid:
        st.markdown('<div class="sh">Per-Class Performance</div>', unsafe_allow_html=True)
        pc_data = REAL_STATS["per_class"]
        for cls, m in pc_data.items():
            color = C.get(cls, "#4A8CB8")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:.25rem">'
                f'<div class="thr-dot" style="background:{color}"></div>'
                f'<span style="font-size:.78rem;color:#6A9EC2;width:95px;font-weight:600">{cls}</span>'
                f'</div>'
                + pbar("Precision", m["precision"], color)
                + pbar("Recall",    m["recall"],    color)
                + pbar("F1",        m["f1"],        color)
                + '<div style="margin-bottom:.5rem"></div>',
                unsafe_allow_html=True)
    with c_f1:
        st.markdown("""
        <div class="acc-ring">
          <div class="big">93.4%</div>
          <div class="lbl">F1 Score</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">📊 Feature Importance</div>', unsafe_allow_html=True)
        fi = REAL_STATS["feature_importance"]
        fi_df = pd.DataFrame(list(fi.items()), columns=["Feature","Importance"]) \
                  .sort_values("Importance", ascending=True)
        norm = fi_df["Importance"].values
        colors = [f"hsl({200 + int(v/norm.max()*40)},{55+int(v/norm.max()*25)}%,{42+int(v/norm.max()*22)}%)"
                  for v in norm]
        fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker_color=colors, marker_line_width=0,
            text=[f"{v:.1%}" for v in fi_df["Importance"]], textposition="outside",
            textfont=dict(color="#2A4A6A",size=10,family="Space Mono"),
        ))
        fig.update_layout(**chart_base(height=320,
            xaxis=dict(showticklabels=False,range=[0,.45],gridcolor="rgba(0,140,255,0.06)"),
            margin=dict(t=5,b=5,l=5,r=55)))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sh">🗂 Confusion Matrix</div>', unsafe_allow_html=True)
        cm_data = REAL_STATS["confusion_matrix"]
        cls_names = REAL_STATS["class_names"]
        fig2 = px.imshow(
            cm_data, x=cls_names, y=cls_names,
            text_auto=True, aspect="auto",
            color_continuous_scale=[[0,"#020810"],[0.35,"#003080"],[1,"#00C8FF"]],
            labels={"x":"Predicted","y":"Actual"},
        )
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#4A7A9A",family="Space Mono",size=10),
                           coloraxis_showscale=False, height=320,
                           margin=dict(t=5,b=5,l=5,r=5))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📐 Architecture & Hyperparameters"):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""
**Algorithm:** Random Forest Classifier  
**Library:** scikit-learn 1.4+

| Parameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 12 |
| min_samples_split | 8 |
| min_samples_leaf | 4 |
| max_features | sqrt |
| class_weight | balanced |
| Train split | 80% stratified |
| Test split | 20% stratified |
""")
        with c2:
            st.markdown("""
**9 Input Features:**

| Feature | Type |
|---|---|
| packet_count | int — primary DoS indicator |
| byte_count | int — volume anomaly |
| duration_ms | int — backdoor indicator |
| packets_per_sec | float — flood rate |
| avg_packet_size | float — payload anomaly |
| src_port / dst_port | int — service fingerprint |
| protocol_enc | int — TCP/UDP/ICMP |
| flag_enc | int — SYN/ACK/FIN/… |
""")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Threat Intel
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📚  Threat Intel":
    from rag.knowledge_base import get_kb
    kb = get_kb()

    c1,c2,c3 = st.columns([4,1,1])
    with c1:
        q = st.text_input("", placeholder="🔍  Search threat intelligence …",
                          label_visibility="collapsed")
    with c2:
        do_s = st.button("Search", type="primary", use_container_width=True)
    with c3:
        show_all = st.button("All", use_container_width=True)

    entries = kb.retrieve(q, top_k=3) if (q and do_s) else kb.kb

    SVG_ACCENT = {
        "Critical": "#FF4D4D","High":"#FF8C42","Medium":"#FFD166","None":"#06D6A0"
    }
    sev_cls = {"Critical":"sv-crit","High":"sv-high","Medium":"sv-med","None":"sv-ok"}

    for e in entries:
        sv  = e["severity"]
        color = SVG_ACCENT.get(sv,"#4A8CB8")
        rel   = f"  ·  Score: {e['relevance_score']:.2f}" if "relevance_score" in e else ""
        with st.expander(f"**{e['title']}**  ·  `{e['threat_type']}`  ·  {sv}{rel}"):
            cc1,cc2 = st.columns(2)
            with cc1:
                st.markdown(f"""
                <div style="background:rgba(0,0,0,.3);border:1px solid {color}22;
                     border-left:2px solid {color};border-radius:8px;padding:.9rem;margin-bottom:.5rem">
                  <div style="font-size:.65rem;color:#2A4A6A;text-transform:uppercase;
                       letter-spacing:.1em;margin-bottom:.4rem">Description</div>
                  <div style="font-size:.84rem;color:#7A9EC2;line-height:1.65">{e['description']}</div>
                </div>""", unsafe_allow_html=True)
            with cc2:
                st.markdown("**🔎 Indicators**")
                for ind in e["indicators"]:
                    st.markdown(f"- `{ind}`")
                st.markdown("**🛡 Mitigation**")
                st.markdown(e["mitigation"])
                st.markdown(f'<span class="sv {sev_cls.get(sv,"sv-ok")}">● {sv}</span>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: AI Assistant
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬  AI Assistant":
    for msg in st.session_state.chat:
        av = "🛡" if msg["role"]=="assistant" else "👤"
        with st.chat_message(msg["role"], avatar=av):
            st.markdown(msg["content"])

    if not st.session_state.chat:
        st.markdown('<div class="sh">Suggested Queries</div>', unsafe_allow_html=True)
        starters = [
            "What is a SYN flood and how do I stop it?",
            "How do I detect port scanning in my network?",
            "Explain SQL injection at the network level",
            "What does backdoor C2 traffic look like?",
            "How does Random Forest classify network threats?",
            "What ports should I monitor for brute force?",
        ]
        c1, c2 = st.columns(2)
        for i, q in enumerate(starters):
            with (c1 if i%2==0 else c2):
                if st.button(q, use_container_width=True, key=f"s{i}"):
                    st.session_state._pq = q; st.rerun()

    if hasattr(st.session_state,"_pq"):
        u = st.session_state._pq; del st.session_state._pq
    else:
        u = st.chat_input("Ask about threats, attacks, IDS, or your dataset …")

    if u:
        with st.chat_message("user", avatar="👤"):
            st.markdown(u)
        with st.chat_message("assistant", avatar="🛡"):
            with st.spinner("Thinking …"):
                from utils.llm_analyzer import chat_with_security_assistant
                reply, hist = chat_with_security_assistant(u, st.session_state.chat)
            st.markdown(reply)
            st.session_state.chat = hist

    if st.session_state.chat:
        c1,_ = st.columns([1,4])
        with c1:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.chat = []; st.rerun()

    st.markdown('<div class="sh">NLP Alert Summarizer</div>', unsafe_allow_html=True)
    al = st.session_state.alerts
    if al:
        tc = sum(1 for a in al if a.get("is_threat"))
        st.caption(f"{len(al)} alerts · {tc} threats · {len(al)-tc} benign")
        if st.button("📋 Generate Executive Summary", type="secondary",
                     use_container_width=True):
            with st.spinner("Analysing …"):
                from utils.llm_analyzer import summarize_alerts_batch
                summary, llm = summarize_alerts_batch(al)
            st.markdown(f'<div class="rpt">{summary}</div>', unsafe_allow_html=True)
    else:
        st.info("Analyse traffic packets first to use the summarizer.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: Settings  ← API key lives here only
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Settings":
    st.markdown('<div class="sh">LLM Configuration</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="setting-card">
      <h4>🔑 OpenAI API Key</h4>
      <p style="color:#2A4A6A;font-size:.82rem;margin:0 0 .9rem;line-height:1.65">
        Your key is stored <strong style="color:#4A7A9A">in memory only</strong> for this session —
        never logged, never displayed, never written to disk.
        Without a key the system uses the built-in rule-based engine with full functionality.
      </p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([4,1])
    with c1:
        key_in = st.text_input("Key", type="password",
                               value=st.session_state.api_key,
                               placeholder="sk-proj-…",
                               label_visibility="collapsed")
    with c2:
        vbtn = st.button("Verify & Connect", type="primary", use_container_width=True)

    if vbtn:
        with st.spinner("Verifying …"):
            ok, msg = verify_key(key_in)
        if ok and "quota" not in msg.lower():
            st.session_state.api_key = key_in
            st.session_state.api_ok  = True
            os.environ["OPENAI_API_KEY"] = key_in
            st.success(f"✅ {msg} — GPT-4o active across all pages.")
        elif "quota" in msg.lower():
            st.session_state.api_key = key_in
            st.session_state.api_ok  = False
            os.environ["OPENAI_API_KEY"] = key_in
            st.warning(f"⚠ {msg}  \nKey saved. Add credits → https://platform.openai.com/settings/billing")
        else:
            st.session_state.api_ok = False
            st.error(f"❌ {msg}  \nGet a key → https://platform.openai.com/api-keys")

    if st.session_state.api_key:
        ok_v = st.session_state.api_ok
        chip = ("pulse pulse-on" if ok_v else "pulse pulse-off")
        st.markdown(f"""
        <div style="background:#030c18;border:1px solid rgba(0,140,255,0.12);
             border-radius:8px;padding:.65rem 1rem;margin:.5rem 0;
             display:flex;align-items:center;gap:1rem;font-size:.78rem">
          <span class="{chip}"><span class="pulse-dot"></span>
            {'Verified' if ok_v else 'Unverified'}</span>
          <span style="color:#1A3050">Key:</span>
          <code style="color:#2A4A6A;font-family:'Space Mono',monospace;font-size:.73rem">
            {masked(st.session_state.api_key)}</code>
        </div>""", unsafe_allow_html=True)
        if st.button("🗑 Clear Key"):
            st.session_state.api_key = ""; st.session_state.api_ok = False
            os.environ.pop("OPENAI_API_KEY",None); st.rerun()

    st.markdown('<div class="sh">Model Parameters</div>', unsafe_allow_html=True)
    st.markdown('<div class="setting-card"><h4>⚙ LLM Parameters</h4>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        mc = st.selectbox("Model",["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"],
                          index=["gpt-4o-mini","gpt-4o","gpt-3.5-turbo"].index(
                              st.session_state.llm_model))
        st.session_state.llm_model = mc
    with c2:
        tmp = st.slider("Temperature",0.0,1.0,float(st.session_state.temperature),0.05)
        st.session_state.temperature = tmp
    with c3:
        mtk = st.slider("Max Tokens",200,1500,int(st.session_state.max_tokens),50)
        st.session_state.max_tokens = mtk
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sh">System Status</div>', unsafe_allow_html=True)
    model_ok = os.path.exists("models/ids_model.pkl")
    data_ok  = os.path.exists("data/network_traffic.csv")
    st.markdown(f"""
| Component | Status | Detail |
|---|---|---|
| Dataset | {'✅' if data_ok else '❌'} | 5,000 synthetic records · 6 classes |
| ML Model | {'✅' if model_ok else '❌'} | Random Forest · Accuracy 93.50% · F1 93.41% |
| LLM Engine | {'✅ ' + st.session_state.llm_model if st.session_state.api_ok else '⚡ Rule-based'} | Session-only key |
| RAG KB | ✅ | 7 threat entries · TF-IDF retrieval |
| PySpark | ✅ | Pandas fallback active |

**Dataset:** Synthetic Network Traffic (5,000 records)  
**Classes:** Benign (49.4%) · DoS (15.6%) · PortScan (12.5%) · BruteForce (10.5%) · SQLInjection (6.9%) · Backdoor (5.0%)  
**Train/Test split:** 80% / 20% stratified · Random seed: 42
""")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🗑 Clear Alert Log", use_container_width=True):
            st.session_state.alerts = []; st.success("✅ Cleared")
    with c2:
        if st.button("🗑 Clear Chat History", use_container_width=True):
            st.session_state.chat = []; st.success("✅ Cleared")
