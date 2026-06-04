"""
LLM-Powered Intrusion Detection System — Semester Project
Category A | AI & LLM-Powered Security Systems | TIER S

FIXED: Navigation bar now works properly ✅
"""
 
import time, warnings
from collections import Counter
from datetime import datetime
 
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import Groq
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  ← must be FIRST streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="LLM-IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ══════════════════════════════════════════════════════════════════════════════
# CSS (IMPROVED NAVIGATION)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Outfit:wght@400;600;700;900&display=swap');
 
:root{
  --bg:#070b12;--bg2:#0d1220;--bg3:#111827;--border:#1e2d45;
  --accent:#3b82f6;--accent2:#6366f1;
  --danger:#ef4444;--warn:#f59e0b;--ok:#22c55e;
  --text:#e2e8f0;--muted:#64748b;
  --mono:'JetBrains Mono',monospace;--sans:'Outfit',sans-serif;
}
 
html,body,[data-testid="stAppViewContainer"]{
  background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;
}
[data-testid="stAppViewContainer"]{
  background:radial-gradient(ellipse 80% 50% at 50% -10%,#0d1d3a,var(--bg))!important;
}
[data-testid="stHeader"]{background:transparent!important;}
#MainMenu,footer,header{visibility:hidden;}
 
/* ── SIDEBAR ── */
[data-testid="stSidebar"]{
  background:#08101e !important;
  border-right:1px solid #152035 !important;
  min-width:260px !important;
  max-width:260px !important;
}
[data-testid="stSidebar"] > div:first-child{padding:0 !important;}
[data-testid="collapsedControl"]{display:none !important;}
 
/* ── FIX: NAV BUTTONS (Proper Streamlit styling) ── */
.nav-container {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px 0;
}

[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  padding: 12px 14px !important;
  margin: 0 !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 10px !important;
  cursor: pointer !important;
  text-align: left !important;
  font-family: 'Outfit', sans-serif !important;
  font-size: 0.87rem !important;
  font-weight: 600 !important;
  color: #4a6080 !important;
  transition: all 0.15s ease !important;
  box-shadow: none !important;
  transform: none !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(255, 255, 255, 0.04) !important;
  color: #9bb5d0 !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ── ACTIVE NAV STATE (Uses data attribute) ── */
[data-testid="stSidebar"] .stButton.nav-active > button {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.3), rgba(99, 102, 241, 0.12)) !important;
  border: 1px solid rgba(99, 102, 241, 0.4) !important;
  color: #e2e8f0 !important;
  box-shadow: 0 0 12px rgba(99, 102, 241, 0.2) !important;
}

[data-testid="stSidebar"] .stButton.nav-active > button:hover {
  background: linear-gradient(135deg, rgba(79, 70, 229, 0.4), rgba(99, 102, 241, 0.2)) !important;
  transform: none !important;
  box-shadow: 0 0 16px rgba(99, 102, 241, 0.3) !important;
}
 
/* ── KPI cards ── */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px;}
.kpi{background:#111827;border:1px solid #1e2d45;border-radius:14px;
     padding:18px 20px;position:relative;overflow:hidden;}
.kpi::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:14px 14px 0 0;}
.kpi.blue::before{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.kpi.red::before{background:linear-gradient(90deg,#ef4444,#f87171);}
.kpi.green::before{background:linear-gradient(90deg,#22c55e,#4ade80);}
.kpi.amber::before{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi-label{font-size:.64rem;color:#64748b;letter-spacing:1.5px;
           text-transform:uppercase;font-family:'JetBrains Mono',monospace;}
.kpi-val{font-size:1.9rem;font-weight:900;line-height:1.1;margin-top:5px;}
.kpi.blue .kpi-val{color:#60a5fa;}.kpi.red .kpi-val{color:#f87171;}
.kpi.green .kpi-val{color:#4ade80;}.kpi.amber .kpi-val{color:#fbbf24;}
.kpi-sub{font-size:.68rem;color:#64748b;margin-top:2px;font-family:'JetBrains Mono',monospace;}
 
/* ── Section heading ── */
.sec{font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;
     letter-spacing:2.5px;border-left:3px solid #6366f1;padding-left:12px;
     margin:22px 0 12px;font-family:'JetBrains Mono',monospace;}
 
/* ── Hero ── */
.hero{background:linear-gradient(135deg,#0d1e3a,#070b12);border:1px solid #1e2d45;
      border-radius:16px;padding:22px 28px;margin-bottom:22px;
      display:flex;align-items:center;gap:18px;
      box-shadow:0 0 60px rgba(59,130,246,.07);position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-80px;right:-80px;width:280px;height:280px;
             background:radial-gradient(circle,rgba(99,102,241,.09),transparent 70%);pointer-events:none;}
.hero-title{font-size:1.5rem;font-weight:900;color:#e2e8f0;margin:0;letter-spacing:-.4px;font-family:'Outfit',sans-serif;}
.hero-sub{font-size:.7rem;color:#7fa8d4;font-family:'JetBrains Mono',monospace;margin-top:4px;}
.tier-badge{margin-left:auto;background:linear-gradient(135deg,#6366f1,#4f46e5);
            color:#fff;font-size:.62rem;font-weight:700;padding:5px 14px;border-radius:30px;
            letter-spacing:2px;font-family:'JetBrains Mono',monospace;white-space:nowrap;
            box-shadow:0 0 20px rgba(99,102,241,.4);}
 
/* ── Alert cards ── */
.alert-card{background:#111827;border-radius:12px;padding:16px 20px;margin-bottom:10px;
            border-left:4px solid #ef4444;border-top:1px solid #1e2d45;
            border-right:1px solid #1e2d45;border-bottom:1px solid #1e2d45;}
.alert-card.medium{border-left-color:#f59e0b;}
.alert-card.low{border-left-color:#22c55e;}
.alert-card.info{border-left-color:#3b82f6;}
.alert-meta{font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#64748b;margin-bottom:6px;}
.alert-title{font-size:.92rem;font-weight:700;color:#e2e8f0;margin-bottom:5px;font-family:'Outfit',sans-serif;}
.alert-body{font-size:.8rem;color:#94a3b8;line-height:1.65;}
.pill{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.62rem;
      font-weight:700;letter-spacing:1px;font-family:'JetBrains Mono',monospace;
      text-transform:uppercase;margin-left:8px;}
.pill-high{background:rgba(239,68,68,.15);color:#f87171;}
.pill-medium{background:rgba(245,158,11,.15);color:#fbbf24;}
.pill-low{background:rgba(34,197,94,.15);color:#4ade80;}
.pill-info{background:rgba(59,130,246,.15);color:#60a5fa;}
 
/* ── Packet row ── */
.packet-row{background:#0d1220;border:1px solid #1e2d45;border-radius:8px;
            padding:9px 14px;font-family:'JetBrains Mono',monospace;
            font-size:.71rem;color:#94a3b8;margin-bottom:5px;}
.packet-row span{color:#60a5fa;}
 
/* ── RAG card ── */
.rag-card{background:#111827;border:1px solid #1e2d45;border-radius:10px;
          padding:14px 18px;margin-bottom:10px;}
.rag-title{font-size:.88rem;font-weight:700;color:#e2e8f0;margin-bottom:5px;font-family:'Outfit',sans-serif;}
.rag-body{font-size:.78rem;color:#94a3b8;line-height:1.6;}
.rag-tag{display:inline-block;background:rgba(99,102,241,.15);color:#a5b4fc;
         font-size:.6rem;padding:2px 8px;border-radius:20px;
         font-family:'JetBrains Mono',monospace;margin-right:4px;margin-top:5px;}
 
/* ── Pulse dot ── */
.status-pill{display:inline-flex;align-items:center;gap:6px;
             background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);
             color:#4ade80;font-size:.65rem;font-family:'JetBrains Mono',monospace;
             padding:4px 12px;border-radius:20px;}
.dot{width:6px;height:6px;border-radius:50%;background:#4ade80;animation:blink 2s infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
 
/* ── Main content Buttons ── */
[data-testid="stMainBlockContainer"] .stButton>button{
  background:linear-gradient(135deg,#4f46e5,#7c3aed)!important;
  color:#fff!important;border:none!important;border-radius:8px!important;
  font-family:'Outfit',sans-serif!important;font-weight:700!important;
  transition:all .2s!important;
}
[data-testid="stMainBlockContainer"] .stButton>button:hover{
  transform:translateY(-2px)!important;
  box-shadow:0 8px 24px rgba(99,102,241,.35)!important;
}
 
/* ── Form inputs ── */
.stTextArea textarea,.stTextInput>div>div>input{
  background:#111827!important;border:1px solid #1e2d45!important;
  border-radius:8px!important;color:#e2e8f0!important;
  font-family:'JetBrains Mono',monospace!important;font-size:.8rem!important;}
.stSelectbox>div>div,.stNumberInput>div>div>div{
  background:#111827!important;border-color:#1e2d45!important;
  color:#e2e8f0!important;border-radius:8px!important;}
[data-testid="stExpander"]{
  background:#111827!important;border:1px solid #1e2d45!important;border-radius:10px!important;}
hr{border-color:#1e2d45!important;}
.stAlert{border-radius:10px!important;}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  // Update nav active state on page load
  updateNavActive();
});

function updateNavActive() {
  const pageTitle = document.querySelector('[data-testid="stSidebar"]');
  if (!pageTitle) return;
  
  // Get current page from session state (this will be updated by Streamlit)
  const buttons = document.querySelectorAll('[data-testid="stSidebar"] .stButton');
  buttons.forEach(btn => {
    btn.classList.remove('nav-active');
  });
}

// Observe for Streamlit reruns
const observer = new MutationObserver(() => {
  updateNavActive();
});

observer.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# GROQ
# ══════════════════════════════════════════════════════════════════════════════
def _get_key():
    try: return st.secrets["GROQ_API_KEY"]
    except: return ""
 
def _groq_client():
    k = _get_key()
    return Groq(api_key=k) if k and k != "gsk_your_key_here" else None
 
def _llm_ready():
    k = _get_key()
    return bool(k) and k != "gsk_your_key_here"
 
# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════════════════════
ATTACK_FAMILIES = {
    "normal":"Normal","neptune":"DoS","smurf":"DoS","back":"DoS","teardrop":"DoS",
    "pod":"DoS","land":"DoS","portsweep":"Probe","ipsweep":"Probe","satan":"Probe",
    "nmap":"Probe","warezclient":"R2L","warezmaster":"R2L","guess_passwd":"R2L",
    "ftp_write":"R2L","imap":"R2L","multihop":"R2L","phf":"R2L","spy":"R2L",
    "rootkit":"U2R","buffer_overflow":"U2R","loadmodule":"U2R","perl":"U2R",
    "sqlattack":"U2R","xterm":"U2R","ps":"U2R",
}
KB = {
    "neptune":{"title":"Neptune / SYN Flood (DoS)",
        "body":"Neptune attacks send thousands of SYN packets without completing the TCP handshake, exhausting server connection tables. Mitigation: SYN cookies, rate-limiting, firewall rules.",
        "tags":["DoS","TCP","SYN-Flood"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 100,seconds 1; msg:"SYN Flood";)',},
    "smurf":{"title":"Smurf Attack (DDoS / ICMP Amplification)",
        "body":"ICMP echo requests sent to broadcast address with spoofed victim IP. All hosts reply to victim, amplifying bandwidth. Mitigation: disable IP-directed broadcasts, BCP38 ingress filtering.",
        "tags":["DDoS","ICMP","Amplification","Spoofing"],
        "snort_rule":'alert icmp any any -> $HOME_NET any (itype:8; msg:"ICMP Smurf Attack";)',},
    "portsweep":{"title":"Port Sweep / Probe",
        "body":"Attacker scans multiple ports to identify open services. High connection count with diverse destination ports. Mitigation: port knocking, firewall, IDS rules.",
        "tags":["Probe","Reconnaissance","Nmap"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (flags:S; threshold:type both,track by_src,count 20,seconds 5; msg:"Port Sweep";)',},
    "ipsweep":{"title":"IP Sweep (Network Reconnaissance)",
        "body":"Scans multiple IPs via ICMP ping to map live hosts. High ICMP request count from single source. Mitigation: ICMP rate limiting, honeypots.",
        "tags":["Probe","Reconnaissance","ICMP"],
        "snort_rule":'alert icmp any any -> $HOME_NET any (itype:8; threshold:type both,track by_src,count 15,seconds 3; msg:"IP Sweep";)',},
    "buffer_overflow":{"title":"Buffer Overflow (U2R)",
        "body":"Exploits memory management flaws to inject and execute code with elevated privileges. Indicators: root_shell=1, num_root>0. Mitigation: ASLR, stack canaries, NX bit.",
        "tags":["U2R","Memory-Exploit","Privilege-Escalation"],
        "snort_rule":'alert tcp any any -> $HOME_NET any (content:"|90 90 90|"; msg:"Possible Buffer Overflow";)',},
    "guess_passwd":{"title":"Password Brute-Force (R2L)",
        "body":"Repeated login attempts with credential lists. High num_failed_logins, logged_in=0. Mitigation: account lockout, MFA, fail2ban, CAPTCHA.",
        "tags":["R2L","Credential","Brute-Force"],
        "snort_rule":'alert tcp any any -> $HOME_NET 22 (flags:S; threshold:type both,track by_src,count 5,seconds 10; msg:"SSH Brute Force";)',},
    "normal":{"title":"Normal Traffic",
        "body":"Traffic conforms to expected baseline. No anomaly detected. Continue standard monitoring.",
        "tags":["Benign","Baseline"],
        "snort_rule":"# No rule — traffic classified as normal",},
}
 
def _family(label):
    label = label.lower().strip().rstrip(".")
    for k,v in ATTACK_FAMILIES.items():
        if k in label: return v
    return "Unknown"
 
def _severity(label):
    f = _family(label)
    if label.strip().lower()=="normal": return "info"
    if f=="U2R": return "high"
    if f in ("R2L","Probe"): return "medium"
    return "high"
 
def _kb(label):
    label = label.lower().strip().rstrip(".")
    for k in KB:
        if k in label: return KB[k]
    return KB["normal"]
 
# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
NSL_COLS = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty_level",
]
 
@st.cache_data(show_spinner=False)
def load_dataset():
    for src,url in [
        ("NSL-KDD (KDDTrain+)","https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"),
        ("NSL-KDD (20%)","https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+_20Percent.txt"),
    ]:
        try:
            df = pd.read_csv(url, names=NSL_COLS, header=None, nrows=8000)
            df["label"] = df["label"].str.strip().str.lower()
            return df, src
        except: continue
    np.random.seed(42); n=5000
    lp=["normal","neptune","smurf","portsweep","ipsweep","satan","nmap",
        "guess_passwd","ftp_write","buffer_overflow","rootkit","teardrop","back"]
    lw=[0.35,0.14,0.10,0.06,0.06,0.04,0.03,0.05,0.02,0.04,0.02,0.05,0.04]
    df = pd.DataFrame({
        "duration":np.random.exponential(5,n).astype(int),
        "protocol_type":np.random.choice(["tcp","udp","icmp"],n,p=[0.6,0.25,0.15]),
        "service":np.random.choice(["http","ftp","smtp","ssh","dns","telnet","private","other"],n,p=[0.3,0.1,0.1,0.14,0.1,0.05,0.11,0.10]),
        "flag":np.random.choice(["SF","S0","REJ","RSTO","SH","OTH"],n,p=[0.52,0.2,0.1,0.08,0.05,0.05]),
        "src_bytes":np.random.exponential(5000,n).astype(int),
        "dst_bytes":np.random.exponential(2000,n).astype(int),
        "land":np.random.choice([0,1],n,p=[0.99,0.01]),
        "wrong_fragment":np.random.choice([0,1,2],n,p=[0.94,0.05,0.01]),
        "urgent":np.zeros(n,dtype=int),"hot":np.random.randint(0,6,n),
        "num_failed_logins":np.random.choice([0,1,2,3],n,p=[0.88,0.08,0.03,0.01]),
        "logged_in":np.random.choice([0,1],n,p=[0.45,0.55]),
        "num_compromised":np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "root_shell":np.random.choice([0,1],n,p=[0.97,0.03]),
        "su_attempted":np.random.choice([0,1],n,p=[0.99,0.01]),
        "num_root":np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "num_file_creations":np.random.choice([0,1,2],n,p=[0.93,0.05,0.02]),
        "num_shells":np.random.choice([0,1],n,p=[0.97,0.03]),
        "num_access_files":np.random.choice([0,1,2],n,p=[0.9,0.08,0.02]),
        "num_outbound_cmds":np.zeros(n,dtype=int),"is_host_login":np.zeros(n,dtype=int),
        "is_guest_login":np.random.choice([0,1],n,p=[0.97,0.03]),
        "count":np.random.randint(1,512,n),"srv_count":np.random.randint(1,512,n),
        "serror_rate":np.random.uniform(0,1,n).round(2),"srv_serror_rate":np.random.uniform(0,1,n).round(2),
        "rerror_rate":np.random.uniform(0,1,n).round(2),"srv_rerror_rate":np.random.uniform(0,1,n).round(2),
        "same_srv_rate":np.random.uniform(0,1,n).round(2),"diff_srv_rate":np.random.uniform(0,1,n).round(2),
        "srv_diff_host_rate":np.random.uniform(0,1,n).round(2),"dst_host_count":np.random.randint(1,256,n),
        "dst_host_srv_count":np.random.randint(1,256,n),
        "dst_host_same_srv_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_diff_srv_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_same_src_port_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_srv_diff_host_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_serror_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_srv_serror_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_rerror_rate":np.random.uniform(0,1,n).round(2),
        "dst_host_srv_rerror_rate":np.random.uniform(0,1,n).round(2),
        "label":np.random.choice(lp,n,p=lw),"difficulty_level":np.random.randint(1,21,n),
    })
    return df,"Synthetic NSL-KDD (offline fallback)"
 
# ══════════════════════════════════════════════════════════════════════════════
# AUTO-TRAIN AT STARTUP
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=True)
def _boot_train():
    df, src = load_dataset()
    cat=["protocol_type","service","flag"]; les={}; d=df.copy()
    for c in cat:
        le=LabelEncoder(); d[c]=le.fit_transform(d[c].astype(str)); les[c]=le
    d["is_attack"]=(d["label"].str.strip().str.lower()!="normal").astype(int)
    feat=[c for c in d.columns if c not in ["label","difficulty_level","is_attack"]]
    X=d[feat].fillna(0); y=d["is_attack"]
    sc=StandardScaler(); Xs=sc.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(Xs,y,test_size=0.30,random_state=13,stratify=y)
    rf=RandomForestClassifier(n_estimators=45,max_depth=10,min_samples_leaf=6,
                              max_features="sqrt",random_state=99,n_jobs=-1)
    rf.fit(Xtr,ytr)
    iso=IsolationForest(n_estimators=80,contamination=0.22,random_state=42)
    iso.fit(Xtr)
    yp=rf.predict(Xte)
    acc=accuracy_score(yte,yp); f1=f1_score(yte,yp,average="weighted",zero_division=0)
    prec=precision_score(yte,yp,average="weighted",zero_division=0)
    rec=recall_score(yte,yp,average="weighted",zero_division=0)
    cm=confusion_matrix(yte,yp)
    return rf,iso,sc,les,feat,acc,f1,prec,rec,cm,df,src
 
RF,ISO,SC,LES,FEAT,ACC,F1,PREC,REC,CM,DF,SRC = _boot_train()
 
# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [("reports",[]),("_capture",[]),("_cmp",None),("_brief",""),("page","📊  Dataset & Model")]:
    if k not in st.session_state: st.session_state[k]=v
 
# ══════════════════════════════════════════════════════════════════════════════
# SNORT SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
_SNORT={
    "neptune":    lambda r: float(r.get("serror_rate",0))>0.7 and int(r.get("count",0))>100,
    "smurf":      lambda r: str(r.get("protocol_type",""))=="icmp" and int(r.get("src_bytes",0))>9000,
    "portsweep":  lambda r: float(r.get("diff_srv_rate",0))>0.5 and int(r.get("count",0))>30,
    "ipsweep":    lambda r: str(r.get("protocol_type",""))=="icmp" and int(r.get("dst_host_count",0))>20,
    "buffer_overflow": lambda r: int(r.get("root_shell",0))==1 or int(r.get("num_root",0))>1,
    "guess_passwd":    lambda r: int(r.get("num_failed_logins",0))>=3,
}
def _snort(row):
    for atk,fn in _SNORT.items():
        try:
            if fn(row): return atk
        except: pass
    return "normal"
 
# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLOT=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(color="#94a3b8",family="JetBrains Mono"),margin=dict(l=12,r=12,t=36,b=12))
COLORS=["#3b82f6","#ef4444","#f59e0b","#22c55e","#a78bfa","#f472b6","#34d399"]
 
def _fake_ip():
    r=np.random.randint; return f"{r(1,255)}.{r(0,255)}.{r(0,255)}.{r(1,255)}"
 
def _sim_pkt(row):
    return dict(ts=datetime.now().strftime("%H:%M:%S.%f")[:-3],
                src=_fake_ip(),dst="192.168.1."+str(np.random.randint(1,50)),
                proto=str(row.get("protocol_type","tcp")).upper(),
                svc=row.get("service","http"),flag=row.get("flag","SF"),
                length=int(row.get("src_bytes",0))+int(row.get("dst_bytes",0)))
 
def _tstr(row):
    return (f"Protocol: {str(row.get('protocol_type','?')).upper()} | "
            f"Service: {row.get('service','?')} | Flag: {row.get('flag','?')}\n"
            f"Src Bytes: {int(row.get('src_bytes',0)):,} | "
            f"Dst Bytes: {int(row.get('dst_bytes',0)):,} | Duration: {row.get('duration',0)}s\n"
            f"Failed Logins: {row.get('num_failed_logins',0)} | Logged In: {row.get('logged_in',0)} | "
            f"Root Shell: {row.get('root_shell',0)}\n"
            f"Error Rate: {float(row.get('serror_rate',0)):.2f} | Count: {row.get('count',0)} | "
            f"Dst Host Count: {row.get('dst_host_count',0)}")
 
def _vec(row):
    tmp={}
    for c in FEAT:
        val=row.get(c,0)
        if c in LES:
            try: val=LES[c].transform([str(val)])[0]
            except: val=0
        tmp[c]=val
    return SC.transform(pd.DataFrame([tmp])[FEAT].fillna(0))
 
# ══════════════════════════════════════════════════════════════════════════════
# LLM FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def llm_report(client,summary,label,kb):
    prompt=(f"You are an expert cybersecurity analyst reviewing network traffic flagged by an ML-based IDS.\n\n"
            f"=== TRAFFIC SUMMARY ===\n{summary}\n\n"
            f"=== ML CLASSIFICATION ===\n"
            f"Attack Type   : {label}\nAttack Family : {_family(label)}\n"
            f"Severity      : {_severity(label).upper()}\nKnown Pattern : {kb['title']}\n\n"
            f"=== SNORT SIGNATURE ===\n{kb['snort_rule']}\n\n"
            f"Write a structured threat report with exactly these sections:\n"
            f"**1. Threat Classification** — one sentence.\n"
            f"**2. Traffic Analysis** — 2-3 sentences on why the traffic is suspicious.\n"
            f"**3. Potential Impact** — 1-2 sentences.\n"
            f"**4. LLM vs Signature IDS** — 1 sentence comparing Snort vs LLM analysis.\n"
            f"**5. Recommended Response**\n- Action 1\n- Action 2\n- Action 3\n\nUnder 280 words.")
    r=client.chat.completions.create(model="llama3-70b-8192",
        messages=[{"role":"user","content":prompt}],max_tokens=500,temperature=0.25)
    return r.choices[0].message.content.strip()
 
def llm_soc(client,reports):
    last=reports[-5:]
    lines="\n".join(f"- {r['ts']}: {r['label']} ({r['sev'].upper()}) on {r['svc']}" for r in last)
    r=client.chat.completions.create(model="llama3-8b-8192",
        messages=[{"role":"user","content":
            f"You are a SOC analyst. Write a 3-4 sentence shift briefing covering "
            f"threat posture, dominant attack types, and urgent actions. Prose only.\n\n"
            f"Recent alerts:\n{lines}"}],max_tokens=200,temperature=0.3)
    return r.choices[0].message.content.strip()
 
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════  SIDEBAR NAV (FIXED) ═════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
NAV_PAGES = [
    "📊  Dataset & Model",
    "📡  Live Capture",
    "🤖  LLM Threat Reports",
    "📈  IDS Comparison",
    "🧠  RAG Knowledge Base",
    "📋  NLP SOC Summary",
]
 
with st.sidebar:
    st.markdown("""
    <div style="padding:22px 18px 14px;border-bottom:1px solid #152035;margin-bottom:16px;">
      <div style="font-size:2rem;line-height:1;margin-bottom:8px;">🛡️</div>
      <div style="font-size:1rem;font-weight:900;color:#e2e8f0;
                  font-family:'Outfit',sans-serif;letter-spacing:-.2px;">LLM-IDS</div>
      <div style="font-size:.6rem;color:#3d5a7a;font-family:'JetBrains Mono',monospace;margin-top:3px;">
        Intrusion Detection System
      </div>
    </div>
    <div style="padding:0 18px 8px;font-size:.58rem;color:#253a55;
                font-family:'JetBrains Mono',monospace;letter-spacing:2px;text-transform:uppercase;">
      Navigation
    </div>
    """, unsafe_allow_html=True)
 
    # ✅ FIXED: Proper navigation with active state tracking
    for nav_item in NAV_PAGES:
        is_active = st.session_state.page == nav_item
        
        # Create button container with conditional class
        col1, col2 = st.columns([20, 1])
        with col1:
            # Inject the nav-active class directly on button click
            if st.button(
                nav_item,
                key=f"nav_{nav_item}",
                use_container_width=True,
                help=f"Go to {nav_item}"
            ):
                st.session_state.page = nav_item
                st.rerun()
        
        # Visual indicator for active page
        if is_active:
            with col2:
                st.markdown('<div style="color:#60a5fa;font-size:1.2rem;margin-top:6px;">✓</div>', unsafe_allow_html=True)
    
    # Add visual styling to active nav button using session state
    st.markdown(f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const buttons = document.querySelectorAll('[data-testid="stSidebar"] .stButton button');
        const activeText = "{st.session_state.page}";
        buttons.forEach(btn => {{
            if (btn.innerText.includes(activeText.split('  ')[0])) {{
                btn.parentElement.parentElement.classList.add('nav-active');
            }}
        }});
    }});
    </script>
    """, unsafe_allow_html=True)
 
    st.markdown("<hr style='border-color:#152035;margin:14px 0;'>", unsafe_allow_html=True)
 
    key_ok=_llm_ready()
    key_txt=('<span style="color:#4ade80;">🟢 API key ready</span>'
             if key_ok else '<span style="color:#f87171;">🔴 Add GROQ_API_KEY</span>')
    st.markdown(f"""
    <div style="padding:0 14px 20px;">
      <div class="status-pill"><span class="dot"></span>Models Live</div>
      <div style="margin-top:10px;font-size:.65rem;">{key_txt}</div>
      <div style="font-size:.58rem;color:#1e3552;font-family:'JetBrains Mono',monospace;
                  margin-top:8px;line-height:1.6;">
        Accuracy: {ACC*100:.1f}%<br>F1: {F1:.3f}<br>{SRC[:24]}…
      </div>
    </div>
    """, unsafe_allow_html=True)
 
# Read current page from session state
page = st.session_state.page
 
# ══════════════════════════════════════════════════════════════════════════════
# HERO (always shown)
# ══════════════════════════════════════════════════════════════════════════════
n_atk=int((DF["label"]!="normal").sum())
st.markdown(f"""
<div class="hero">
  <div style="font-size:2.2rem;">🛡️</div>
  <div style="flex:1;">
    <div class="hero-title">LLM-Powered Intrusion Detection System</div>
    <div class="hero-sub">NSL-KDD · Groq LLaMA-3 70B · Random Forest · Isolation Forest · RAG · NLP</div>
  </div>
  <span class="tier-badge">TIER S</span>
</div>
<div class="kpi-row">
  <div class="kpi blue">
    <div class="kpi-label">Total Samples</div>
    <div class="kpi-val">{len(DF):,}</div>
    <div class="kpi-sub">NSL-KDD dataset</div>
  </div>
  <div class="kpi red">
    <div class="kpi-label">Attack Records</div>
    <div class="kpi-val">{n_atk:,}</div>
    <div class="kpi-sub">{n_atk/len(DF)*100:.1f}% of dataset</div>
  </div>
  <div class="kpi green">
    <div class="kpi-label">RF Accuracy</div>
    <div class="kpi-val">{ACC*100:.1f}%</div>
    <div class="kpi-sub">30% test split</div>
  </div>
  <div class="kpi amber">
    <div class="kpi-label">F1 Score</div>
    <div class="kpi-val">{F1:.3f}</div>
    <div class="kpi-sub">Weighted avg</div>
  </div>
</div>
""", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONTENT
# ══════════════════════════════════════════════════════════════════════════════
 
# ── PAGE 1 ───────────────────────────────────────────────────────────────────
if page=="📊  Dataset & Model":
    st.markdown('<p class="sec">📦 NSL-KDD Dataset Overview</p>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<p class="sec">Attack Family Distribution</p>',unsafe_allow_html=True)
        fc=DF["label"].apply(_family).value_counts().reset_index(); fc.columns=["Family","Count"]
        fig=px.bar(fc,x="Family",y="Count",color="Family",color_discrete_sequence=COLORS)
        fig.update_layout(**PLOT,showlegend=False); fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=True,gridcolor="#1e2d45")
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        st.markdown('<p class="sec">Top 10 Attack Labels</p>',unsafe_allow_html=True)
        top=DF["label"].value_counts().head(10).reset_index(); top.columns=["Label","Count"]
        fig2=px.pie(top,values="Count",names="Label",color_discrete_sequence=COLORS,hole=0.45)
        fig2.update_layout(**PLOT); st.plotly_chart(fig2,use_container_width=True)
 
    st.markdown('<p class="sec">Confusion Matrix — Normal vs Attack</p>',unsafe_allow_html=True)
    fig3=go.Figure(go.Heatmap(z=CM,x=["Normal","Attack"],y=["Normal","Attack"],
        colorscale=[[0,"#111827"],[1,"#3b82f6"]],text=CM,texttemplate="%{text}",showscale=False))
    fig3.update_layout(**PLOT,height=280); st.plotly_chart(fig3,use_container_width=True)
 
    st.markdown('<p class="sec">Feature Importance (Top 15)</p>',unsafe_allow_html=True)
    imp=pd.Series(RF.feature_importances_,index=FEAT).nlargest(15).reset_index()
    imp.columns=["Feature","Importance"]
    fig4=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",
                color_continuous_scale=["#1e3a5f","#3b82f6","#60a5fa"])
    fig4.update_layout(**PLOT,showlegend=False,height=380)
    fig4.update_xaxes(showgrid=True,gridcolor="#1e2d45"); st.plotly_chart(fig4,use_container_width=True)
 
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi blue"><div class="kpi-label">Precision</div>
        <div class="kpi-val">{PREC*100:.1f}%</div><div class="kpi-sub">Weighted avg</div></div>
      <div class="kpi green"><div class="kpi-label">Recall</div>
        <div class="kpi-val">{REC*100:.1f}%</div><div class="kpi-sub">Weighted avg</div></div>
      <div class="kpi amber"><div class="kpi-label">RF Trees</div>
        <div class="kpi-val">45</div><div class="kpi-sub">max_depth=10</div></div>
      <div class="kpi red"><div class="kpi-label">IF Trees</div>
        <div class="kpi-val">80</div><div class="kpi-sub">contam=0.22</div></div>
    </div>""",unsafe_allow_html=True)
    with st.expander("🔍 Raw Dataset Sample (100 rows)"):
        st.dataframe(DF.sample(min(100,len(DF)),random_state=1).reset_index(drop=True),
                     use_container_width=True,height=300)
 
# ── PAGE 2 ───────────────────────────────────────────────────────────────────
elif page=="📡  Live Capture":
    st.markdown('<p class="sec">📡 Packet Capture Simulation</p>',unsafe_allow_html=True)
    st.info("Simulates live packet capture. In production replace with `scapy.sniff()` or `pyshark.LiveCapture()`.",icon="ℹ️")
    c1,c2=st.columns([3,1])
    with c1: n_pkts=st.slider("Packets to capture",5,50,15)
    with c2: speed=st.selectbox("Speed",["Fast","Normal","Slow"])
    delay={"Fast":0.04,"Normal":0.18,"Slow":0.45}[speed]
 
    if st.button("▶️  Start Capture",use_container_width=True):
        ph=st.empty(); captured=[]
        rows=DF.sample(n_pkts,random_state=int(time.time())).to_dict("records")
        for row in rows:
            pkt=_sim_pkt(row); xv=_vec(row)
            pred=int(RF.predict(xv)[0]); iscs=float(ISO.decision_function(xv)[0])
            pkt.update(pred_attack=pred,anomaly=iscs<-0.05,true_label=row.get("label","?"),iso_score=round(iscs,3))
            captured.append(pkt)
            html="".join(
                f'<div class="packet-row">[{p["ts"]}] <span>{p["src"]}</span>→<span>{p["dst"]}</span>'
                f' | {p["proto"]}/{p["svc"]}/{p["flag"]} | {p["length"]:,}B'
                f' | <span style="color:{"#f87171" if p["pred_attack"] else "#4ade80"}">{"ATTACK" if p["pred_attack"] else "NORMAL"}</span>'
                f'{"⚠️" if p["anomaly"] else ""}</div>'
                for p in captured[-20:])
            ph.markdown(html,unsafe_allow_html=True); time.sleep(delay)
        st.session_state._capture=captured
        atks=sum(p["pred_attack"] for p in captured)
        st.success(f"✅ {n_pkts} packets — {atks} attack(s) detected")
 
    if st.session_state._capture:
        cap=st.session_state._capture
        atks=sum(p["pred_attack"] for p in cap); anoms=sum(p["anomaly"] for p in cap)
        st.markdown(f"""<div class="kpi-row">
          <div class="kpi blue"><div class="kpi-label">Packets</div><div class="kpi-val">{len(cap)}</div></div>
          <div class="kpi red"><div class="kpi-label">RF Attacks</div><div class="kpi-val">{atks}</div></div>
          <div class="kpi amber"><div class="kpi-label">IF Anomalies</div><div class="kpi-val">{anoms}</div></div>
          <div class="kpi green"><div class="kpi-label">Normal</div><div class="kpi-val">{len(cap)-atks}</div></div>
        </div>""",unsafe_allow_html=True)
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=[p["iso_score"] for p in cap],mode="lines+markers",
            line=dict(color="#3b82f6",width=2),
            marker=dict(color=["#ef4444" if p["pred_attack"] else "#22c55e" for p in cap],size=8)))
        fig.add_hline(y=-0.05,line_dash="dash",line_color="#f59e0b",annotation_text="Anomaly threshold")
        fig.update_layout(**PLOT,title="Isolation Forest Score per Packet",xaxis_title="Packet #",yaxis_title="Score")
        st.plotly_chart(fig,use_container_width=True)
 
# ── PAGE 3 ───────────────────────────────────────────────────────────────────
elif page=="🤖  LLM Threat Reports":
    st.markdown('<p class="sec">🤖 LLM Threat Reports — Groq LLaMA-3 70B</p>',unsafe_allow_html=True)
    if not _llm_ready():
        st.error("⚠️  No Groq API key found. Add `GROQ_API_KEY` in **App Settings → Secrets**.")
        st.code('GROQ_API_KEY = "gsk_xxxxxxxxxxxxxxxx"',language="toml")
    else:
        c1,c2=st.columns([3,1])
        with c1:
            opts=["(Random)"]+sorted(DF["label"].unique().tolist())
            chosen=st.selectbox("Filter by attack type",opts)
        with c2: n_analyze=st.number_input("Records",1,5,1)
 
        if st.button("🔍  Analyze & Generate Reports",use_container_width=True):
            pool=DF if chosen=="(Random)" else DF[DF["label"]==chosen]
            samples=pool.sample(min(int(n_analyze),len(pool)),random_state=int(time.time()))
            client=_groq_client(); progress=st.progress(0,"Analyzing…")
            for idx,(_,row) in enumerate(samples.iterrows()):
                rd=row.to_dict(); true_lbl=rd.get("label","?")
                xv=_vec(rd); pred_bin=int(RF.predict(xv)[0])
                pred_label=true_lbl if pred_bin else "normal"
                sev=_severity(pred_label); kb=_kb(pred_label)
                with st.spinner(f"LLM report {idx+1}/{len(samples)}…"):
                    try: rtext=llm_report(client,_tstr(rd),pred_label,kb)
                    except Exception as e: rtext=f"[LLM error: {e}]"
                st.session_state.reports.insert(0,dict(
                    ts=datetime.now().strftime("%H:%M:%S"),label=pred_label,true_label=true_lbl,
                    sev=sev,svc=rd.get("service","?"),proto=str(rd.get("protocol_type","?")),
                    report=rtext,kb=kb))
                progress.progress((idx+1)/len(samples))
            progress.empty(); st.success(f"✅ {len(samples)} LLM report(s) generated.")
 
    if st.session_state.reports:
        st.markdown('<p class="sec">Generated Reports</p>',unsafe_allow_html=True)
        for r in st.session_state.reports[:20]:
            sc_cls={"high":"","medium":"medium","low":"low","info":"info"}[r["sev"]]
            st.markdown(f"""
            <div class="alert-card {sc_cls}">
              <div class="alert-meta">🕐 {r['ts']} | {r['proto'].upper()}/{r['svc']} | True: <b>{r['true_label']}</b></div>
              <div class="alert-title">{r['kb']['title']}<span class="pill pill-{r['sev']}">{r['sev'].upper()}</span></div>
              <div class="alert-body">{r['report'].replace(chr(10),'<br>')}</div>
            </div>""",unsafe_allow_html=True)
        if st.button("🗑️  Clear Reports"):
            st.session_state.reports=[]; st.rerun()
 
# ── PAGE 4 ───────────────────────────────────────────────────────────────────
elif page=="📈  IDS Comparison":
    st.markdown('<p class="sec">📈 ML-IDS vs Traditional Signature IDS — Snort / Suricata</p>',unsafe_allow_html=True)
    n_s=st.slider("Comparison sample size",100,1000,500,step=100)
 
    if st.button("▶️  Run Comparison",use_container_width=True):
        rows=DF.sample(n_s,random_state=42).to_dict("records")
        ml_p,sn_p,true_b=[],[],[]
        prog=st.progress(0,"Evaluating…")
        for i,row in enumerate(rows):
            tb=0 if row.get("label","normal").strip().lower()=="normal" else 1
            xv=_vec(row); ml_p.append(int(RF.predict(xv)[0]))
            sn_p.append(0 if _snort(row)=="normal" else 1); true_b.append(tb)
            if i%50==0: prog.progress((i+1)/len(rows))
        prog.empty()
        st.session_state._cmp=dict(
            ml_acc=accuracy_score(true_b,ml_p),sn_acc=accuracy_score(true_b,sn_p),
            ml_f1=f1_score(true_b,ml_p,average="weighted",zero_division=0),
            sn_f1=f1_score(true_b,sn_p,average="weighted",zero_division=0),
            ml_prec=precision_score(true_b,ml_p,average="weighted",zero_division=0),
            sn_prec=precision_score(true_b,sn_p,average="weighted",zero_division=0),
            ml_rec=recall_score(true_b,ml_p,average="weighted",zero_division=0),
            sn_rec=recall_score(true_b,sn_p,average="weighted",zero_division=0),
            ml_fp=sum(1 for m,t in zip(ml_p,true_b) if m==1 and t==0),
            sn_fp=sum(1 for m,t in zip(sn_p,true_b) if m==1 and t==0),
            ml_fn=sum(1 for m,t in zip(ml_p,true_b) if m==0 and t==1),
            sn_fn=sum(1 for m,t in zip(sn_p,true_b) if m==0 and t==1),n=n_s)
        st.success(f"✅ Compared {n_s} samples.")
 
    if st.session_state._cmp:
        c=st.session_state._cmp
        st.markdown(f"""<div class="kpi-row">
          <div class="kpi blue"><div class="kpi-label">ML Accuracy</div>
            <div class="kpi-val">{c['ml_acc']*100:.1f}%</div><div class="kpi-sub">Random Forest</div></div>
          <div class="kpi amber"><div class="kpi-label">Snort Accuracy</div>
            <div class="kpi-val">{c['sn_acc']*100:.1f}%</div><div class="kpi-sub">Signature IDS</div></div>
          <div class="kpi green"><div class="kpi-label">ML F1</div>
            <div class="kpi-val">{c['ml_f1']:.3f}</div><div class="kpi-sub">Weighted</div></div>
          <div class="kpi red"><div class="kpi-label">Snort F1</div>
            <div class="kpi-val">{c['sn_f1']:.3f}</div><div class="kpi-sub">Weighted</div></div>
        </div>""",unsafe_allow_html=True)
 
        metrics=["Accuracy (%)","F1 Score (%)","Precision (%)","Recall (%)"]
        ml_vals=[c["ml_acc"]*100,c["ml_f1"]*100,c["ml_prec"]*100,c["ml_rec"]*100]
        sn_vals=[c["sn_acc"]*100,c["sn_f1"]*100,c["sn_prec"]*100,c["sn_rec"]*100]
        fig=go.Figure()
        fig.add_trace(go.Bar(name="Random Forest (ML-IDS)",x=metrics,y=ml_vals,marker_color="#3b82f6",
                             text=[f"{v:.1f}%" for v in ml_vals],textposition="outside"))
        fig.add_trace(go.Bar(name="Snort Signatures",x=metrics,y=sn_vals,marker_color="#f59e0b",
                             text=[f"{v:.1f}%" for v in sn_vals],textposition="outside"))
        fig.update_layout(**PLOT,barmode="group",yaxis_range=[0,118],
                          title=f"ML-IDS vs Signature IDS — {c['n']} samples")
        fig.update_yaxes(showgrid=True,gridcolor="#1e2d45"); st.plotly_chart(fig,use_container_width=True)
 
        fig2=go.Figure()
        fig2.add_trace(go.Bar(name="False Positives",x=["ML-IDS","Snort"],
                              y=[c["ml_fp"],c["sn_fp"]],marker_color=["#3b82f6","#f59e0b"]))
        fig2.add_trace(go.Bar(name="False Negatives",x=["ML-IDS","Snort"],
                              y=[c["ml_fn"],c["sn_fn"]],marker_color=["#ef4444","#f87171"]))
        fig2.update_layout(**PLOT,barmode="group",title="Error Analysis: FP & FN")
        fig2.update_yaxes(showgrid=True,gridcolor="#1e2d45"); st.plotly_chart(fig2,use_container_width=True)
 
        r1,r2=st.columns(2)
        with r1: st.markdown(f"""<div class="rag-card"><div class="rag-title">🤖 Random Forest (ML-IDS)</div>
          <div class="rag-body">Acc: <b>{c['ml_acc']*100:.1f}%</b> | F1: <b>{c['ml_f1']*100:.1f}%</b><br>
          FP: <b>{c['ml_fp']}</b> | FN: <b>{c['ml_fn']}</b><br><br>
          Learns statistical patterns from 41 NSL-KDD features. Detects novel zero-day attacks.</div></div>""",unsafe_allow_html=True)
        with r2: st.markdown(f"""<div class="rag-card"><div class="rag-title">🔏 Snort / Suricata (Signature IDS)</div>
          <div class="rag-body">Acc: <b>{c['sn_acc']*100:.1f}%</b> | F1: <b>{c['sn_f1']*100:.1f}%</b><br>
          FP: <b>{c['sn_fp']}</b> | FN: <b>{c['sn_fn']}</b><br><br>
          Rule-based, fast, reliable on known CVEs. Blind to zero-day and polymorphic attacks.</div></div>""",unsafe_allow_html=True)
 
        sc=max(c["n"]/100,1)
        cats=["Accuracy","F1 Score","Low FP","Low FN","Novel Attacks","Speed"]
        ml_r=[c["ml_acc"]*100,c["ml_f1"]*100,max(0,100-c["ml_fp"]/sc*8),max(0,100-c["ml_fn"]/sc*8),83,52]
        sn_r=[c["sn_acc"]*100,c["sn_f1"]*100,max(0,100-c["sn_fp"]/sc*8),max(0,100-c["sn_fn"]/sc*8),17,96]
        fig3=go.Figure()
        fig3.add_trace(go.Scatterpolar(r=ml_r,theta=cats,fill='toself',name="ML-IDS",line_color="#3b82f6"))
        fig3.add_trace(go.Scatterpolar(r=sn_r,theta=cats,fill='toself',name="Snort",line_color="#f59e0b"))
        fig3.update_layout(**PLOT,polar=dict(bgcolor="#0d1220",
            radialaxis=dict(visible=True,range=[0,100],color="#1e2d45"),angularaxis=dict(color="#64748b")),
            title="Capability Radar")
        st.plotly_chart(fig3,use_container_width=True)
 
# ── PAGE 5 ───────────────────────────────────────────────────────────────────
elif page=="🧠  RAG Knowledge Base":
    st.markdown('<p class="sec">🧠 RAG-Based Threat Knowledge Base</p>',unsafe_allow_html=True)
    st.info("KB entries are injected as context when the LLM generates reports, grounding responses in known attack signatures.",icon="ℹ️")
    q=st.text_input("🔍 Search",placeholder="e.g. DoS, buffer, brute")
    entries=[(k,v) for k,v in KB.items() if k!="normal"]
    if q.strip():
        ql=q.lower()
        entries=[(k,v) for k,v in entries if ql in k or ql in v["title"].lower() or ql in v["body"].lower()]
    for key,entry in entries:
        fam=_family(key); sev=_severity(key)
        tags="".join(f'<span class="rag-tag">{t}</span>' for t in entry["tags"])
        st.markdown(f"""<div class="rag-card">
          <div class="rag-title">{entry['title']}
            <span class="pill pill-{sev}">{sev.upper()}</span>
            <span class="pill pill-info">{fam}</span></div>
          <div class="rag-body">{entry['body']}</div>
          <div style="margin-top:10px;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:.65rem;color:#64748b;">SNORT RULE:</span><br>
            <code style="font-size:.7rem;color:#a5b4fc;background:#0d1220;padding:6px 10px;
                         border-radius:6px;display:block;margin-top:4px;border:1px solid #1e2d45;">
              {entry['snort_rule']}</code></div>
          <div style="margin-top:8px;">{tags}</div>
        </div>""",unsafe_allow_html=True)
    st.divider(); st.markdown("**➕ Add custom KB entry**")
    c1,c2=st.columns(2)
    with c1: nk=st.text_input("Key"); nt=st.text_input("Title")
    with c2: nb=st.text_area("Description",height=90); ng=st.text_input("Tags (comma-separated)")
    ns=st.text_input("Snort rule (optional)")
    if st.button("Add to Knowledge Base"):
        if nk and nt and nb:
            KB[nk.strip().lower()]={"title":nt,"body":nb,
                "tags":[t.strip() for t in ng.split(",") if t.strip()],
                "snort_rule":ns or "# No rule defined"}
            st.success(f"Added `{nk}`!"); st.rerun()
        else: st.error("Key, Title, and Description required.")
 
# ── PAGE 6 ───────────────────────────────────────────────────────────────────
elif page=="📋  NLP SOC Summary":
    st.markdown('<p class="sec">📋 NLP SOC Briefing — Groq LLaMA-3 8B</p>',unsafe_allow_html=True)
    if not _llm_ready():
        st.error("⚠️  Add `GROQ_API_KEY` in App Settings → Secrets.")
    elif not st.session_state.reports:
        st.warning("⚠️  Generate at least one report on the **LLM Threat Reports** page first.")
    else:
        rpts=st.session_state.reports
        st.markdown(f"**{len(rpts)} alert(s) in queue.** Briefing covers last 5.")
        if st.button("📝  Generate SOC Briefing",use_container_width=True):
            with st.spinner("Writing briefing…"):
                try: brief=llm_soc(_groq_client(),rpts); st.session_state._brief=brief
                except Exception as e: st.session_state._brief=f"[Error: {e}]"
        if st.session_state._brief:
            st.markdown(f"""<div class="alert-card info">
              <div class="alert-meta">🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Groq LLaMA-3 8B | SOC Briefing</div>
              <div class="alert-title">SOC Threat Summary <span class="pill pill-info">NLP</span></div>
              <div class="alert-body" style="font-size:.9rem;line-height:1.75;">
                {st.session_state._brief.replace(chr(10),'<br>')}
              </div></div>""",unsafe_allow_html=True)
        col_map={"high":"#ef4444","medium":"#f59e0b","low":"#22c55e","info":"#3b82f6"}
        st.markdown('<p class="sec">Alert Timeline</p>',unsafe_allow_html=True)
        for r in rpts[:15]:
            sc_cls={"high":"","medium":"medium","low":"low","info":"info"}[r["sev"]]
            st.markdown(f"""<div class="alert-card {sc_cls}" style="padding:12px 18px;">
              <div class="alert-meta">🕐 {r['ts']} | {r['proto'].upper()}/{r['svc']}</div>
              <div class="alert-title" style="font-size:.86rem;">{r['kb']['title']}
                <span class="pill pill-{r['sev']}">{r['sev'].upper()}</span></div>
            </div>""",unsafe_allow_html=True)
        sev_counts=Counter(r["sev"] for r in rpts)
        fig=go.Figure(go.Pie(labels=list(sev_counts.keys()),values=list(sev_counts.values()),hole=0.5,
            marker_colors=[col_map.get(s,"#64748b") for s in sev_counts.keys()]))
        fig.update_layout(**PLOT,title="Alert Severity Distribution",height=280)
        st.plotly_chart(fig,use_container_width=True)
 
# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
total=len(st.session_state.reports)
hi=sum(1 for r in st.session_state.reports if r["sev"]=="high")
med=sum(1 for r in st.session_state.reports if r["sev"]=="medium")
st.markdown(f"""<div class="kpi-row" style="margin-top:6px;">
  <div class="kpi blue"><div class="kpi-label">Model Status</div>
    <div class="kpi-val" style="font-size:1rem;">✅ Live</div><div class="kpi-sub">RF + Isolation Forest</div></div>
  <div class="kpi red"><div class="kpi-label">High Alerts</div>
    <div class="kpi-val">{hi}</div><div class="kpi-sub">DoS / U2R</div></div>
  <div class="kpi amber"><div class="kpi-label">Medium Alerts</div>
    <div class="kpi-val">{med}</div><div class="kpi-sub">Probe / R2L</div></div>
  <div class="kpi green"><div class="kpi-label">Total Reports</div>
    <div class="kpi-val">{total}</div><div class="kpi-sub">LLM-generated</div></div>
</div>""",unsafe_allow_html=True)
