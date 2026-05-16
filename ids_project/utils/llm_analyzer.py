"""
llm_analyzer.py  —  OpenAI LLM integration with graceful rule-based fallback.
Reads LLM settings from Streamlit session_state when available.
"""
import os
import streamlit as st
from rag.knowledge_base import get_kb

# ── Threat metadata ──────────────────────────────────────────────────────────
SEVERITY = {
    "DoS":          ("Critical", "🔴"),
    "BruteForce":   ("High",     "🟠"),
    "PortScan":     ("Medium",   "🟡"),
    "SQLInjection": ("Critical", "🔴"),
    "Backdoor":     ("Critical", "🔴"),
    "Benign":       ("None",     "🟢"),
}
MITIGATION = {
    "DoS":
        "Enable SYN cookies (`sysctl net.ipv4.tcp_syncookies=1`), deploy rate-limiting rules "
        "(iptables/nftables), activate DDoS scrubbing via Cloudflare or AWS Shield, and "
        "implement connection throttling on load balancers.",
    "BruteForce":
        "Block source IP with fail2ban (≥5 failed attempts), enforce SSH key-based auth, "
        "disable password login (`PasswordAuthentication no`), enable MFA, and restrict "
        "SSH/RDP access to VPN-only CIDRs.",
    "PortScan":
        "Drop unsolicited SYN probes with firewall rules, implement port-knocking, enable "
        "network segmentation, configure IDS alerts on probe frequency thresholds, and "
        "limit ICMP echo responses.",
    "SQLInjection":
        "Deploy a WAF (ModSecurity / AWS WAF) with OWASP ruleset, audit all application "
        "query builders for parameterised queries, use least-privilege DB accounts, and "
        "enable application-layer logging for anomaly detection.",
    "Backdoor":
        "Isolate the compromised host immediately (network quarantine), capture memory "
        "forensics, scan running processes for known C2 signatures, check crontabs and "
        "startup scripts, sinkhole the C2 domain, and re-image if compromise is confirmed.",
    "Benign":
        "No immediate action required. Continue routine monitoring.",
}
IMPACT = {
    "DoS":          "Service outage — legitimate users unable to reach the target service.",
    "BruteForce":   "Account compromise — if successful, attacker gains full system access.",
    "PortScan":     "Reconnaissance — attacker maps open services for subsequent exploitation.",
    "SQLInjection": "Data breach — attacker may exfiltrate, modify, or destroy database contents.",
    "Backdoor":     "Full system compromise — persistent attacker access and potential data exfiltration.",
    "Benign":       "No impact — normal network operation.",
}

# ── Settings helpers ──────────────────────────────────────────────────────────
def _get(key, default):
    """Read from st.session_state if running inside Streamlit, else env/default."""
    try:
        return st.session_state.get(key, default)
    except Exception:
        return os.getenv(key.upper(), default)

def _api_key() -> str:
    return _get("api_key", os.getenv("OPENAI_API_KEY", ""))

def _model() -> str:
    return _get("llm_model", "gpt-4o-mini")

def _temperature() -> float:
    return float(_get("temperature", 0.3))

def _max_tokens() -> int:
    return int(_get("max_tokens", 700))

# ── Fallback report ───────────────────────────────────────────────────────────
def _fallback_report(record: dict, prediction: dict) -> str:
    ptype = prediction.get("prediction", "Unknown")
    conf  = prediction.get("confidence", 0)
    sev, icon = SEVERITY.get(ptype, ("Unknown","⚪"))
    mit  = MITIGATION.get(ptype, "Review manually.")
    imp  = IMPACT.get(ptype, "Unknown impact.")

    kb      = get_kb()
    entries = kb.retrieve(f"{ptype} attack network flood scan", top_k=1)
    detail  = entries[0]["description"] if entries else "No additional intelligence available."

    if ptype == "Benign":
        return (
            f"**{icon} Traffic Classification: Benign**\n\n"
            f"The ML model classified this flow as **benign** with **{conf}% confidence**. "
            f"The traffic characteristics (destination port `{record.get('dst_port')}`, "
            f"protocol `{record.get('protocol')}`, "
            f"packet count `{record.get('packet_count')}`) are consistent with normal "
            f"network activity and show no anomalous indicators.\n\n"
            f"**Recommended Action:** Continue routine monitoring. No response required."
        )

    return (
        f"**{icon} Threat Classification: {ptype}** — Severity: **{sev}**\n\n"
        f"**Confidence:** {conf}%\n\n"
        f"---\n\n"
        f"**What this traffic pattern indicates:**\n"
        f"{detail}\n\n"
        f"**Observed indicators in this flow:**\n"
        f"- Source: `{record.get('src_ip')}:{record.get('src_port')}` → "
        f"Destination: `{record.get('dst_ip')}:{record.get('dst_port')}`\n"
        f"- Protocol: `{record.get('protocol')}` | Flag: `{record.get('flag')}`\n"
        f"- Packets: `{record.get('packet_count'):,}` | Bytes: `{record.get('byte_count'):,}`\n"
        f"- Duration: `{record.get('duration_ms')} ms` | "
        f"Rate: `{record.get('packets_per_sec')} pkt/s`\n\n"
        f"**Potential impact:**\n"
        f"{imp}\n\n"
        f"**Immediate response actions:**\n"
        f"{mit}\n\n"
        f"**Long-term hardening:**\n"
        f"Implement network segmentation, enforce least-privilege policies, "
        f"deploy a SIEM for continuous correlation, and schedule quarterly penetration testing."
    )

# ── OpenAI report ─────────────────────────────────────────────────────────────
def _openai_report(record: dict, prediction: dict, api_key: str) -> str:
    from openai import OpenAI

    kb      = get_kb()
    ptype   = prediction.get("prediction", "Unknown")
    entries = kb.retrieve(
        f"{ptype} port {record.get('dst_port')} "
        f"protocol {record.get('protocol')} flag {record.get('flag')}",
        top_k=2)
    ctx = kb.format_context_for_llm(entries)

    prompt = f"""You are a senior SOC analyst reviewing a network IDS alert.

## Traffic Flow
| Field | Value |
|---|---|
| Source | {record.get('src_ip')}:{record.get('src_port')} |
| Destination | {record.get('dst_ip')}:{record.get('dst_port')} |
| Protocol | {record.get('protocol')} / {record.get('flag')} |
| Packets | {record.get('packet_count'):,} |
| Bytes | {record.get('byte_count'):,} |
| Duration | {record.get('duration_ms')} ms |
| Packet Rate | {record.get('packets_per_sec')} pkt/s |
| Avg Packet Size | {record.get('avg_packet_size')} bytes |

## ML Detection
- **Predicted Threat:** {ptype}
- **Confidence:** {prediction.get('confidence')}%

## Retrieved Threat Intelligence
{ctx}

Write a concise 4-paragraph professional analyst report:
1. What this traffic pattern indicates and why it was flagged
2. Severity assessment and likely attacker objective
3. Potential impact if the attack succeeds
4. Immediate containment steps and long-term hardening recommendations

Use markdown formatting. Be specific, actionable, and accurate."""

    client = OpenAI(api_key=api_key)
    resp   = client.chat.completions.create(
        model=_model(),
        messages=[
            {"role":"system","content":
             "You are a senior cybersecurity analyst specialising in network intrusion detection "
             "and incident response. Be precise, professional, and actionable."},
            {"role":"user","content":prompt},
        ],
        temperature=_temperature(),
        max_tokens=_max_tokens(),
    )
    return resp.choices[0].message.content

def analyze_threat(record: dict, prediction: dict) -> tuple[str, bool]:
    """
    Returns (report_text, llm_used).
    llm_used=True means GPT-4o was used; False = rule-based fallback.
    """
    key = _api_key()
    if key and len(key) > 20:
        try:
            return _openai_report(record, prediction, key), True
        except Exception as e:
            err = str(e)
            fallback = _fallback_report(record, prediction)
            return fallback + f"\n\n> ⚠️ GPT-4o unavailable: `{err[:80]}`", False
    return _fallback_report(record, prediction), False

# ── Batch summarizer ──────────────────────────────────────────────────────────
def _fallback_summary(alerts: list) -> str:
    from collections import Counter
    counts = Counter(a.get("prediction","?") for a in alerts)
    threats= {k:v for k,v in counts.items() if k!="Benign"}
    benign = counts.get("Benign",0)
    total  = len(alerts)
    if not threats:
        return f"✅ All {total} analysed events appear **benign**. No action required."
    top = max(threats, key=threats.get)
    lines = [
        f"**Security Situation Report — {total} events analysed**\n",
        f"Out of {total} network events, **{sum(threats.values())} were flagged as malicious** "
        f"and {benign} classified as benign.",
        "\n**Threat breakdown:**",
    ]
    for t,c in sorted(threats.items(), key=lambda x:-x[1]):
        sev,ico = SEVERITY.get(t,("Unknown","⚪"))
        lines.append(f"- {ico} **{t}**: {c} alerts  ·  Severity: {sev}")
    lines += [
        f"\n**Highest priority:** Investigate **{top}** activity immediately.",
        f"{MITIGATION.get(top,'')}",
        "\n**Recommended SOC actions:**",
        "1. Block top attacker IPs at the perimeter firewall",
        "2. Review auth logs for successful breaches",
        "3. Escalate Critical events to the IR team",
        "4. Enable enhanced logging on affected services",
    ]
    return "\n".join(lines)

def summarize_alerts_batch(alerts: list) -> tuple[str, bool]:
    if not alerts: return "No alerts to summarise.", False
    key = _api_key()
    if key and len(key) > 20:
        try:
            from openai import OpenAI
            from collections import Counter
            counts = Counter(a.get("prediction") for a in alerts)
            lines  = "\n".join(
                f"{i+1}. [{a.get('timestamp','?')}] {a.get('prediction')} "
                f"({a.get('confidence',0)}%)  Src:{a.get('src_ip','?')} → Port {a.get('dst_port','?')}"
                for i,a in enumerate(alerts[:20]))
            client = OpenAI(api_key=key)
            resp   = client.chat.completions.create(
                model=_model(),
                messages=[
                    {"role":"system","content":"You are a SOC analyst writing executive summaries."},
                    {"role":"user","content":(
                        f"Recent IDS alerts (last {len(alerts)} events):\n{lines}\n\n"
                        f"Breakdown: {dict(counts)}\n\n"
                        "Write a 3-paragraph executive summary covering: current threat landscape, "
                        "critical threats needing immediate attention, and recommended priority actions."
                    )},
                ],
                temperature=_temperature(), max_tokens=500,
            )
            return resp.choices[0].message.content, True
        except Exception as e:
            return _fallback_summary(alerts) + f"\n\n> ⚠️ `{str(e)[:80]}`", False
    return _fallback_summary(alerts), False

# ── Security assistant chat ────────────────────────────────────────────────────
CANNED = {
    "syn flood":    "A **SYN flood** exploits TCP's 3-way handshake by sending massive SYN packets without completing them, exhausting server memory. **Fix:** SYN cookies, rate-limiting, CDN/scrubbing.",
    "port scan":    "**Port scanning** probes ports to discover open services. Tools: Nmap, Masscan. **Detect:** high connection rate to varied ports. **Block:** firewall rules, port-knocking.",
    "sql":          "**SQL Injection** embeds malicious SQL in HTTP params. Network signs: anomalous payload size on port 80/443. **Prevent:** WAF, parameterised queries, input validation.",
    "backdoor":     "**Backdoor/C2 beaconing** uses non-standard ports (4444, 8888) with low-frequency, long-duration sessions. **Respond:** isolate host, memory forensics, re-image.",
    "brute force":  "**Brute force** repeats auth attempts on SSH(22)/FTP(21)/RDP(3389). **Block:** fail2ban, key-based auth, account lockout, MFA, VPN-only access.",
    "ids":          "An **IDS** (Intrusion Detection System) monitors network traffic for suspicious patterns using signatures, anomaly detection, or ML classifiers. This system uses Random Forest ML + LLM explanation.",
    "rag":          "**RAG (Retrieval-Augmented Generation)** grounds LLM outputs in a retrieved knowledge base, reducing hallucination and improving factual accuracy for threat reports.",
}

def chat_with_security_assistant(user_msg: str, history: list) -> tuple[str, list]:
    key = _api_key()
    if key and len(key) > 20:
        try:
            from openai import OpenAI
            kb      = get_kb()
            entries = kb.retrieve(user_msg, top_k=2)
            ctx     = kb.format_context_for_llm(entries)
            sys_p   = (
                "You are an expert cybersecurity analyst and IDS assistant. "
                "Help SOC analysts understand threats, alerts, and mitigations.\n\n"
                f"Current Threat Intelligence:\n{ctx}"
            )
            msgs = [{"role":"system","content":sys_p}]
            msgs += history[-10:]
            msgs.append({"role":"user","content":user_msg})
            client = OpenAI(api_key=key)
            resp   = client.chat.completions.create(
                model=_model(), messages=msgs,
                temperature=_temperature(), max_tokens=600)
            reply = resp.choices[0].message.content
        except Exception as e:
            reply = _canned(user_msg) + f"\n\n> ⚠️ GPT-4o unavailable: `{str(e)[:60]}`"
    else:
        reply = _canned(user_msg)
    history.append({"role":"user",      "content":user_msg})
    history.append({"role":"assistant", "content":reply})
    return reply, history

def _canned(msg: str) -> str:
    ml = msg.lower()
    for kw, r in CANNED.items():
        if kw in ml:
            return r + "\n\n> *Configure OpenAI key in ⚙️ Advanced Settings for full GPT-4o responses.*"
    return (
        "I can answer questions about: **SYN flood, port scanning, SQL injection, "
        "brute force, backdoors, IDS concepts, and RAG**.\n\n"
        "For full AI-powered responses, configure your OpenAI key in **⚙️ Advanced Settings**."
    )
