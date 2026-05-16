"""
knowledge_base.py  —  TF-IDF RAG threat knowledge base.
Retrieves relevant threat intel to enrich LLM prompts.
"""

import json, os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = [
    dict(id="dos_001", threat_type="DoS", title="SYN Flood Attack",
         description="SYN flood exploits the TCP three-way handshake. Attacker sends massive SYN packets without completing handshake, exhausting server connection tables. Signature: huge packet_count, SYN flags, short duration, TCP to ports 80/443.",
         indicators=["high packet_count >5000","SYN flag","dst_port 80/443","TCP","high packets_per_sec"],
         mitigation="Rate limiting, SYN cookies, firewall threshold rules, CDN/DDoS protection (Cloudflare).",
         severity="Critical"),
    dict(id="dos_002", threat_type="DoS", title="UDP Flood / Amplification",
         description="UDP flood sends massive UDP packets to random ports overwhelming bandwidth. Amplification leverages DNS/NTP to multiply traffic. High byte_count, UDP protocol, random destination ports.",
         indicators=["UDP protocol","high byte_count","random dst_port","very high packets_per_sec"],
         mitigation="Ingress filtering BCP38, rate-limit UDP, block unused UDP services.",
         severity="High"),
    dict(id="scan_001", threat_type="PortScan", title="TCP SYN Stealth Scan",
         description="Port scanning probes many ports to discover open services. SYN scan sends SYN without completing handshake (half-open). Single source hitting many different destination ports with low bytes per connection.",
         indicators=["SYN flag","varying dst_port 1-1024","low byte_count","moderate packet_count 100-1000"],
         mitigation="IDS/IPS signatures, firewall port filtering, port knocking, network segmentation.",
         severity="Medium"),
    dict(id="brute_001", threat_type="BruteForce", title="SSH Brute Force",
         description="Automated credential guessing on SSH (port 22). Many repeated TCP connections from same IP with authentication attempts. High connection count over long duration.",
         indicators=["dst_port 22","high packet_count over long duration_ms","TCP protocol","repeated src_ip"],
         mitigation="Fail2ban, key-based SSH auth, disable password auth, non-standard SSH port, MFA.",
         severity="High"),
    dict(id="brute_002", threat_type="BruteForce", title="FTP/RDP Brute Force",
         description="Credential attacks on FTP port 21 or RDP port 3389. Tools: Hydra, Medusa. High frequency connections to a single service port from the same source.",
         indicators=["dst_port 21 or 3389","high packet_count","long duration_ms","same src_ip"],
         mitigation="Disable FTP, use SFTP, restrict RDP to VPN only, account lockout policies.",
         severity="High"),
    dict(id="sqli_001", threat_type="SQLInjection", title="HTTP SQL Injection",
         description="SQL injection embeds malicious SQL in HTTP request parameters targeting port 80/443/8080. Network signature: abnormal payload sizes, TCP web traffic with unusual byte counts per request.",
         indicators=["dst_port 80/443/8080","TCP","abnormal avg_packet_size","HTTP traffic pattern"],
         mitigation="Parameterized queries, WAF (ModSecurity), input validation, least-privilege DB accounts.",
         severity="Critical"),
    dict(id="back_001", threat_type="Backdoor", title="Reverse Shell / C2 Beacon",
         description="Backdoor traffic creates persistent channels to attacker C2 servers. High non-standard ports (4444-9999), long session duration, periodic low-rate beaconing, encrypted-looking byte patterns.",
         indicators=["dst_port 4444-9999","very long duration_ms >10000","low packet_count","non-standard port"],
         mitigation="Egress firewall filtering, EDR endpoint detection, DNS sinkholing, threat intel feeds.",
         severity="Critical"),
    dict(id="benign_001", threat_type="Benign", title="Normal Web Browsing",
         description="Standard HTTP/HTTPS traffic. Moderate packet counts, ports 80/443, typical client-server handshakes, reasonable byte counts and durations.",
         indicators=["dst_port 80/443","moderate packet_count 5-50","normal byte_count","short duration"],
         mitigation="No action required.",
         severity="None"),
]

_kb_inst = None

class ThreatKnowledgeBase:
    def __init__(self):
        self.kb = KB
        texts = [f"{e['title']} {e['description']} {' '.join(e['indicators'])}" for e in KB]
        self._vec = TfidfVectorizer(stop_words="english")
        self._mat = self._vec.fit_transform(texts)

    def retrieve(self, query: str, top_k=3) -> list:
        q   = self._vec.transform([query])
        sc  = cosine_similarity(q, self._mat).flatten()
        idx = sc.argsort()[-top_k:][::-1]
        out = []
        for i in idx:
            e = self.kb[i].copy()
            e["relevance_score"] = round(float(sc[i]), 4)
            out.append(e)
        return out

    def get_by_type(self, t: str) -> list:
        return [e for e in self.kb if e["threat_type"].lower() == t.lower()]

    def to_llm_context(self, entries: list) -> str:
        if not entries:
            return "No specific threat intelligence found."
        parts = []
        for e in entries:
            parts.append(
                f"### {e['title']}  (Type: {e['threat_type']} | Severity: {e['severity']})\n"
                f"**Description:** {e['description']}\n"
                f"**Indicators:** {', '.join(e['indicators'])}\n"
                f"**Mitigation:** {e['mitigation']}"
            )
        return "\n\n---\n\n".join(parts)

    def save(self, path="rag/knowledge_base.json"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,"w") as f: json.dump(self.kb, f, indent=2)


def get_kb() -> ThreatKnowledgeBase:
    global _kb_inst
    if _kb_inst is None:
        _kb_inst = ThreatKnowledgeBase()
    return _kb_inst


if __name__ == "__main__":
    kb = get_kb()
    res = kb.retrieve("SYN flood high packet TCP", top_k=2)
    for r in res:
        print(f"[{r['relevance_score']}] {r['title']} — {r['threat_type']}")
    kb.save()
    print("Knowledge base saved.")
