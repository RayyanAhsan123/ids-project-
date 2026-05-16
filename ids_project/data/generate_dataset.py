"""
generate_dataset.py
Generates a realistic synthetic network traffic CSV for IDS training.
Labels: Benign, DoS, PortScan, BruteForce, SQLInjection, Backdoor
"""

import os, random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

ATTACK_TYPES = ["Benign", "DoS", "PortScan", "BruteForce", "SQLInjection", "Backdoor"]
PROTOCOLS    = ["TCP", "UDP", "ICMP"]
FLAGS        = ["SYN", "ACK", "SYN-ACK", "FIN", "RST", "PSH-ACK"]

WEIGHTS = dict(Benign=0.50, DoS=0.15, PortScan=0.12,
               BruteForce=0.10, SQLInjection=0.08, Backdoor=0.05)


def _make(label):
    r = dict(
        src_ip=f"192.168.{random.randint(1,10)}.{random.randint(1,254)}",
        dst_ip=f"10.0.{random.randint(0,5)}.{random.randint(1,254)}",
        src_port=random.randint(1024, 65535),
        dst_port=0, protocol=random.choice(PROTOCOLS), flag=random.choice(FLAGS),
        packet_count=0, byte_count=0, duration_ms=0,
        packets_per_sec=0.0, avg_packet_size=0.0, label=label,
    )
    if label == "Benign":
        r.update(dst_port=random.choice([80,443,8080,22,53,25]),
                 packet_count=random.randint(5,50),
                 byte_count=random.randint(500,50000),
                 duration_ms=random.randint(100,5000))
    elif label == "DoS":
        r.update(dst_port=random.choice([80,443]), protocol="TCP", flag="SYN",
                 packet_count=random.randint(5000,50000),
                 byte_count=random.randint(100000,5000000),
                 duration_ms=random.randint(1000,10000))
    elif label == "PortScan":
        r.update(dst_port=random.randint(1,1024), flag="SYN",
                 packet_count=random.randint(100,1000),
                 byte_count=random.randint(1000,10000),
                 duration_ms=random.randint(500,3000))
    elif label == "BruteForce":
        r.update(dst_port=random.choice([22,21,3389]),
                 packet_count=random.randint(200,2000),
                 byte_count=random.randint(10000,200000),
                 duration_ms=random.randint(2000,30000))
    elif label == "SQLInjection":
        r.update(dst_port=random.choice([80,443,8080]), protocol="TCP",
                 packet_count=random.randint(10,100),
                 byte_count=random.randint(2000,20000),
                 duration_ms=random.randint(50,500))
    elif label == "Backdoor":
        r.update(dst_port=random.randint(4444,9999),
                 packet_count=random.randint(20,200),
                 byte_count=random.randint(5000,100000),
                 duration_ms=random.randint(10000,60000))

    if r["duration_ms"] > 0:
        r["packets_per_sec"] = round(r["packet_count"] / (r["duration_ms"] / 1000), 4)
    if r["packet_count"] > 0:
        r["avg_packet_size"] = round(r["byte_count"] / r["packet_count"], 4)
    return r


def generate_dataset(n=5000, path="data/network_traffic.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = random.choices(list(WEIGHTS), weights=list(WEIGHTS.values()), k=n)
    df = pd.DataFrame([_make(l) for l in labels])
    df.to_csv(path, index=False)
    print(f"✅ Dataset → {path}  ({n} records)")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    generate_dataset()

def generate_record(label: str) -> dict:
    """Public alias so app.py can call generate_record(label)."""
    return _make(label)
