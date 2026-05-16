"""
spark_processor.py  —  Batch traffic analysis.
Uses PySpark when Java is available; falls back to pandas automatically.
"""

import os
import pandas as pd

try:
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    SPARK = True
except ImportError:
    SPARK = False


# ── pandas implementation ─────────────────────────────────────────────────────

def _pandas_analysis(path: str) -> dict:
    print("🐼 Pandas batch analysis …")
    df = pd.read_csv(path)
    n  = len(df)

    threat_dist = df["label"].value_counts().to_dict()

    atk = df[df["label"] != "Benign"]
    top_src = (atk.groupby("src_ip").size()
                   .sort_values(ascending=False).head(10)
                   .reset_index(name="attacks")
                   .to_dict("records"))

    top_ports = (atk.groupby(["dst_port","label"]).size()
                    .sort_values(ascending=False).head(10)
                    .reset_index(name="count")
                    .to_dict("records"))
    # cast port to int for JSON
    for r in top_ports: r["dst_port"] = int(r["dst_port"])

    dos_hv  = int(df[(df["label"]=="DoS") & (df["packet_count"]>1000)].shape[0])
    proto   = df["protocol"].value_counts().to_dict()

    # per-hour bucketing (fake timestamps for demo)
    df["hour"] = pd.Series(range(n)) // (n // 24) % 24
    timeline = df.groupby(["hour","label"]).size().reset_index(name="count").to_dict("records")

    return dict(
        total_records        = n,
        threat_distribution  = threat_dist,
        top_attacker_ips     = top_src,
        top_attacked_ports   = top_ports,
        high_volume_dos      = dos_hv,
        protocol_distribution= proto,
        timeline             = timeline,
        engine               = "pandas",
    )


# ── spark implementation ──────────────────────────────────────────────────────

def _spark_analysis(path: str) -> dict:
    print("⚡ Spark batch analysis …")
    spark = (SparkSession.builder
             .appName("IDS_Analysis")
             .config("spark.sql.shuffle.partitions","4")
             .config("spark.driver.memory","2g")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.csv(path, header=True, inferSchema=True)
    n  = df.count()

    threat_dist = {r["label"]: r["count"]
                   for r in df.groupBy("label").count().collect()}

    atk = df.filter(F.col("label") != "Benign")

    top_src = [{"src_ip":r["src_ip"],"attacks":r["count"]}
               for r in atk.groupBy("src_ip").count()
                           .orderBy(F.col("count").desc()).limit(10).collect()]

    top_ports = [{"dst_port":r["dst_port"],"label":r["label"],"count":r["count"]}
                 for r in atk.groupBy("dst_port","label").count()
                             .orderBy(F.col("count").desc()).limit(10).collect()]

    dos_hv = df.filter((F.col("label")=="DoS") & (F.col("packet_count")>1000)).count()
    proto  = {r["protocol"]:r["count"]
              for r in df.groupBy("protocol").count().collect()}

    spark.stop()
    return dict(
        total_records        = n,
        threat_distribution  = threat_dist,
        top_attacker_ips     = top_src,
        top_attacked_ports   = top_ports,
        high_volume_dos      = dos_hv,
        protocol_distribution= proto,
        timeline             = [],
        engine               = "PySpark",
    )


# ── public entry point ────────────────────────────────────────────────────────

def run_batch_analysis(path: str = "data/network_traffic.csv") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return _spark_analysis(path) if SPARK else _pandas_analysis(path)


if __name__ == "__main__":
    import json
    print(json.dumps(run_batch_analysis(), indent=2, default=str))
