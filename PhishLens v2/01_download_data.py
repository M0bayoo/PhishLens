# Downloads the three source datasets into data/raw/:
#   phiusiil_raw.csv, phishtank_raw.csv, tranco_top1m.csv

import csv
import json
import zipfile
import urllib.request
from pathlib import Path

RAW = Path('data/raw')
RAW.mkdir(parents=True, exist_ok=True)


print("[1/3] Downloading PhiUSIIL...")
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=967)
table = dataset.data.features.copy()
table['label'] = dataset.data.targets
table.to_csv(RAW / 'phiusiil_raw.csv', index=False)
print(f"  Saved {len(table):,} rows")


print("[2/3] Downloading PhishTank...")
try:
    request = urllib.request.Request(
        "http://data.phishtank.com/data/online-valid.json",
        headers={"User-Agent": "Mozilla/5.0"})

    with urllib.request.urlopen(request, timeout=60) as response:
        entries = json.loads(response.read().decode())

    with open(RAW / 'phishtank_raw.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["url", "phish_id", "target",
                         "submission_time", "verified", "online"])
        for entry in entries:
            writer.writerow([entry.get("url", ""), entry.get("phish_id", ""),
                             entry.get("target", ""), entry.get("submission_time", ""),
                             entry.get("verified", ""), entry.get("online", "")])
    print(f"  Saved {len(entries):,} entries")

# PhishTank rate-limits and blocks automated clients; manual download is
# the documented fallback.
except Exception as error:
    print(f"  FAILED: {error}")
    print("  Download manually from:")
    print("    http://data.phishtank.com/data/online-valid.csv")
    print("  and save as data/raw/phishtank_raw.csv")


print("[3/3] Downloading Tranco Top 1M...")
zip_path = RAW / 'tranco.zip'
request = urllib.request.Request("https://tranco-list.eu/top-1m.csv.zip",
                                 headers={"User-Agent": "Mozilla/5.0"})

with urllib.request.urlopen(request, timeout=120) as response:
    zip_path.write_bytes(response.read())

with zipfile.ZipFile(zip_path) as archive:
    inner_name = archive.namelist()[0]
    archive.extract(inner_name, RAW)
    (RAW / inner_name).rename(RAW / 'tranco_top1m.csv')

zip_path.unlink()
print("  Saved tranco_top1m.csv")
print("\nNext: python 02_merge_data.py")
