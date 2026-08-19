# Merges the three source datasets into data/processed/unified_urls.csv.
# Label convention inherited from PhiUSIIL: 1 = legitimate, 0 = phishing.

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path('data/raw')
PROCESSED = Path('data/processed')
PROCESSED.mkdir(parents=True, exist_ok=True)

LEGITIMATE = 1
PHISHING = 0


phiusiil = pd.read_csv(RAW / 'phiusiil_raw.csv')
phiusiil['source'] = 'phiusiil'
phiusiil['clean_url'] = phiusiil['URL']
print(f"PhiUSIIL: {len(phiusiil):,} rows")


phishtank = pd.read_csv(RAW / 'phishtank_raw.csv')

# Restricted to entries PhishTank has both confirmed and observed online,
# matching the "recent, confirmed phishing" sampling frame.
if 'verified' in phishtank.columns:
    phishtank = phishtank[phishtank['verified'].astype(str).str.lower() == 'yes']
if 'online' in phishtank.columns:
    phishtank = phishtank[phishtank['online'].astype(str).str.lower() == 'yes']

phishtank = phishtank.sample(n=min(5000, len(phishtank)), random_state=42)
phishtank['label'] = PHISHING
phishtank['source'] = 'phishtank'
phishtank['clean_url'] = phishtank['url']
print(f"PhishTank (verified + online): {len(phishtank):,} rows")


tranco = pd.read_csv(RAW / 'tranco_top1m.csv', header=None,
                     names=['rank', 'domain'])

# Sampled across three popularity bands so the legitimate class is not
# composed solely of high-traffic sites.
very_popular = tranco[tranco['rank'] <= 10_000]
popular      = tranco[(tranco['rank'] > 10_000) & (tranco['rank'] <= 100_000)]
less_popular = tranco[(tranco['rank'] > 100_000) & (tranco['rank'] <= 1_000_000)]

random_seed = np.random.RandomState(42)
tranco_sample = pd.concat([
    very_popular.sample(n=2000, random_state=random_seed),
    popular.sample(n=2000, random_state=random_seed),
    less_popular.sample(n=2000, random_state=random_seed),
], ignore_index=True)

tranco_sample['clean_url'] = 'https://' + tranco_sample['domain'].astype(str)
tranco_sample['label'] = LEGITIMATE
tranco_sample['source'] = 'tranco'
print(f"Tranco (3 popularity bands): {len(tranco_sample):,} rows")


columns = ['clean_url', 'label', 'source']
combined = pd.concat([phiusiil[columns], phishtank[columns], tranco_sample[columns]],
                     ignore_index=True)
combined = combined.rename(columns={'clean_url': 'url'})

before = len(combined)
combined = combined.drop_duplicates(subset='url').reset_index(drop=True)
print(f"\nTotal: {before:,} -> {len(combined):,} "
      f"(removed {before - len(combined)} duplicates)")
print(combined['source'].value_counts().to_string())
print(combined['label'].value_counts()
      .rename({1: 'legitimate', 0: 'phishing'}).to_string())

combined.to_csv(PROCESSED / 'unified_urls.csv', index=False)
print("\nSaved data/processed/unified_urls.csv")
print("Next: python 03_harvest_deeplinks.py")
