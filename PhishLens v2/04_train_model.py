# Trains the Phase 1 Random Forest classifier and saves it to
# models/model_final.joblib.

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from features import url_to_features

DEEPLINK_WEIGHT = 8.0
TREES = 150
MAX_DEPTH = 14
SEED = 42


# PhiUSIIL rows are excluded: a leakage check showed a classifier could
# identify a row's source dataset with 100% accuracy, inflating benchmark
# scores without contributing phishing signal.
all_urls = pd.read_csv('data/processed/unified_urls.csv')
base = all_urls[all_urls['source'].isin(['phishtank', 'tranco'])]
base = base[['url', 'label', 'source']]
print(f"Base rows: {len(base):,}")

deeplinks = pd.read_csv('data/processed/tranco_deeplinks.csv')
deeplinks = deeplinks.rename(columns={'URL': 'url'})[['url']]
deeplinks['label'] = 1
deeplinks['source'] = 'deeplink'
print(f"Real deep links: {len(deeplinks):,}")


# Equivalent renderings of the same address, so a trailing slash or a www.
# prefix cannot alter the verdict
def spelling_variants(url):
    variants = []
    parts = urlparse(url)
    if parts.path == '':
        variants.append(url + '/')
    if not parts.netloc.startswith('www.'):
        with_www = url.replace('://', '://www.', 1)
        variants.append(with_www)
        variants.append(with_www + '/')
    return variants


variant_urls = []
for url in base[base['source'] == 'tranco']['url'].astype(str):
    variant_urls.extend(spelling_variants(url))

variants = pd.DataFrame({'url': variant_urls})
variants['label'] = 1
variants['source'] = 'format_variant'

training_data = pd.concat([base, deeplinks, variants], ignore_index=True)
training_data = training_data.drop_duplicates('url').reset_index(drop=True)
print(f"Training set: {len(training_data):,} rows")

training_data['website'] = training_data['url'].astype(str).apply(
    lambda u: urlparse(u).netloc.lower().replace('www.', ''))

# Harvested deep links are few but carry the correction for the path-length
# bias, so they are weighted up relative to the rest of the training set.
row_importance = np.ones(len(training_data))
row_importance[training_data['source'] == 'deeplink'] = DEEPLINK_WEIGHT

print("Extracting features...")
features = pd.DataFrame([url_to_features(u) for u in training_data['url']])
features = features.astype(np.float32)
feature_names = list(features.columns)

# Grouped by website so no domain appears in both partitions, preventing
# near-duplicate URL leakage across the split.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
train_rows, test_rows = next(splitter.split(
    features, training_data['label'], groups=training_data['website']))

print("Training...")
model = RandomForestClassifier(n_estimators=TREES, max_depth=MAX_DEPTH,
                               random_state=SEED, n_jobs=-1)
model.fit(features.iloc[train_rows],
          training_data['label'].iloc[train_rows],
          sample_weight=row_importance[train_rows])

test_scores = model.predict_proba(features.iloc[test_rows])[:, 1]
auc = roc_auc_score(training_data['label'].iloc[test_rows], test_scores)
print(f"\nTest AUC: {auc:.4f}")

Path('models').mkdir(exist_ok=True)
joblib.dump({'model': model, 'features': feature_names},
            'models/model_final.joblib')


# The deployed extension applies two rules before consulting the model
# score. Reproduced here so validation reflects deployed behaviour rather
# than the classifier in isolation.
def final_verdict(url, model_score):
    flags = url_to_features(url)
    if flags['brand_is_legit'] == 1:
        return 1, 'rule: genuine brand domain'
    if flags['brand_domain_mismatch'] == 1:
        return 0, 'rule: imitates a brand'
    return (1 if model_score >= 0.5 else 0), f'model score {model_score:.3f}'


VALIDATION_SET = [
    ("https://www.bbc.co.uk/", 1),
    ("https://www.sofascore.com/", 1),
    ("https://en.wikipedia.org/wiki/Phishing", 1),
    ("https://www.google.com/search?q=car", 1),
    ("https://www.irfanview.com/faq.htm", 1),
    ("https://paypal.com/signin", 1),
    ("http://secure-paypal-account.verify-login.xyz/update.php", 0),
    ("https://accounts.paypa1.com/signin", 0),
]

check_features = pd.DataFrame(
    [url_to_features(u) for u, _ in VALIDATION_SET])[feature_names].astype(np.float32)
scores = model.predict_proba(check_features)[:, 1]

print("\n=== VALIDATION (rules + model, as deployed) ===")
passed = 0
for (url, expected), score in zip(VALIDATION_SET, scores):
    predicted, reason = final_verdict(url, score)
    if predicted == expected:
        passed += 1
    print(f"  {'PASS' if predicted == expected else 'FAIL'}  "
          f"{url[:48]:<48}  {reason}")

print(f"\n{passed}/{len(VALIDATION_SET)} passed")
if passed < len(VALIDATION_SET):
    print("Failures indicate an insufficient or insufficiently varied")
    print("deep-link harvest. Re-run step 3, then run this script again.")
else:
    print("Next: python 05_export_model.py")
