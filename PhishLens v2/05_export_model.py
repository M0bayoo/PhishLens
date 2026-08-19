# Exports the trained forest to forest_model_e.json for in-browser
# evaluation. Each tree is flattened into parallel arrays a dependency-free
# JavaScript tree-walker can traverse, replacing the ONNX Runtime Web
# approach that could not be loaded inside a Manifest V3 service worker.

import json
import joblib
import numpy as np
import pandas as pd

saved = joblib.load('models/model_final.joblib')
model = saved['model']
feature_names = saved['features']

# f  = feature index tested at this node (-2 marks a leaf)
# th = decision threshold
# L  = child node index when value <= threshold
# R  = child node index when value > threshold
# v  = leaf value, as P(legitimate)
trees = []
for tree_model in model.estimators_:
    tree = tree_model.tree_
    counts = tree.value[:, 0, :]
    trees.append({
        'f':  [int(x) for x in tree.feature],
        'th': [float(x) for x in tree.threshold],
        'L':  [int(x) for x in tree.children_left],
        'R':  [int(x) for x in tree.children_right],
        'v':  [float(row[1] / max(row.sum(), 1e-12)) for row in counts],
    })

with open('forest_model_e.json', 'w') as f:
    json.dump({'features': feature_names, 'trees': trees}, f)


# Traverses every tree and averages the leaf values, mirroring what the
# browser does
def score_with_json(values):
    total = 0
    for tree in trees:
        node = 0
        while tree['f'][node] != -2:
            if values[tree['f'][node]] <= tree['th'][node]:
                node = tree['L'][node]
            else:
                node = tree['R'][node]
        total += tree['v'][node]
    return total / len(trees)


# Numerical parity against scikit-learn is verified before exit
random_data = np.random.RandomState(0).rand(50, len(feature_names)) * 40
test_rows = pd.DataFrame(random_data, columns=feature_names).astype(np.float64)

python_scores = model.predict_proba(test_rows)[:, 1]
json_scores = np.array([score_with_json(row) for row in test_rows.values])
biggest_difference = np.abs(python_scores - json_scores).max()

print(f"Parity verified. Maximum deviation: {biggest_difference}")
print(f"Saved forest_model_e.json ({len(trees)} trees, {len(feature_names)} features)")
print("\nCopy forest_model_e.json into extension/model/ and reload the extension.")
