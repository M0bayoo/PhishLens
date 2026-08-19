# Generates a feature-importance chart from the trained model.
#
# Importance is mean decrease in impurity: how much each feature reduced
# class mixing across the forest's splits. Features that fire on a small
# subset of URLs score low on this measure even when decisive for those
# cases, so it reflects breadth of use rather than value per use.
#
# Run after 04_train_model.py.

import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

saved = joblib.load('models/model_final.joblib')
model = saved['model']
feature_names = saved['features']

# Sorted ascending so the largest contributor appears at the top of a
# horizontal bar chart
importance = pd.Series(model.feature_importances_,
                       index=feature_names).sort_values()

figure, axis = plt.subplots(figsize=(8, 9))
importance.plot(kind='barh', ax=axis, color='teal')
axis.set_title(f'PhishLens v2 - Feature Importance '
               f'({len(feature_names)} features)')
axis.set_xlabel('Mean decrease in impurity')
figure.tight_layout()
figure.savefig('feature_importance.png', dpi=150)

print(f"Saved feature_importance.png ({len(feature_names)} features)")

print("\nTop 10 features:")
print(importance.tail(10)[::-1].round(4).to_string())

# The pre-fix model was dominated by path_length at roughly twenty times
# any other feature, which was the dataset-bias fingerprint. Reporting its
# share here gives a directly comparable number for the write-up.
top_share = importance.iloc[-1] / importance.sum() * 100
runner_up = importance.iloc[-2] / importance.sum() * 100
print(f"\nTop feature share      : {importance.index[-1]} "
      f"({top_share:.1f}% of total importance)")
print(f"Second feature share   : {importance.index[-2]} "
      f"({runner_up:.1f}%)")
print(f"Ratio top:second       : {importance.iloc[-1] / importance.iloc[-2]:.1f}x")

if 'path_length' in importance.index:
    path_share = importance['path_length'] / importance.sum() * 100
    print(f"path_length share      : {path_share:.1f}%")
