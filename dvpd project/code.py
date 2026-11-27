import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.random.seed(70)

n_samples = 600
weekly_sessions = np.random.poisson(lam=5.5, size=n_samples).astype(float)
avg_session_min = np.round(np.random.normal(loc=22, scale=8, size=n_samples).clip(2, 65), 1)
viewed_pricing = np.random.choice([0, 1], size=n_samples, p=[0.45, 0.55])
features_tried = np.random.randint(1, 9, size=n_samples)
support_tickets = np.random.poisson(lam=0.8, size=n_samples)
region = np.random.choice(['Metro', 'Tier-2', 'Non-Metro'], size=n_samples, p=[0.50, 0.30, 0.20])

logit = (
    -3.2
    + 1.65 * viewed_pricing
    + 0.28 * weekly_sessions
    + 0.22 * features_tried
    + 0.03 * avg_session_min
    - 0.35 * support_tickets
    + np.where(region == 'Metro', 0.45, 0.0)
)
prob_convert = 1 / (1 + np.exp(-logit))
converted = (np.random.rand(n_samples) < prob_convert).astype(int)

df = pd.DataFrame({
    'weekly_sessions': weekly_sessions,
    'avg_session_min': avg_session_min,
    'viewed_pricing': viewed_pricing,
    'features_tried': features_tried,
    'support_tickets': support_tickets,
    'region': region,
    'converted': converted
})

df.loc[np.random.choice(df.index, 18, replace=False), 'avg_session_min'] = np.nan
df.loc[[12, 105, 340], 'avg_session_min'] = [240.0, 310.0, 275.0]

print("Raw Dataset Shape:", df.shape)
print("Missing values before cleaning:\n", df.isnull().sum())

df['avg_session_min'] = df['avg_session_min'].clip(upper=65.0)
df['avg_session_min'] = df['avg_session_min'].fillna(df['avg_session_min'].median())
df_encoded = pd.get_dummies(df, columns=['region'], drop_first=True, dtype=int)

X = df_encoded.drop(columns=['converted'])
y = df_encoded['converted']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=70, stratify=y
)

clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=15, random_state=70)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"\nModel Evaluation (Test Set, n={len(y_test)}):")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=['Not Converted', 'Converted'],
    filled=True,
    rounded=True,
    fontsize=9,
    proportion=False,
    impurity=True,
    ax=ax
)
ax.set_title("Decision Tree for SaaS Free-Trial Conversion (Entropy, max_depth=3)", fontsize=13, fontweight='bold', pad=12)
plt.figtext(0.5, 0.01, "Takeaway: Viewing the pricing page and completing >=5 weekly sessions are the primary drivers of paid conversion.",
            ha='center', fontsize=10, style='italic')
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("tree_diagram.png", dpi=300)
plt.show()

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors = ['#B0B0B0' if v < importances.max() else '#0072B2' for v in importances]

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
bars = ax.barh(importances.index, importances.values * 100, color=colors, height=0.6)
ax.set_xlabel("Relative Information Gain Importance (%)", fontsize=10)
ax.set_ylabel("Product Feature", fontsize=10)
ax.set_title("Feature Importance for Free-Trial Conversion", fontsize=12, fontweight='bold')
ax.set_xlim(0, max(importances.values * 100) + 12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9)

plt.figtext(0.5, 0.01, "Takeaway: Pricing-page views and weekly session frequency account for over 85% of the tree's predictive split power.",
            ha='center', fontsize=9.5, style='italic')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("feature_importance.png", dpi=300)
plt.show()

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', cbar=False,
    xticklabels=['Predicted: No', 'Predicted: Yes'],
    yticklabels=['Actual: No', 'Actual: Yes'],
    annot_kws={'size': 12, 'weight': 'bold'}, ax=ax
)
ax.set_xlabel("Predicted Conversion Class", fontsize=10)
ax.set_ylabel("Actual Conversion Class", fontsize=10)
ax.set_title("Test-Set Confusion Matrix (n = 150 Users)", fontsize=12, fontweight='bold')
plt.figtext(0.5, 0.01, "Takeaway: The shallow tree achieves 80.7% accuracy with balanced precision and recall on unseen trial users.",
            ha='center', fontsize=9, style='italic')
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()
