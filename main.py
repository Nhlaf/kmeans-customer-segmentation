
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "customers.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the customer dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Please place customers.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())


# --------------------------------------------------
# 3. Prepare features for clustering
# --------------------------------------------------

income_col = "Annual Income (k$)"
spending_col = "Spending Score (1-100)"

required_columns = [income_col, spending_col]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Expected columns not found: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Convert the selected features to numeric values.
features = df[required_columns].apply(
    pd.to_numeric, errors="coerce"
)

# Remove rows with missing or invalid feature values.
valid_rows = features.notna().all(axis=1)
df = df.loc[valid_rows].copy()
features = features.loc[valid_rows].copy()

if len(features) < 10:
    raise ValueError(
        "At least 10 valid customer records are needed "
        "for this demonstration."
    )

print("\nValid customer records:", len(features))


# --------------------------------------------------
# 4. Scale the features
# --------------------------------------------------

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print("\nFeatures scaled successfully.")


# --------------------------------------------------
# 5. Find a suitable number of clusters
#    using the Elbow Method
# --------------------------------------------------

inertias = []
cluster_range = range(1, 11)

for k in cluster_range:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_features)
    inertias.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(
    list(cluster_range),
    inertias,
    marker="o"
)

plt.title("Elbow Method for K-Means")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(cluster_range))
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "elbow_method.png",
    dpi=300
)
plt.close()

print("\nElbow chart saved to outputs/elbow_method.png")


# --------------------------------------------------
# 6. Train the K-Means model
# --------------------------------------------------

# Start with 5 clusters as a candidate.
# Review the elbow chart to decide whether this is suitable.
BEST_K = 5

kmeans = KMeans(
    n_clusters=BEST_K,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(scaled_features)

print(f"\nK-Means model trained with {BEST_K} clusters.")


# --------------------------------------------------
# 7. Analyze the customer groups
# --------------------------------------------------

cluster_profiles = df.groupby("Cluster")[
    required_columns
].mean().round(2)

cluster_sizes = df["Cluster"].value_counts().sort_index()

print("\n========== CLUSTER SIZES ==========")
print(cluster_sizes)

print("\n========== AVERAGE CUSTOMER PROFILE ==========")
print(cluster_profiles)


# --------------------------------------------------
# 8. Visualize the customer clusters
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x=income_col,
    y=spending_col,
    hue="Cluster",
    palette="viridis",
    s=90
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "customer_clusters.png",
    dpi=300
)
plt.close()

print("\nCluster chart saved to outputs/customer_clusters.png")


# --------------------------------------------------
# 9. Save the results
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "clustered_customers.csv",
    index=False
)

cluster_profiles.to_csv(
    OUTPUT_DIR / "cluster_profiles.csv"
)

print("\n========== PROJECT COMPLETED ==========")
print("Clustered customer data saved.")
print("Cluster profiles saved.")
print("Visualizations saved in the outputs folder.")
