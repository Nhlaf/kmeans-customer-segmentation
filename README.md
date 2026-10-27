# Customer Segmentation Using K-Means

## Project Overview

This project applies unsupervised machine learning to segment customers based on their annual income and spending scores.

The goal is to identify groups of customers with similar characteristics and demonstrate how clustering can support data-driven business decisions.

## Business Problem

Businesses often serve customers with different spending behaviors and income levels. Treating every customer the same can make marketing less effective.

This project uses K-Means clustering to identify customer segments and explore how the results could inform targeted marketing strategies.

## Objectives

- Explore and understand customer data.
- Check for missing values and duplicate records.
- Scale numerical features before clustering.
- Use the Elbow Method to investigate a suitable number of clusters.
- Train a K-Means clustering model.
- Visualize customer segments and summarize their characteristics.
- Save the clustering results for further analysis.

## Dataset

The project uses the Mall Customers dataset, containing customer demographics, annual income, and spending scores.

Source: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

The model uses these two features:

- `Annual Income (k$)`
- `Spending Score (1-100)`

CustomerID is excluded from the clustering features because it is an identifier rather than a behavioral measurement.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- K-Means clustering

## Methodology

1. Load the customer dataset.
2. Inspect the data and check its quality.
3. Select the numerical features.
4. Remove records with missing or invalid selected features.
5. Standardize the features using StandardScaler.
6. Generate an Elbow Method plot.
7. Train K-Means using five clusters as an initial choice.
8. Analyze cluster sizes and average customer characteristics.
9. Save visualizations and processed results.

## Results

The initial model processed 200 customer records and identified five clusters.

| Cluster | Customers | Average annual income ($k) | Average spending score |
|---|---:|---:|---:|
| 0 | 81 | 55.30 | 49.52 |
| 1 | 39 | 86.54 | 82.13 |
| 2 | 22 | 25.73 | 79.36 |
| 3 | 35 | 88.20 | 17.11 |
| 4 | 23 | 26.30 | 20.91 |

The clusters represent different combinations of income and spending behavior. These patterns may help businesses explore premium customer engagement, targeted promotions, and customer retention strategies.

Five clusters were used for the initial analysis. The Elbow Method plot should be reviewed to assess whether this choice is appropriate.

## Project Outputs

The `outputs/` folder contains:

- `elbow_method.png` — inertia across different cluster counts.
- `customer_clusters.png` — visualization of customer segments.
- `clustered_customers.csv` — customer records with assigned cluster labels.
- `cluster_profiles.csv` — average income and spending score for each cluster.

## Project Structure

```text
.
├── data/
│   └── customers.csv
├── outputs/
│   ├── clustered_customers.csv
│   ├── cluster_profiles.csv
│   ├── customer_clusters.png
│   └── elbow_method.png
├── main.py
├── README.md
└── requirements.txt
```

## How to Run the Project

Python 3.10 or later is recommended.

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python main.py
```

The included dataset is at `data/customers.csv`. The analysis saves updated results and visualizations in the `outputs/` directory.

## Limitations and Future Improvements

- Evaluate the number of clusters using the Elbow Method and additional validation metrics such as silhouette score.
- Compare alternative feature combinations.
- Investigate the effect of different preprocessing choices.
- Add automated tests and clearer error handling.
- Develop an interactive dashboard for exploring customer segments.

## Conclusion

This project demonstrates a practical machine learning workflow, from data exploration and preprocessing to clustering, visualization, and business interpretation.

It provides a foundation for further experimentation with unsupervised learning and customer analytics.