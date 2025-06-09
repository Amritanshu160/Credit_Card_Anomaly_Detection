# 🧠 Graph-Based Credit Card Anomaly Detection

This is an interactive **Streamlit** web application for **anomaly detection** using **graph theory**. It transforms datasets into a graph using **K-Nearest Neighbors**, visualizes relationships between transactions, and identifies anomalous nodes based on graph properties like degree, centrality, and clustering.

---

## 🚀 Features

- Upload any CSV dataset (with or without a `Class` column)
- Preprocessing and standardization using `StandardScaler`
- Synthetic anomaly injection (adjustable)
- K-Nearest Neighbors graph construction with threshold filtering
- Graph metrics dashboard: centrality, clustering, connectivity
- Community detection using greedy modularity
- Interactive graph visualization using Plotly
- Feature-wise anomaly detection with IQR method and pie chart
- Sidebar UI for custom parameter control

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit, Plotly, Seaborn, Matplotlib
- **Backend**: Python, NetworkX, scikit-learn, NumPy, pandas

---

## 📁 File Structure

```
Graph-Anomaly-App/
├── app.py                    # Main Streamlit application
├── README.md                 # Documentation
```

---

## 📦 Dependencies

Install all necessary libraries with:

```bash
pip install streamlit pandas networkx matplotlib seaborn plotly scikit-learn numpy
```

---

## ▶️ Running the App

Start the app using Streamlit:

```bash
streamlit run app.py
```

Then open the browser window that appears or navigate to `http://localhost:8501`.

---

## 📊 How It Works

### 1. **Data Input**
- Upload a `.csv` file with numeric columns.
- If `Class` column is present (0 = normal, 1 = anomaly), it will be used to balance the dataset.
- Otherwise, first 500 rows are used.

### 2. **Preprocessing**
- Standard scaling is applied to numeric features.
- Optional synthetic anomalies are injected into the data.

### 3. **Graph Construction**
- A graph is built using **KNN**.
- Edges are created based on inverse distance and filtered by a threshold.
- Nodes represent data points.

### 4. **Anomaly Detection**
- Nodes with degree below a certain threshold are marked as anomalous.
- Feature-wise IQR-based anomaly detection is also shown via pie chart.

### 5. **Visualizations**
- Interactive graph of transactions (anomalies in red)
- Community detection view
- Histograms of selected features
- Class distribution chart

---

## ⚙️ Sidebar Controls

- **Number of Nearest Neighbors**: 1–20
- **Anomaly Degree Threshold**: 0–10
- **Edge Weight Threshold**: 0.0–1.0
- **Synthetic Anomalies**: 0–50
- **Anomaly Factor**: 1–20

---

## 🧠 Community Detection

Uses `greedy_modularity_communities` from NetworkX to find clusters within the graph. Results are visualized interactively with distinct edge groups for each community.

---

## 🧑‍💻 Author

**Amritanshu Bhardwaj**  
Department of AI & Data Science  
BMS College of Engineering

---

