import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import plotly.express as px

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("File Uploaded Successfully!")
        return data
    else:
        st.info("Please upload a dataset.")
        return None

# Function to preprocess data
def preprocess_data(data):
    scaler = StandardScaler()
    features = data.drop(['Class'], axis=1, errors='ignore')  # Exclude target column
    scaled_features = scaler.fit_transform(features)
    return scaled_features

# Function to inject synthetic anomalies
def inject_synthetic_anomalies(data, num_anomalies=10, anomaly_factor=10):
    st.info(f"Injecting {num_anomalies} synthetic anomalies into the dataset...")
    anomalies = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0), size=(num_anomalies, data.shape[1]))
    data_with_anomalies = np.vstack([data, anomalies])
    return data_with_anomalies

# Function to build graph using Nearest Neighbors with optimizations
def build_graph(data, n_neighbors=5, edge_threshold=0.1):
    st.info("Building graph using K-Nearest Neighbors (Sparse and Optimized)...")
    G = nx.Graph()

    # Use NearestNeighbors with parallel jobs for faster computation
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
    nn.fit(data)
    distances, indices = nn.kneighbors(data)

    # Add edges with weight above a threshold
    edges = []
    for i, neighbors in enumerate(indices):
        for j, neighbor in enumerate(neighbors):
            if i != neighbor:  # Exclude self-loops
                weight = 1 / (distances[i][j] + 1e-5)  # Inverse distance to avoid division by zero
                if weight > edge_threshold:  # Add only significant edges
                    edges.append((i, neighbor, weight))

    # Add edges to graph
    G.add_weighted_edges_from(edges)
    st.success(f"Graph built successfully with {len(G.nodes())} nodes and {len(G.edges())} edges (filtered by threshold).")
    return G

# Function to visualize graph using Plotly (Dynamic Visualization)
def plot_interactive_graph(G, anomalies):
    st.subheader("Interactive Graph Visualization")
    pos = nx.spring_layout(G, seed=42)
    
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_x.append(None)
        edge_y.append(None)

    node_x, node_y, node_color = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append("red" if node in anomalies else "blue")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='gray', width=0.5)))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10, color=node_color), text=list(G.nodes()), hoverinfo='text'))
    fig.update_layout(title="Transaction Graph (Red = Anomalies)", showlegend=False)
    st.plotly_chart(fig)

# Function to detect anomalies
def detect_anomalies(G, threshold=1):
    st.info("Detecting anomalies...")
    anomalies = [node for node, degree in G.degree() if degree <= threshold]
    st.success(f"Found {len(anomalies)} anomalies (nodes with degree <= {threshold}).")
    return anomalies

# Function to display graph metrics
def display_graph_metrics(G):
    st.subheader("Graph Metrics Dashboard")
    centrality = nx.degree_centrality(G)
    clustering = nx.clustering(G)
    connected_components = nx.number_connected_components(G)
    
    st.write("### Top Nodes by Degree Centrality")
    top_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    st.write(pd.DataFrame(top_centrality, columns=["Node", "Centrality"]))

    st.write("### Average Clustering Coefficient:", nx.average_clustering(G))
    st.write("### Number of Connected Components:", connected_components)

# Function for feature-wise anomaly detection
def feature_wise_anomalies(data):
    st.subheader("Feature-wise Anomaly Detection")
    anomaly_percentages = {}
    for i, feature in enumerate(data.T):
        q1, q3 = np.percentile(feature, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        anomalies = ((feature < lower_bound) | (feature > upper_bound)).sum()
        anomaly_percentages[f"Feature {i + 1}"] = anomalies / len(feature) * 100

    st.write("Anomaly Percentages per Feature:")
    st.write(anomaly_percentages)

    st.subheader("Pie Chart of Feature-Wise Anomalies")
    labels = list(anomaly_percentages.keys())
    values = list(anomaly_percentages.values())

    fig = px.pie(
        names=labels,
        values=values,
        title="Percentage of Anomalies in each Feature",
        hole=0.4 #To make it a donut chart
    )
    st.plotly_chart(fig)

def detect_communities(G):
    st.subheader("Community Detection")
    from networkx.algorithms.community import greedy_modularity_communities

    communities = list(greedy_modularity_communities(G))
    st.success(f"Detected {len(communities)} communities.")

    # Plot the graph with different edge styles for communities
    fig = go.Figure()
    pos = nx.spring_layout(G, seed=42)

    for i, community in enumerate(communities):
        edge_x, edge_y = [], []
        for node in community:
            for neighbor in G.neighbors(node):
                if neighbor in community:
                    x0, y0 = pos[node]
                    x1, y1 = pos[neighbor]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=2, dash='dash'), name=f'Community {i+1}'))

    node_x, node_y = zip(*pos.values())
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', marker=dict(size=10), name='Nodes'))

    fig.update_layout(title="Community Detection", showlegend=True)
    st.plotly_chart(fig)    

# Main Streamlit app
def main():
    st.set_page_config(page_title="Graph-Based Anomaly Detection", layout="wide")
    st.title("Graph-Based Anomaly Detection using NetworkX")

    # Sidebar controls
    st.sidebar.header("User Inputs")
    n_neighbors = st.sidebar.slider("Number of Nearest Neighbors", 1, 20, 5)
    degree_threshold = st.sidebar.slider("Anomaly Degree Threshold", 0, 10, 1)
    edge_threshold = st.sidebar.slider("Edge Weight Threshold", 0.0, 1.0, 0.1, step=0.05)
    num_synthetic_anomalies = st.sidebar.slider("Number of Synthetic Anomalies", 0, 50, 10)
    anomaly_factor = st.sidebar.slider("Anomaly Factor", 1, 20, 10)

    # Load data
    data = load_data()
    if data is not None:
        st.subheader("Dataset Overview")
        st.dataframe(data.head())
        st.write("Shape of the dataset:", data.shape)

        # Ensure anomalies are included
        if 'Class' in data.columns:
            anomalies_data = data[data['Class'] == 1]
            normal_data = data[data['Class'] == 0]
            sample_size = min(len(normal_data), 500 - len(anomalies_data))
            normal_data_sample = normal_data.sample(n=sample_size, random_state=42)
            combined_data = pd.concat([anomalies_data, normal_data_sample], axis=0).reset_index(drop=True)
            st.success(f"Using {len(combined_data)} rows, including {len(anomalies_data)} anomalies.")
        else:
            st.warning("No 'Class' column found. Using first 500 rows as default.")
            combined_data = data.iloc[:500]

        # Preprocess data
        processed_data = preprocess_data(combined_data)

        # Inject synthetic anomalies
        processed_data = inject_synthetic_anomalies(processed_data, num_anomalies=num_synthetic_anomalies, anomaly_factor=anomaly_factor)

        # Build graph
        G = build_graph(processed_data, n_neighbors=n_neighbors, edge_threshold=edge_threshold)
        st.write(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


        # Display graph metrics
        display_graph_metrics(G)

        # Detect anomalies
        anomalies = detect_anomalies(G, threshold=degree_threshold)
        st.write("Anomalous Nodes:", anomalies)

        st.write("Feature wise Anomalies Detection")
        feature_wise_anomalies(processed_data)

        # Visualize graph interactively
        plot_interactive_graph(G, anomalies)
        detect_communities(G)
        # Additional Visualizations
        st.subheader("Additional Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Feature Distributions")
            for col in combined_data.columns[:5]:
                fig, ax = plt.subplots()
                sns.histplot(combined_data[col], kde=True, ax=ax)
                plt.title(f"Distribution of {col}")
                st.pyplot(fig)

        with col2:
            st.write("### Anomaly Count Plot")
            if 'Class' in combined_data.columns:
                fig = px.histogram(combined_data, x='Class', title="Class Distribution (Fraud/Non-Fraud)")
                st.plotly_chart(fig)
            else:
                st.warning("No 'Class' column found in the dataset for anomaly labels.")

if __name__ == "__main__":
    main() 
