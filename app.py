import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import io

st.set_page_config(layout="wide")

# Header and branding
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center;'>
  <h1 style='margin-bottom: 0;'>Query Embedding Visualizer</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style='margin-top: 0;'>About This Tool</h3>
<p style='font-size: 1rem;'>
This tool helps SEOs and Relevance Engineers visualize how closely queries (or user-written content) relate to a Head Query and Simulated Sub Queries. By mapping high-dimensional embeddings in a 2D/3D space, you can evaluate cosine similarity, detect intent clusters, and better support content and relevance strategies for increased visibility online.
</p>
---
""", unsafe_allow_html=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

uploaded_file = st.file_uploader("Upload a CSV file with queries (must include a 'query' column)", type="csv")

example_queries = [
    "How to choose a pediatric dentist for my child?",
    "Best children's dentists near me with high safety ratings",
    "Are there dentists specializing in anxious children?",
    "Pediatric dentist vs. general dentist for children: what's the difference?",
    "Top-rated pediatric dentists specializing in sedation dentistry",
    "Finding a child-friendly dentist covered by my insurance (Blue Cross Blue Shield)",
    "How to find a dentist who understands my 3-year-old's needs?",
    "Reviews of family dentists with a gentle approach to children",
    "What questions should I ask a potential dentist for my child?",
    "Safe dental practices for children: what to look for",
    "How much does a typical dental checkup for a child cost?",
    "Finding a dentist with experience handling children with special needs (autism)",
    "Best pediatric dentists in [City, State] with flexible appointment scheduling",
    "Compare prices of pediatric dental services in my area",
    "Finding a dentist who uses modern and safe equipment for children",
    "How to prepare my child for their first dentist appointment"
]

head_query = st.text_input("Enter the head query", "how to find safe and kind dentist for my child")

use_example = st.checkbox("Use example fan-out queries")
user_queries = st.text_area("Enter up to 28 sub-queries, one per line")
user_labels = st.text_area("(Optional) Label each sub-query, one per line (e.g. Awareness, Consideration, Decision)")
st.caption("ℹ️ Ensure labels match the number and order of sub-queries to label them correctly.")
user_content = st.text_area("Optional: Paste your content to compare (will be visualized separately)", height=200)

projection_mode = st.radio("Select projection mode", options=["3D", "2D"], index=0)

trigger = st.button("Generate Visualizations and Table")

if trigger:
    queries = []
    labels = []
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "query" in df.columns:
            queries = df["query"].dropna().tolist()
        else:
            st.error("CSV must have a column named 'query'")
    elif use_example:
        queries = example_queries
    elif user_queries:
        queries = [q.strip() for q in user_queries.splitlines() if q.strip()]

    queries = queries[:28]

    if head_query and queries:
        all_queries = [head_query] + queries
        sub_labels = [f"Sub-query {i+1}" for i in range(len(queries))]

        if user_labels:
            custom_labels = [l.strip() for l in user_labels.splitlines() if l.strip()]
            if len(custom_labels) == len(queries):
                sub_labels = custom_labels
            else:
                st.warning("The number of labels doesn’t match the number of sub-queries. Default labels will be used.")

        labels = ["Head Query"] + sub_labels

        if user_content:
            all_queries.append(user_content)
            labels.append("Your Content")

        embeddings = model.encode(all_queries)

        reducer = umap.UMAP(n_components=3 if projection_mode == "3D" else 2, random_state=42)
        emb_proj = reducer.fit_transform(embeddings)

        df_plot = pd.DataFrame(emb_proj, columns=["x", "y"] if projection_mode == "2D" else ["x", "y", "z"])
        df_plot["label"] = labels
        df_plot["query"] = all_queries

        similarities = cosine_similarity([embeddings[0]], embeddings[1:1+len(queries)])[0]
        sim_labels = ["-"] + [f"{sim:.2f}" for sim in similarities]
        if user_content:
            user_sim = cosine_similarity([embeddings[0]], [embeddings[-1]])[0][0]
            sim_labels.append(f"{user_sim:.2f}")
        df_plot["Similarity to Head"] = sim_labels

        k = min(len(df_plot), 5)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df_plot["cluster"] = kmeans.fit_predict(embeddings)

        selected_index = st.selectbox("Highlight a specific query (enlarges the point)", options=range(len(df_plot)), format_func=lambda i: df_plot.iloc[i]["query"])
        df_plot["size"] = [10 if i == selected_index else 6 for i in range(len(df_plot))]

        if projection_mode == "3D":
            fig = px.scatter_3d(df_plot, x="x", y="y", z="z", hover_name="query", hover_data={"Similarity to Head": True, "label": True, "cluster": True}, color="label", size="size", title="Semantic Embedding Visualization")
        else:
            fig = px.scatter(df_plot, x="x", y="y", hover_name="query", hover_data={"Similarity to Head": True, "label": True, "cluster": True}, color="label", size="size", title="Semantic Embedding Visualization")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Clusters** are automatically calculated groupings of semantically similar queries. Use these clusters to inform how you might group content into topical pages, hub structures, or internal linking strategies.")

        st.subheader("Embedding Table (Preview of Exportable Data)")
        embedding_df = pd.DataFrame(embeddings)
        embedding_df.insert(0, "query", all_queries)
        embedding_df.insert(1, "label", labels)
        embedding_df.insert(2, "Similarity to Head", sim_labels)
        embedding_df.insert(3, "cluster", df_plot["cluster"])
        st.dataframe(embedding_df)

        st.subheader("Export Options")
        export_format = st.radio("Choose format to export:", ["CSV", "Chart HTML"], horizontal=True)

        export_trigger = st.button("Download Now")

        if export_trigger:
            if export_format == "CSV":
                csv_data = embedding_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download CSV", data=csv_data, file_name="query_embeddings.csv", mime="text/csv")
            else:
                html_str = fig.to_html(full_html=True, include_plotlyjs='cdn')
                st.download_button("⬇ Download HTML Chart", data=html_str, file_name="embedding_chart.html", mime="text/html")
