# **Graph Construction Method (Adapted from Kim et al., 2025, Water Research)**

Below is the **exact procedure** for converting your dataset into graph inputs for a Graph‐Based Neural Network (GNN), **following the method used in the cited Water Research paper**. This approach models **microbial communities** alongside any relevant environmental variables (e.g., VFAs) **as a single integrated network**.

---

## 1. Preprocessing & Filtering

1. **Filter Low‐Abundance OTUs**  
   - Discard rare OTUs that do not meet a minimum prevalence threshold (e.g., >1% total abundance and present in ≥5% of samples).  
   - This step reduces noise from extremely rare taxa.

2. **Taxonomic Merging**  
   - Merge the remaining OTUs **at the family level** (or another chosen taxonomic level used in the paper).  
   - The end result is a reduced set of taxonomic groups (e.g., tens of families) instead of 1800+ OTUs.

3. **Transform & Normalize**  
   - Apply **double‐square‐root** (or log) to microbial relative abundances.  
   - Min‐max or otherwise **scale** any measured variables (e.g., VFAs), so they align with the model’s expected input range.

---

## 2. Node & Edge Definition

1. **Nodes**  
   - **Each microbial family** becomes one node in the graph.  
   - **Environmental variables** (e.g., acetic acid, propanoic acid) can also be included as separate nodes, to capture microbe–chemical interactions.

2. **Edges (Mantel Test for Significant Interactions)**  
   - **Compute distance matrices** for microbial families (e.g., Bray–Curtis) and for environmental variables (e.g., Euclidean).  
   - **Run Mantel tests** (999 permutations) to identify significantly correlated pairs of nodes (p < 0.05).  
   - **Draw an edge** between two nodes **only** if the Mantel correlation is significant.  
   - Store the correlation value (or p‐value) as an **edge attribute**.

By doing so, you end up with a network that captures **meaningful interactions** among families and between families & environmental variables (VFAs).

---

## 3. Building Individual Graphs

1. **One Graph per Sample**  
   - For each of your 60 samples, **use the same node set and edges** (derived from the overall significant interactions).  
   - **Node Features:** the **abundances** (or scaled values) measured **in that specific sample**.  
   - If you have a time series, you may similarly build one graph for each sample‐time combination, repeating the process.

2. **Graph Labels (Optional)**  
   - If you are predicting a property (e.g., gas production), assign **that sample’s measurement** as the label for that graph.

Hence, you will have **60 graph instances** (one per sample), each with identical structure but **differing node feature values** that reflect the sample measurements.

---

## 4. GNN Training (As in the Paper)

1. **Input to the GCN**  
   - Each graph is described by:  
     - **Adjacency** (edges) from the Mantel‐based network.  
     - **Node features** (abundance values, scaled environment data).  

2. **Model Architecture**  
   - They used a **Graph Convolutional Network** with convolutional layers that process node embeddings, followed by fully connected layers for final predictions of:  
     - Microbial abundances at the next time step, and  
     - Biogas production rate.

3. **Loss Function & Metrics**  
   - In the paper, they optimized with **Mean Squared Error (MSE)** and reported **R²** for both microbial abundances and gas production.

4. **Interpretation**  
   - After training, they applied **GNNExplainer** to identify which nodes (families) and edges (interactions) most affected the predictions.

---

## 5. Final Summary of the Procedure

1. **Filter & Merge OTUs** → reduce 1800+ OTUs to a manageable set of families.  
2. **Apply Transformations** (double‐square‐root, scaling).  
3. **Build a Single Interaction Network**  
   - Use Mantel tests to link families and environment variables (p < 0.05).  
4. **Create One Graph per Sample**  
   - Same edges for all; node features differ by sample.  
5. **Train the GCN** to predict target variables (e.g., next‐step microbial abundances, gas output).

This reproduces **the core method** from Kim et al. (2025) and ensures your pipeline **mirrors** their graph‐based deep learning approach for **microbiome** and **biogas** predictions.
