# TokenGen: Visualize Token Prediction Evolution & Attention Dynamics
TokenGen is an interactive visualization tool for exploring how transformer-based language models predict tokens layer-by-layer. It provides insights into the evolution of token probabilities and attention head dynamics across transformer blocks. Built with Streamlit and Plotly.
Webbsite: [TokenGen](https://tokengen.streamlit.app/)

# Features
- **Token Probability Timeline**: Track how token predictions evolve through each transformer layer.
- **Attention Heatmaps**: Visualize aggregated attention patterns across layers and tokens.
- **Head Clustering**: Discover patterns in attention heads using UMAP dimensionality reduction and K-means clustering.
- **Model Comparison**: Compare predictions and attention patterns between two models.
- **Contrastive Analysis**: Analyze model preferences between two tokens across layers.

# Installation
1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/tokengen.git
   cd tokengen
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
  
# Usage
1. Launch Streamlit app:
   ```bash
   streamlit run token_prob_timeline.py
2. In the browser:
   - Select a model (or compare two)
   - Enter your text prompt
   - Adjust visualization parameters
   - Explore different tabs and visualizations

# Key Visualizations
1. Probability Visualization
   
