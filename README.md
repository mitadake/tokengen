# TokenGen: Visualize Token Prediction Evolution & Attention Dynamics
TokenGen is an interactive visualization tool for exploring how transformer-based language models predict tokens layer-by-layer. It provides insights into the evolution of token probabilities and attention head dynamics across transformer blocks. Built with Streamlit and Plotly.
Website: [TokenGen](https://tokengen.streamlit.app/)

[![Video Thumbnail](https://img.youtube.com/vi/9_k6cHsg9hI/0.jpg)](https://www.youtube.com/watch?v=9_k6cHsg9hI)

# Features
- **Token Probability Timeline**: Track how token predictions evolve through each transformer layer.
- **Attention Heatmaps**: Visualize aggregated attention patterns across layers and tokens.
- **Head Clustering**: Discover patterns in attention heads using UMAP dimensionality reduction and K-means clustering.
- **Model Comparison**: Compare predictions and attention patterns between two models.
- **Contrastive Analysis**: Analyze model preferences between two tokens across layers.

# Installation
1. Clone repository:
   ```bash
   git clone https://github.com/mitadake/tokengen.git
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
1. Token Prediction:<br>
![Predicted token](https://github.com/mitadake/tokengen/blob/main/src/pred_token.png)<br>
Predicts the token based on the model chosen.

2. Probability Timeline:<br>
![Probability Timeline](https://github.com/mitadake/tokengen/blob/main/src/prob_timeline.png)<br>
Shows how different token probabilities change through successive transformer layers.

3. Attention Heatmap:<br>
![Attention Heatmap](https://github.com/mitadake/tokengen/blob/main/src/atten_heatmap.png)<br>
Displays layer-wise attention patterns aggregated across all attention heads.

4. Head Clustering:<br>
![Head Clustering](https://github.com/mitadake/tokengen/blob/main/src/head_clustering.png)<br>
Groups of similar attention heads using K-Means to reveal functional patterns.

# Supported Models
- GPT-2 (base, medium)
- DistilGPT-2
- OPT-1.3b

_Note: Larger models require more memory and GPU resources. But a similar visualization can be done for them._

# Example Analysis
Try the default prompt: _"The world is full of amazing"_
1. Observe probability shifts:
- See how _"things"_ overtakes _"people"_ in later layers for GPT-2 medium model.
- Notice how grammatical tokens remain strong throughout.
2. Analyze attention patterns:
- See how early layers focus on determiners (_"The"_).
- Notice later layers attending to descriptive words (_"amazing"_).
3. Compare models:
- Try GPT-2 vs. OPT-1.3b.
- Observe different attention allocation strategies.

# Contrastive Mode
Compare how two tokens fare across layers:
1. Enable "Contrastive Explanation Mode".
2. Enter tokens (e.g., _"people"_ vs _"things"_).
3. See which layers prefer each token.

![Contrastive Analysis](https://github.com/mitadake/tokengen/blob/main/src/token_diff.png)

# Notes
- First run will download selected models.
- Loading the model may take time.
- Clear the cache after analysis of the two/ three models. 
- Work to be done: optimize the inference time and model loading using open-source tools like unsloth and onnx.
- Future work: Visualization support for custom models uploaded or from Hugging Face by the user.

# Contributing
Contributions welcome! Please open an issue first to discuss proposed changes.

# License
MIT License

# Acknowledgements
- Built with [Hugging Face Transformers](https://huggingface.co/)
- Visualization by [Plotly](https://plotly.com/)
- Dimensionality reduction with [UMAP](https://umap-learn.readthedocs.io/)
- Inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/)

<a href="https://github.com/mitadake/tokengen" target="_blank">
  <img src="https://img.shields.io/badge/GitHub-View%20Source-brightgreen?style=flat-square" alt="GitHub View Source">
</a>
<a href="https://tokengen.streamlit.app/" target="_blank">
  <img src="https://img.shields.io/badge/Open%20in%20Streamlit-black?style=flat-square" alt="Open in Streamlit">
</a>

