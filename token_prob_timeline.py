import torch
import torch._dynamo

from transformers import AutoModelForCausalLM, AutoTokenizer
import plotly.graph_objects as go
import streamlit as st
from umap.umap_ import UMAP
from sklearn.cluster import KMeans
import numpy as np

st.set_page_config(
    page_title="TokenGen",  
    page_icon="🧮", 
)

# Suppress compile errors silently
torch._dynamo.config.suppress_errors = True

st.title("🧮 Token Probability Timeline with Attention")
st.markdown("Visualize **token prediction evolution** and **attention heads** layer-by-layer.")
st.markdown("Note: Model Loading may take time. Clear cache for inference of multiple model.")
st.markdown("Use below example or write your own text to predict. I provide explanation of the example below.")
with st.expander("Example"):
    st.markdown("For example: 'The world is full of amazing', the attention graph shows word 'The' has higher attention (bright/warm colors). Why does model give attention to this word in comparison to others?")
    st.markdown("This can be due to lot of reasons like training data bias, positional anchoring. I like to think of it in the this way, let's image writing a sentence: 'The _____ is full of amazing _____.'")
    st.markdown("Even though 'amazing' is descriptive, your brain still checks the first word ('The') to ensure the sentence starts correctly. The model does something similar, it uses 'The' to anchor the sentence structure, even if the 'meaning' comes from other words.")
st.markdown("Key Takeaway: ****Attention isn't always about meaning****. It can reflect grammatical rules, positional habits, or training data quirks. Just like humans sometimes fixate on small words (e.g., 'a' vs. 'the'), the model does too even if it's not obvious to us!")
st.markdown("Now try running without 'the': 'World is full of amazing' and observe the difference.")


mode = st.radio("Mode", ["Single Model", "Compare Two Models"])

model_list = [
    "gpt2",
    "distilgpt2",
    "gpt2-medium",
    "facebook/opt-1.3b",
]

if mode == "Single Model":
    model_1_name = st.selectbox("Select Model", model_list, index=2)
    model_2_name = None
    contrast_mode = st.checkbox("Enable Contrastive Explanation Mode")
    if contrast_mode:
        st.markdown("Try 'people' and 'things' for default example. Probability difference across layers.")
        col1, col2 = st.columns(2)
        with col1:
            contrast_token_a = st.text_input("Token A", value="")
        with col2:
            contrast_token_b = st.text_input("Token B", value="")
else:
    col1, col2 = st.columns(2)
    with col1:
        model_1_name = st.selectbox("Model A", model_list, index=0)
    with col2:
        model_2_name = st.selectbox("Model B", model_list, index=2)

input_text = st.text_input("Prompt", "The world is full of amazing")
top_k = st.slider("Top-k Tokens", 1, 10, 5)
n_clusters = st.slider("Number of clusters", 2, 10, 5, key="n_clusters")
position_index = -1


# ==== Load and Compile Models ====
@st.cache_resource
def load_model(name):
    model = AutoModelForCausalLM.from_pretrained(
        name,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = torch.compile(model)
    except Exception as e:
        print(f"torch.compile() failed for {name}. Using normal mode.\nError: {e}")

    return model, tokenizer


if st.button("Run"):

    def get_outputs(model_name, tokenizer, input_ids):
        model, _ = load_model(model_name)
        with torch.no_grad():
            outputs = model(input_ids)
        return outputs

    def get_layerwise_probs(outputs, model_name, tokenizer, input_ids, position_index, top_k):
        final_logits = outputs.logits
        hidden_states = outputs.hidden_states
        lm_head_weights = load_model(model_name)[0].lm_head.weight

        top1_id = torch.argmax(final_logits[0, -1])
        top1_token = tokenizer.convert_ids_to_tokens([top1_id])[0].replace("Ġ", "").replace("▁", "")

        token_probs_by_layer = []
        all_tokens = set()

        for layer_hidden in hidden_states[1:]:
            token_hidden = layer_hidden[0, position_index]
            logits = torch.matmul(token_hidden, lm_head_weights.T)
            probs = torch.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, top_k)
            topk_tokens = [tokenizer.convert_ids_to_tokens([i])[0].replace("Ġ", "").replace("▁", "") for i in topk_indices]
            token_probs_by_layer.append({t: p.item() for t, p in zip(topk_tokens, topk_probs)})
            all_tokens.update(topk_tokens)

        return token_probs_by_layer, all_tokens, top1_token

    def build_plot(token_probs_by_layer, all_tokens, title):
        fig = go.Figure()
        layers = list(range(1, len(token_probs_by_layer)+1))
        for token in all_tokens:
            y_values = [layer_probs.get(token, 0.0) for layer_probs in token_probs_by_layer]
            fig.add_trace(go.Scatter(x=layers, y=y_values, mode="lines+markers", name=token))
        fig.update_layout(title=title, xaxis_title="Layer", yaxis_title="Probability", yaxis=dict(range=[0, 1]), legend_title="Tokens")
        return fig

    def build_combined_attention_heatmap(attentions, token_labels, token_index, model_name):
        # Stack attentions: shape [num_layers, num_heads, seq_len, seq_len] -> mean over heads -> [num_layers, seq_len]
        attn_tensor = torch.stack(attentions)  # [L, B, H, T, T]
        attn_tensor = attn_tensor[:, 0]  # remove batch dim [L, H, T, T]
        mean_attn = attn_tensor.mean(1)  # average over heads -> [L, T, T]

        layerwise = mean_attn[:, token_index, :]  # [L, T]
        z = layerwise.detach().cpu().numpy()
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=token_labels,
            y=[f"Layer {i+1}" for i in range(z.shape[0])],
            colorscale='Viridis'
        ))
        fig.update_layout(
            title=f"{model_name} - Combined Attention Heatmap (Token index {token_index})",
            xaxis_title="Token",
            yaxis_title="Layer",
            height=600
        )
        return fig

    def build_attention_clustering(attentions, token_labels, position_index, model_name):
        """Visualize attention head clustering using UMAP"""
        # Process attention patterns for all heads
        patterns = []
        head_labels = []
        
        # Extract attention patterns for the target position across all layers/heads
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: [batch, heads, seq_len, seq_len]
            layer_attn = layer_attn[0].cpu().numpy()   # remove batch dim
            seq_len = layer_attn.shape[-1]
            target_pos = position_index if position_index != -1 else seq_len - 1
            
            for head_idx in range(layer_attn.shape[0]):
                head_pattern = layer_attn[head_idx, target_pos]
                patterns.append(head_pattern)
                head_labels.append(f"L{layer_idx+1}H{head_idx+1}")

        # Reduce dimensionality with UMAP
        reducer = UMAP(n_components=2, random_state=42)
        embeddings = reducer.fit_transform(patterns)

        # Cluster using K-means
        # n_clusters = min(5, len(patterns))  # Default to 5 clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        # Create interactive plot
        fig = go.Figure()
        
        # Add clusters
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            fig.add_trace(go.Scatter(
                x=embeddings[mask, 0],
                y=embeddings[mask, 1],
                mode='markers',
                name=f'Cluster {cluster_id+1}',
                text=[head_labels[i] for i in np.where(mask)[0]],
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f"{model_name} Attention Head Clustering",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            height=600
        )
        
        return fig, clusters, patterns, head_labels

    def build_cluster_interpretation(clusters, patterns, token_labels, model_name):
        """Show characteristic attention patterns for each cluster"""
        unique_clusters = np.unique(clusters)
        figs = []
        
        for cluster_id in unique_clusters:
            # Get patterns belonging to this cluster
            cluster_patterns = [patterns[i] for i in np.where(clusters == cluster_id)[0]]
            avg_pattern = np.mean(cluster_patterns, axis=0)
            
            # Create heatmap figure
            fig = go.Figure(data=go.Heatmap(
                z=[avg_pattern],  # Single row for average pattern
                x=token_labels,
                y=[f"Cluster {cluster_id+1} Avg"],
                colorscale='Viridis',
                showscale=False
            ))
            
            fig.update_layout(
                title=f"Cluster {cluster_id+1} Characteristic Pattern",
                xaxis_title="Input Tokens",
                yaxis_title="",
                height=200,
                margin=dict(t=40, b=20)
            )
            figs.append(fig)
        
        return figs

    def get_contrastive_analysis(token_probs, tokenizer, input_ids, target_token, contrast_token, position_index):
        """Calculate layer-wise contrast between two tokens"""
        layer_deltas = []
        # token_id_a = tokenizer.encode(target_token)[0]
        # token_id_b = tokenizer.encode(contrast_token)[0]
        
        for layer_probs in token_probs:
            # Get probabilities from full distribution
            prob_a = layer_probs.get(target_token, 0)
            prob_b = layer_probs.get(contrast_token, 0)
            layer_deltas.append(prob_a - prob_b)
        
        return layer_deltas

    def build_contrastive_plot(layer_deltas, model_name):
        fig = go.Figure()
        layers = list(range(1, len(layer_deltas)+1))
        
        # Create color array based on delta values
        colors = ['green' if delta >= 0 else 'red' for delta in layer_deltas]
        
        fig.add_trace(go.Scatter(
            x=layers,
            y=layer_deltas,
            mode='lines+markers',
            marker=dict(color=colors, size=8),
            line=dict(color='gray', width=1)
        ))
        
        fig.update_layout(
            title=f"{model_name}: Layer-wise Preference (A vs B)",
            xaxis_title="Transformer Layer",
            yaxis_title="Probability Delta (A - B)",
            hovermode="x",
            showlegend=False,
            shapes=[dict(
                type='line',
                yref='y', y0=0, y1=0,
                xref='paper', x0=0, x1=1,
                line=dict(color='black', dash='dot')
            )]
        )
        return fig

    


    # Run model 1
    model_1, tokenizer_1 = load_model(model_1_name)
    input_ids_1 = tokenizer_1(input_text, return_tensors="pt").input_ids
    if position_index == -1:
        position_index = -1
    outputs_1 = get_outputs(model_1_name, tokenizer_1, input_ids_1)
    probs_1, tokens_1, pred_1 = get_layerwise_probs(outputs_1, model_1_name, tokenizer_1, input_ids_1, position_index, top_k)
    token_labels_1 = [t.replace("Ġ", "").replace("▁", "") for t in tokenizer_1.convert_ids_to_tokens(input_ids_1[0])]

    st.markdown(f"**Model `{model_1_name}` predicted**: :blue[`{pred_1}`]")
    with st.expander("Token Probability Timeline"):
        st.markdown("Note: Each layer is a Transformer block.")
        st.plotly_chart(build_plot(probs_1, tokens_1, model_1_name), use_container_width=True)
    with st.expander("Aggregated Attention Heatmap"):
        st.markdown("Note: The heatmap shows the single value per token per layer representing aggregated (average) attention across all heads, not the full vector of per-head attention.")
        st.plotly_chart(build_combined_attention_heatmap(outputs_1.attentions, token_labels_1, position_index, model_1_name), use_container_width=True)
    
    with st.expander("Attention Head Clustering Analysis"):
        # st.plotly_chart(
        #     build_attention_clustering(outputs_1.attentions, token_labels_1, position_index, model_1_name),
        #     use_container_width=True
        # )
        cluster_fig_1, clusters_1, patterns_1, head_labels_1 = build_attention_clustering(
            outputs_1.attentions, token_labels_1, position_index, model_1_name
        )
        st.plotly_chart(cluster_fig_1, use_container_width=True)

        st.markdown(f"**{model_1_name} Cluster Interpretation: Average Attention Pattern**")
        cluster_figs_1 = build_cluster_interpretation(clusters_1, patterns_1, token_labels_1, model_1_name)
        for fig in cluster_figs_1:
            st.plotly_chart(fig, use_container_width=True)

    if contrast_mode and contrast_token_a and contrast_token_b:
        try:
            contrast_token_a = contrast_token_a.strip().replace("Ġ", "").replace("▁", "")
            contrast_token_b = contrast_token_b.strip().replace("Ġ", "").replace("▁", "")
            if not tokenizer_1.convert_tokens_to_ids(contrast_token_a):
                raise ValueError(f"Invalid token: {contrast_token_a}")
            if not tokenizer_1.convert_tokens_to_ids(contrast_token_b):
                raise ValueError(f"Invalid token: {contrast_token_b}")
            deltas_1 = get_contrastive_analysis(probs_1, tokenizer_1, input_ids_1, 
                                            contrast_token_a, contrast_token_b, position_index)
            
            with st.expander("Contrastive Analysis"):
                st.plotly_chart(build_contrastive_plot(deltas_1, model_1_name))
                
        except Exception as e:
            st.error(f"Contrastive analysis failed: {str(e)}")

    if mode == "Compare Two Models":
        model_2, tokenizer_2 = load_model(model_2_name)
        input_ids_2 = tokenizer_2(input_text, return_tensors="pt").input_ids
        outputs_2 = get_outputs(model_2_name, tokenizer_2, input_ids_2)
        probs_2, tokens_2, pred_2 = get_layerwise_probs(outputs_2, model_2_name, tokenizer_2, input_ids_2, position_index, top_k)
        token_labels_2 = [t.replace("Ġ", "").replace("▁", "") for t in tokenizer_2.convert_ids_to_tokens(input_ids_2[0])]

        st.markdown(f"**Model `{model_2_name}` predicted**: :blue[`{pred_2}`]")
        with st.expander("Token Probability Timeline"):
            st.plotly_chart(build_plot(probs_2, tokens_2, model_2_name), use_container_width=True)
        with st.expander("Aggregated Attention Heatmap"):
            st.markdown("Note: The heatmap shows the single value per token per layer representing aggregated (average) attention across all heads, not the full vector of per-head attention.")
            st.plotly_chart(build_combined_attention_heatmap(outputs_2.attentions, token_labels_2, position_index, model_2_name), use_container_width=True)
        with st.expander("Attention Head Clustering Analysis"):
            # st.plotly_chart(
            #     build_attention_clustering(outputs_2.attentions, token_labels_2, position_index, model_2_name),
            #     use_container_width=True
            # )
            cluster_fig_2, clusters_2, patterns_2, head_labels_2 = build_attention_clustering(
                outputs_2.attentions, token_labels_2, position_index, model_2_name
            )
            st.plotly_chart(cluster_fig_2, use_container_width=True)
            
            st.markdown(f"**{model_2_name} Cluster Interpretation: Average Attention Pattern**")
            cluster_figs_2 = build_cluster_interpretation(clusters_2, patterns_2, token_labels_2, model_2_name)
            for fig in cluster_figs_2:
                st.plotly_chart(fig, use_container_width=True)
