# -*- coding: utf-8 -*-
"""
Credit Card Default Prediction - Streamlit Application

This app allows users to:
- Upload credit card client data
- Select from multiple ML models
- View predictions and evaluation metrics
- Analyze model performance through interactive visualizations
"""

# ============================================
# Import Required Libraries
# ============================================

import os  # File and path operations
import joblib  # Loading saved ML models
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
import streamlit as st  # Web app framework
import plotly.graph_objects as go  # Interactive charts
import plotly.express as px  # Quick plotly visualizations
import seaborn as sns  # Statistical plots
import matplotlib.pyplot as plt  # Basic plotting

# Import ML evaluation metrics
from sklearn.metrics import (
    accuracy_score,  # Overall correctness
    roc_auc_score,  # Area under ROC curve
    precision_score,  # Positive prediction accuracy
    recall_score,  # True positive detection rate
    f1_score,  # Harmonic mean of precision and recall
    matthews_corrcoef,  # Correlation coefficient
    confusion_matrix,  # Prediction matrix
    classification_report  # Detailed metrics report
)

# ============================================
# Configure Streamlit Page Settings
# ============================================

st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="📄",  # Document icon as favicon
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling with professional color scheme
st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main {
        background-color: #f8f9fa;
        padding-top: 0.5rem;
    }
    
    /* Remove default streamlit padding */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Main title styling */
    .main-title {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        margin-top: 0;
        padding: 0.3rem;
        background: linear-gradient(120deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    
    /* Enhanced dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(5, 150, 105, 0.3);
    }
    
    .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #047857 0%, #065f46 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.4);
    }
    
    /* Section headers */
    .section-header {
        color: #1e40af;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 0.8rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.3rem;
    }
    
    /* Info box enhancement */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        background-color: #eff6ff;
    }
    
    /* Metric container */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.12);
    }
    
    /* Metric label */
    div[data-testid="metric-container"] label {
        color: #475569;
        font-weight: 700;
        font-size: 1.6rem;
    }
    
    /* Metric value */
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #1e40af;
        font-size: 2.3rem;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
    }
    
    /* Selectbox styling */
    .stSelectbox {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Selectbox dropdown options */
    .stSelectbox [data-baseweb="select"] > div {
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    /* Dropdown menu items */
    [role="listbox"] [role="option"] {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.5rem 0.8rem !important;
    }
    
    /* File uploader */
    .stFileUploader {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        background-color: white;
    }
    
    /* Code block */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* Divider */
    hr {
        margin: 0.5rem 0;
        border: none;
        border-top: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# Global Configuration
# ============================================

# Target column name in the dataset
TARGET_COL = "default payment next month"

# Dictionary mapping model names to their saved file paths
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl",
}

# ============================================
# Helper Functions
# ============================================

def safe_predict_proba(model, X):
    """
    Safely extract prediction probabilities from any model.
    
    Args:
        model: Trained ML model
        X: Feature data for prediction
    
    Returns:
        Array of probabilities for positive class (default)
    """
    # Try predict_proba first (most common method)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]  # Get probability of class 1
    
    # Fallback to decision_function for models like SVM
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalize decision scores to [0, 1] range
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    
    # If model supports neither, raise an error
    raise ValueError("Model doesn't support probability prediction.")

def compute_metrics(y_true, y_pred, y_prob):
    """
    Calculate all evaluation metrics for model performance.
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
    
    Returns:
        Dictionary containing all metrics
    """
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),  # % of correct predictions
        "AUC": float(roc_auc_score(y_true, y_prob)),  # Area under ROC curve
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),  # Accuracy of positive predictions
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),  # % of actual positives found
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),  # Balance of precision & recall
        "MCC": float(matthews_corrcoef(y_true, y_pred)),  # Correlation coefficient (-1 to 1)
    }

def load_dataframe(uploaded_file):
    """
    Load and clean the uploaded CSV file.
    
    Args:
        uploaded_file: Streamlit file uploader object
    
    Returns:
        Cleaned pandas DataFrame
    """
    df = pd.read_csv(uploaded_file)  # Read CSV into DataFrame
    df.columns = [str(c).strip() for c in df.columns]  # Remove whitespace from column names
    df = df.dropna().reset_index(drop=True)  # Remove rows with missing values
    return df

# ============================================
# Visualization Functions
# ============================================

def plot_confusion_matrix_plotly(cm, class_names=["No Default", "Default"]):
    """
    Create interactive confusion matrix heatmap with contrasting colors.
    
    Args:
        cm: Confusion matrix array
        class_names: Labels for classes
    
    Returns:
        Plotly figure object
    """
    # Normalize confusion matrix for color mapping
    cm_normalized = cm.astype('float') / cm.sum()
    
    # Create custom color matrix for each quadrant
    # Green for correct predictions (TN, TP), Red/Orange for errors (FP, FN)
    colors = [
        ['#10b981', '#ef4444'],  # Row 0: TN (green), FP (red)
        ['#f97316', '#3b82f6']   # Row 1: FN (orange), TP (blue)
    ]
    
    # Create individual rectangles for each cell with custom colors
    shapes = []
    annotations = []
    
    for i in range(2):
        for j in range(2):
            # Add colored rectangle for this cell
            shapes.append(dict(
                type='rect',
                x0=j-0.5, x1=j+0.5,
                y0=i-0.5, y1=i+0.5,
                fillcolor=colors[i][j],
                line=dict(color='white', width=3),
                layer='below'
            ))
            
            # Add text annotations
            percentage = cm[i][j] / cm.sum() * 100
            annotations.append(dict(
                x=j, y=i,
                text=f'<b>{cm[i][j]:,}</b><br>({percentage:.1f}%)',
                showarrow=False,
                font=dict(size=16, color='white', family='Inter', weight='bold'),
                xanchor='center',
                yanchor='middle'
            ))
    
    # Create figure with custom layout
    fig = go.Figure()
    
    # Add a dummy heatmap for structure (transparent)
    fig.add_trace(go.Heatmap(
        z=[[0, 0], [0, 0]],
        x=class_names,
        y=class_names,
        showscale=False,
        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
        hoverinfo='skip'
    ))
    
    # Add custom hover information
    hover_text = [
        ['True Negative (TN)<br>Correctly predicted No Default', 
         'False Positive (FP)<br>Incorrectly predicted Default'],
        ['False Negative (FN)<br>Incorrectly predicted No Default', 
         'True Positive (TP)<br>Correctly predicted Default']
    ]
    
    for i in range(2):
        for j in range(2):
            fig.add_trace(go.Scatter(
                x=[j], y=[i],
                mode='markers',
                marker=dict(size=1, opacity=0),
                hovertemplate=f'<b>{hover_text[i][j]}</b><br>Count: {cm[i][j]:,}<br>Percentage: {cm[i][j]/cm.sum()*100:.2f}%<extra></extra>',
                showlegend=False
            ))
    
    fig.update_layout(
        xaxis_title=dict(text="<b>Predicted Label</b>", font=dict(size=18, family="Inter", weight="bold")),
        yaxis_title=dict(text="<b>Actual Label</b>", font=dict(size=18, family="Inter", weight="bold")),
        width=450,
        height=380,
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, family="Inter, sans-serif"),
        xaxis=dict(
            side='top', 
            tickfont=dict(size=16, family="Inter", weight="bold"),
            showgrid=False,
            zeroline=False,
            range=[-0.6, 1.6]
        ),
        yaxis=dict(
            autorange='reversed', 
            tickfont=dict(size=16, family="Inter", weight="bold"),
            showgrid=False,
            zeroline=False,
            range=[-0.6, 1.6]
        ),
        margin=dict(l=100, r=40, t=60, b=40),
        shapes=shapes,
        annotations=annotations
    )
    
    # Add legend explaining colors
    legend_annotations = [
        dict(x=1.15, y=0.95, xref='paper', yref='paper',
             text='<b>Legend:</b>', showarrow=False,
             font=dict(size=14, family='Inter', weight='bold'),
             xanchor='left', align='left'),
        dict(x=1.15, y=0.85, xref='paper', yref='paper',
             text='🟢 Correct Predictions', showarrow=False,
             font=dict(size=13, family='Inter', weight='bold'),
             xanchor='left', align='left'),
        dict(x=1.15, y=0.75, xref='paper', yref='paper',
             text='🔴 False Positive', showarrow=False,
             font=dict(size=13, family='Inter', weight='bold'),
             xanchor='left', align='left'),
        dict(x=1.15, y=0.65, xref='paper', yref='paper',
             text='🟠 False Negative', showarrow=False,
             font=dict(size=13, family='Inter', weight='bold'),
             xanchor='left', align='left'),
        dict(x=1.15, y=0.55, xref='paper', yref='paper',
             text='🔵 True Positive', showarrow=False,
             font=dict(size=13, family='Inter', weight='bold'),
             xanchor='left', align='left'),
    ]
    
    fig.update_layout(annotations=fig.layout.annotations + tuple(legend_annotations))
    
    return fig

def plot_classification_metrics(report_dict):
    """
    Create bar chart showing precision, recall, and F1 for each class.
    
    Args:
        report_dict: Classification report as dictionary
    
    Returns:
        Plotly figure object
    """
    # Prepare data for visualization
    metrics_data = []
    classes = ['0 (No Default)', '1 (Default)']
    
    for idx, class_label in enumerate(['0', '1']):
        if class_label in report_dict:
            metrics_data.append({
                'Class': classes[idx],
                'Precision': report_dict[class_label]['precision'],
                'Recall': report_dict[class_label]['recall'],
                'F1-Score': report_dict[class_label]['f1-score']
            })
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart
    fig = go.Figure()
    
    colors = {'Precision': '#3b82f6', 'Recall': '#06b6d4', 'F1-Score': '#8b5cf6'}
    
    for metric in ['Precision', 'Recall', 'F1-Score']:
        fig.add_trace(go.Bar(
            name=metric,
            x=df_metrics['Class'],
            y=df_metrics[metric],
            text=[f'{val:.3f}' for val in df_metrics[metric]],
            textposition='auto',
            textfont=dict(size=16, family="Inter", weight="bold"),
            marker_color=colors[metric],
            hovertemplate='<b>%{x}</b><br>' + metric + ': %{y:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': "Classification Metrics by Class",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e40af', 'family': 'Inter', 'weight': 700}
        },
        xaxis_title=dict(text="Class", font=dict(size=18, family="Inter", weight="bold")),
        yaxis_title=dict(text="Score", font=dict(size=18, family="Inter", weight="bold")),
        barmode='group',
        width=400,
        height=300,
        plot_bgcolor='rgba(240,242,246,0.5)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, family="Inter, sans-serif"),
        xaxis=dict(tickfont=dict(size=16, family="Inter", weight="bold")),
        yaxis=dict(tickfont=dict(size=15, family="Inter", weight="bold")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=14, family="Inter", weight="bold")
        ),
        yaxis_range=[0, 1.1],
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig

def plot_confusion_matrix_matplotlib(cm, class_names=["No Default (0)", "Default (1)"]):
    """
    Create static confusion matrix using matplotlib (fallback option).
    
    Args:
        cm: Confusion matrix array
        class_names: Labels for classes
    
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    
    # Create heatmap with blue color scheme
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                square=True, linewidths=1.5, linecolor='white',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Count', 'shrink': 0.8}, 
                annot_kws={'fontsize': 10})
    
    # Add percentages in each cell
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            percentage = cm[i][j] / cm.sum() * 100
            ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                   ha='center', va='center', color='gray', fontsize=8)
    
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=11, fontweight='bold', pad=8)
    ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    return fig

def plot_evaluation_metrics(metrics):
    """
    Create bar chart displaying all model evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        Plotly figure object
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    # Assign distinct colors to each metric
    colors = ['#3b82f6', '#06b6d4', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        text=[f'<b>{val:.4f}</b>' for val in metric_values],
        textposition='outside',
        textfont=dict(size=18, family="Inter", weight="bold", color='#1e293b'),
        marker=dict(
            color=colors[:len(metric_names)],
            line=dict(color='rgba(255,255,255,0.3)', width=1.5)
        ),
        hovertemplate='<b>%{x}</b><br>Value: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Model Evaluation Metrics Overview",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#1e40af', 'family': 'Inter', 'weight': 700}
        },
        xaxis_title=dict(text="Metric", font=dict(size=18, family="Inter", weight="bold")),
        yaxis_title=dict(text="Score", font=dict(size=18, family="Inter", weight="bold")),
        width=800,
        height=450,
        plot_bgcolor='rgba(248,249,250,0.8)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, family="Inter, sans-serif"),
        xaxis=dict(
            tickfont=dict(size=20, family="Inter", weight="bold"),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            tickfont=dict(size=15, family="Inter", weight="bold"),
            gridcolor='rgba(0,0,0,0.1)',
            range=[0, 1.15]
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        showlegend=False
    )
    
    # Add baseline reference line for comparison
    fig.add_hline(
        y=0.5, 
        line_dash="dash", 
        line_color="rgba(100,116,139,0.5)",
        annotation_text="Baseline (0.5)",
        annotation_position="right",
        annotation_font_size=10,
        annotation_font_color="rgba(100,116,139,0.8)"
    )
    
    return fig

def plot_metrics_radar(metrics):
    """
    Create radar/spider chart for metrics comparison.
    
    Args:
        metrics: Dictionary of metric names and values
    
    Returns:
        Plotly figure object
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=metric_values,
        theta=metric_names,
        fill='toself',
        fillcolor='rgba(59, 130, 246, 0.3)',
        line=dict(color='#3b82f6', width=2.5),
        marker=dict(size=8, color='#2563eb'),
        hovertemplate='<b>%{theta}</b><br>Value: %{r:.4f}<extra></extra>',
        name='Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=11, family="Inter"),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family="Inter", weight="bold")
            ),
            bgcolor='rgba(248,249,250,0.5)'
        ),
        title={
            'text': "Metrics Radar View",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#1e40af', 'family': 'Inter', 'weight': 700}
        },
        width=450,
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=13, family="Inter, sans-serif"),
        margin=dict(l=80, r=80, t=80, b=60),
        showlegend=False
    )
    
    return fig

# ============================================
# Streamlit User Interface
# ============================================

# Display main title and subtitle
st.markdown('<h1 class="main-title">Credit Card Default Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Machine Learning Model Comparison & Evaluation Platform</p>', unsafe_allow_html=True)

# Display student information
st.markdown("""
<div style="text-align: center; margin-top: 0.5rem; margin-bottom: 1rem;">
    <p style="color: #1e40af; font-size: 1.2rem; font-weight: 700; margin: 0;">
        <strong>Student Name:</strong> GANESAN V | <strong>BITS ID:</strong> 2025AA05382 | <strong>Course:</strong> ML Assignment 2
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Create three equal columns for user inputs
col_left, col_middle, col_right = st.columns([2, 2, 2])

# Column 1: Model Selection Dropdown
with col_left:
    st.markdown("<p style='font-size: 1.8rem; font-weight: 700; color: #1e40af; margin-bottom: 0.3rem;'><b>Select Model</b></p>", unsafe_allow_html=True)
    model_name = st.selectbox("Choose Model", list(MODEL_FILES.keys()), label_visibility="collapsed")

# Column 2: File Upload Widget
with col_middle:
    st.markdown("<p style='font-size: 1.8rem; font-weight: 700; color: #1e40af; margin-bottom: 0.3rem;'><b>Upload Data</b></p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

# Column 3: Sample Data Download Button
with col_right:
    st.markdown("<p style='font-size: 1.8rem; font-weight: 700; color: #1e40af; margin-bottom: 0.3rem;'><b>Sample Data</b></p>", unsafe_allow_html=True)
    # Provide a sample CSV for users to test with
    try:
        sample_test_path = "test.csv"
        if os.path.exists(sample_test_path):
            with open(sample_test_path, "rb") as file:
                st.download_button(
                    label="Download Sample",
                    data=file,
                    file_name="test.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.warning(f"File not found")
    except Exception as e:
        st.warning(f"Error: {e}")

# Stop execution if no file is uploaded
if uploaded is None:
    st.stop()

# ============================================
# File Preview Section
# ============================================

st.markdown("---")
st.markdown("<h3 style='color: #1e40af; font-size: 2rem; font-weight: 700; margin-top: 0.5rem; margin-bottom: 0.5rem;'><b>Uploaded File Preview</b></h3>", unsafe_allow_html=True)

try:
    # Load uploaded file for preview without affecting main processing
    preview_df = pd.read_csv(uploaded)
    preview_df.columns = [str(c).strip() for c in preview_df.columns]
    
    # Display basic file statistics
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.metric(label="Rows", value=f"{len(preview_df):,}")
    with info_col2:
        st.metric(label="Columns", value=f"{len(preview_df.columns)}")
    with info_col3:
        st.metric(label="Size", value=f"{uploaded.size / 1024:.1f} KB")
    
    # Expandable section with detailed data views
    with st.expander("View Data", expanded=False):
        # Create tabs for different data views
        view_tab1, view_tab2, view_tab3 = st.tabs(["Data Preview", "Statistics", "Info"])
        
        with view_tab1:
            # Show first and last rows side by side
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<p style='font-size: 1.2rem; font-weight: 700;'>First 10 rows:</p>", unsafe_allow_html=True)
                st.dataframe(preview_df.head(10), use_container_width=True, height=300)
            with col_b:
                st.markdown("<p style='font-size: 1.2rem; font-weight: 700;'>Last 10 rows:</p>", unsafe_allow_html=True)
                st.dataframe(preview_df.tail(10), use_container_width=True, height=300)
        
        with view_tab2:
            st.markdown("<p style='font-size: 1.2rem; font-weight: 700;'>Descriptive Statistics:</p>", unsafe_allow_html=True)
            st.dataframe(preview_df.describe(), use_container_width=True)
            
            # Show target variable distribution if present
            if TARGET_COL in preview_df.columns:
                st.markdown(f"<p style='font-size: 1.2rem; font-weight: 700;'>Target Variable Distribution (<code>{TARGET_COL}</code>):</p>", unsafe_allow_html=True)
                target_counts = preview_df[TARGET_COL].value_counts().sort_index()
                target_df = pd.DataFrame({
                    'Class': target_counts.index,
                    'Count': target_counts.values,
                    'Percentage': (target_counts.values / len(preview_df) * 100).round(2)
                })
                st.dataframe(target_df, use_container_width=True)
        
        with view_tab3:
            st.markdown("<p style='font-size: 1.2rem; font-weight: 700;'>Column Information:</p>", unsafe_allow_html=True)
            # Build detailed column info table
            info_data = []
            for col in preview_df.columns:
                info_data.append({
                    'Column': col,
                    'Type': str(preview_df[col].dtype),
                    'Non-Null': f"{preview_df[col].notna().sum():,}",
                    'Null': f"{preview_df[col].isna().sum():,}",
                    'Unique': f"{preview_df[col].nunique():,}"
                })
            info_df = pd.DataFrame(info_data)
            st.dataframe(info_df, use_container_width=True)
    
    # Reset file pointer for model processing
    uploaded.seek(0)
    
except Exception as e:
    st.warning(f"Could not preview file: {e}")
    uploaded.seek(0)

# ============================================
# Load Pre-trained Model
# ============================================

# Get the file path for selected model
model_path = MODEL_FILES[model_name]

# Check if model file exists
if not os.path.exists(model_path):
    st.error(
        f"Model file not found: `{model_path}`.\n\n"
        "Train and save models first using `python train_models.py`."
    )
    st.stop()

# Load the model from disk
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Could not load model file. Error: {e}")
    st.stop()

# ============================================
# Load Data and Make Predictions
# ============================================

# Load the uploaded CSV file
try:
    df = load_dataframe(uploaded)
except Exception as e:
    st.error(f"Could not load uploaded CSV. Error: {e}")
    st.stop()

# Verify dataframe loaded successfully
if df is not None:
    # Check if target column exists
    if TARGET_COL not in df.columns:
        st.error(f"Uploaded CSV must contain the target column: '{TARGET_COL}'.")
        st.stop()

    # Separate features and target variable
    y_true = df[TARGET_COL]  # Actual labels for evaluation
    X = df.drop(columns=[TARGET_COL])  # Input features for prediction

    # Make predictions with the model
    try:
        y_pred = model.predict(X)  # Predicted class labels
        y_prob = safe_predict_proba(model, X)  # Prediction probabilities
    except Exception as e:
        st.error(
            "Prediction failed. This usually happens if the uploaded CSV columns don't match.\n\n"
            f"Error: {e}"
        )
        st.stop()

    # Calculate all evaluation metrics
    metrics = compute_metrics(y_true, y_pred, y_prob)
    cm = confusion_matrix(y_true, y_pred)  # 2x2 confusion matrix
    report = classification_report(y_true, y_pred, digits=4)  # Text report
    report_dict = classification_report(y_true, y_pred, output_dict=True)  # Dict for charts
else:
    st.info("No file uploaded. Please upload a CSV to proceed.")

# ============================================
# Display Model Results
# ============================================

# Only show results if metrics were successfully computed
if 'metrics' in locals():
    st.markdown("---")
    st.markdown(f'<h2 class="section-header">Results for: {model_name}</h2>', unsafe_allow_html=True)

    # Section 1: Metrics Visualization
    st.markdown("<h3 style='color: #1e40af; font-size: 2rem; font-weight: 700; margin-top: 0.5rem; margin-bottom: 0.5rem;'><b>Evaluation Metrics</b></h3>", unsafe_allow_html=True)
    
    # Create tabs for different metric views
    metrics_tab1, metrics_tab2 = st.tabs(["Bar Chart", "Radar Chart"])
    
    # Tab 1: Bar chart showing all metrics
    with metrics_tab1:
        try:
            fig_metrics_bar = plot_evaluation_metrics(metrics)
            st.plotly_chart(fig_metrics_bar, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create bar chart: {e}")
    
    # Tab 2: Radar chart for circular view
    with metrics_tab2:
        try:
            fig_metrics_radar = plot_metrics_radar(metrics)
            st.plotly_chart(fig_metrics_radar, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not create radar chart: {e}")
    
    st.markdown("---")
    
    # Section 2: Confusion Matrix and Classification Report
    viz_col1, viz_col2 = st.columns([1, 1])
    
    # Left column: Confusion Matrix
    with viz_col1:
        st.markdown("<h4 style='color: #1e40af; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem;'><b>Confusion Matrix</b></h4>", unsafe_allow_html=True)
        
        # Display interactive confusion matrix heatmap
        try:
            fig_cm = plot_confusion_matrix_plotly(cm)
            st.plotly_chart(fig_cm, use_container_width=True)
        except Exception as e:
            # Fallback to simple table if plotting fails
            st.warning(f"Could not create confusion matrix plot: {e}. Showing table instead.")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0 (No Default)", "Actual 1 (Default)"],
                columns=["Predicted 0", "Predicted 1"]
            )
            st.dataframe(cm_df, use_container_width=True)
    
    # Right column: Classification Report
    with viz_col2:
        st.markdown("<h4 style='color: #1e40af; font-size: 1.8rem; font-weight: 700; margin-bottom: 0.3rem;'><b>Classification Report</b></h4>", unsafe_allow_html=True)
        
        # Tabs for visual and text report
        tab3, tab4 = st.tabs(["Visual", "Detailed"])
        
        # Visual representation
        with tab3:
            try:
                fig_metrics = plot_classification_metrics(report_dict)
                st.plotly_chart(fig_metrics, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create metrics visualization: {e}")
                st.code(report, language="text")
        
        # Detailed text report
        with tab4:
            st.code(report, language="text")

# Show message if no results to display
else:
    st.info("📤 Upload a CSV file and select a model to see evaluation results.")

# ============================================
# Footer
# ============================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; padding: 0.5rem; font-size: 0.8rem;">
    <p style="margin: 0;">💡 Streamlit ML App | © 2026</p>
</div>
""", unsafe_allow_html=True)

