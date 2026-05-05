"""
🏭 Dashboard — Détection d'anomalies de roulements industriels
NASA IMS Bearing Dataset — Version Cloud (Streamlit)
"""

import os

# KAGGLE CREDENTIALS depuis Streamlit Secrets
os.environ["KAGGLE_USERNAME"] = __import__('streamlit').secrets["KAGGLE_USERNAME"]
os.environ["KAGGLE_KEY"]      = __import__('streamlit').secrets["KAGGLE_KEY"]
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #378ADD;
    }
    .alarm-card {
        background: #2d1a1a;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #D85A30;
    }
    .normal-card {
        background: #1a2d1a;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        border-left: 4px solid #1D9E75;
    }
    h1 { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🏭 Predictive Maintenance Dashboard")
st.markdown("**NASA IMS Bearing Dataset** — Détection d'anomalies par Isolation Forest")
st.markdown("---")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ Paramètres")

@st.cache_data
def get_dataset_path():
    import kagglehub
    path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
    return os.path.join(path, "2nd_test", "2nd_test")

train_size = st.sidebar.slider(
    "📊 Fichiers d'entraînement (période saine)",
    min_value=100, max_value=800, value=500, step=50
)

contamination = st.sidebar.slider(
    "🎯 Contamination attendue (%)",
    min_value=1, max_value=20, value=5
) / 100

n_estimators = st.sidebar.slider(
    "🌲 Nombre d'arbres",
    min_value=50, max_value=500, value=200, step=50
)

bearing_choice = st.sidebar.selectbox(
    "🔩 Roulement à analyser",
    ["Roulement 1", "Roulement 2", "Roulement 3", "Roulement 4"]
)
bearing_idx = int(bearing_choice[-1])

st.sidebar.markdown("---")
st.sidebar.markdown("**👨‍💻 Projet par :** Abdo")
st.sidebar.markdown("**🤖 Algorithme :** Isolation Forest")
st.sidebar.markdown("**📅 Dataset :** NASA IMS 2004")

# ─────────────────────────────────────────────
# CHARGER LES DONNÉES
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path, train_sz):
    records = []
    fichiers = sorted(os.listdir(path))

    progress = st.progress(0, text="Chargement des données...")
    for i, fname in enumerate(fichiers):
        fpath = os.path.join(path, fname)
        try:
            data = pd.read_csv(fpath, sep="\t", header=None)
            data.columns = ["b1", "b2", "b3", "b4"]
            row = {"timestamp": pd.to_datetime(fname, format="%Y.%m.%d.%H.%M.%S")}
            for col in ["b1", "b2", "b3", "b4"]:
                row[f"{col}_rms"]      = np.sqrt(np.mean(data[col]**2))
                row[f"{col}_std"]      = data[col].std()
                row[f"{col}_peak"]     = data[col].abs().max()
                row[f"{col}_kurtosis"] = data[col].kurtosis()
            records.append(row)
        except:
            pass
        progress.progress((i+1)/len(fichiers), text=f"Chargement... {i+1}/{len(fichiers)}")

    progress.empty()
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data
def train_model(df, train_sz, contam, n_est):
    features = [c for c in df.columns if c != "timestamp"]
    X = df[features].values
    X_train = X[:train_sz]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled   = scaler.transform(X)

    model = IsolationForest(
        n_estimators=n_est,
        contamination=contam,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled)

    scores    = -model.score_samples(X_all_scaled)
    anomalies = model.predict(X_all_scaled)
    anomalies = np.where(anomalies == -1, 1, 0)

    return scores, anomalies

# ─────────────────────────────────────────────
# CHARGEMENT
# ─────────────────────────────────────────────
with st.spinner("📥 Téléchargement du dataset NASA..."):
    dataset_path = get_dataset_path()

with st.spinner("⚙️ Chargement et entraînement du modèle..."):
    df = load_data(dataset_path, train_size)
    scores, anomalies = train_model(df, train_size, contamination, n_estimators)

df["anomaly_score"] = scores
df["anomalie"]      = anomalies
b = f"b{bearing_idx}"

# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────
n_anomalies  = anomalies.sum()
pct_anomalie = n_anomalies / len(df) * 100
score_max    = scores.max()
derniere     = df["timestamp"].iloc[-1].strftime("%d %b %Y %H:%M")
statut_final = "🚨 ALARME" if anomalies[-1] == 1 else "✅ NORMAL"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📁 Fichiers analysés", f"{len(df):,}")
col2.metric("🚨 Anomalies détectées", f"{n_anomalies}", f"{pct_anomalie:.1f}%")
col3.metric("📈 Score max", f"{score_max:.4f}")
col4.metric("🔴 Statut final", statut_final)

st.markdown("---")

# ─────────────────────────────────────────────
# GRAPHIQUE 1 : RMS + ANOMALIES
# ─────────────────────────────────────────────
st.subheader(f"📊 Signal RMS — {bearing_choice}")

normal_mask  = df["anomalie"] == 0
anomaly_mask = df["anomalie"] == 1

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df["timestamp"][normal_mask], y=df[f"{b}_rms"][normal_mask],
    mode="lines", name="Normal",
    line=dict(color="#378ADD", width=1.5)
))
fig1.add_trace(go.Scatter(
    x=df["timestamp"][anomaly_mask], y=df[f"{b}_rms"][anomaly_mask],
    mode="markers", name="Anomalie",
    marker=dict(color="#D85A30", size=6, symbol="circle")
))
fig1.add_vrect(
    x0=df["timestamp"].iloc[train_size], x1=df["timestamp"].iloc[train_size+5],
    fillcolor="yellow", opacity=0.3,
    annotation_text="Fin entraînement", annotation_position="top left"
)
fig1.update_layout(
    template="plotly_dark", height=350,
    xaxis_title="Date", yaxis_title="RMS",
    legend=dict(orientation="h", y=1.1)
)
st.plotly_chart(fig1, use_container_width=True)

# ─────────────────────────────────────────────
# GRAPHIQUE 2 : SCORE D'ANOMALIE
# ─────────────────────────────────────────────
st.subheader("🎯 Score d'anomalie dans le temps")

seuil = np.percentile(scores[:train_size], 99)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df["timestamp"], y=df["anomaly_score"],
    fill="tozeroy", mode="lines",
    line=dict(color="#378ADD", width=1),
    fillcolor="rgba(55,138,221,0.2)",
    name="Score anomalie"
))
fig2.add_hline(y=seuil, line_dash="dash", line_color="#D85A30",
               annotation_text=f"Seuil alarme ({seuil:.3f})")
fig2.update_layout(
    template="plotly_dark", height=300,
    xaxis_title="Date", yaxis_title="Score"
)
st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────
# GRAPHIQUE 3 : LES 4 ROULEMENTS
# ─────────────────────────────────────────────
st.subheader("🔩 Comparaison des 4 roulements (RMS)")

colors = ["#378ADD", "#1D9E75", "#D85A30", "#9966CC"]
fig3 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                     subplot_titles=[f"Roulement {i}" for i in range(1, 5)])

for i, color in enumerate(colors, 1):
    fig3.add_trace(
        go.Scatter(x=df["timestamp"], y=df[f"b{i}_rms"],
                   name=f"Roulement {i}", line=dict(color=color, width=1.2)),
        row=i, col=1
    )

fig3.update_layout(template="plotly_dark", height=600, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# ─────────────────────────────────────────────
# TABLEAU DES DERNIÈRES ANOMALIES
# ─────────────────────────────────────────────
st.subheader("📋 Dernières anomalies détectées")

dernières = df[df["anomalie"] == 1][["timestamp", "anomaly_score",
             f"{b}_rms", f"{b}_kurtosis", f"{b}_peak"]].tail(10)
dernières.columns = ["Timestamp", "Score", "RMS", "Kurtosis", "Peak"]
dernières["Score"] = dernières["Score"].round(4)
dernières["RMS"]   = dernières["RMS"].round(4)

st.dataframe(dernières.style.background_gradient(
    subset=["Score"], cmap="Reds"), use_container_width=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown("**🏭 Predictive Maintenance Dashboard** | NASA IMS Bearing Dataset | Isolation Forest | ENSA Kénitra — Génie Industriel")
