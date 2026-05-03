# 🏭 Predictive Maintenance — NASA Bearing Anomaly Detection

> Détection automatique de pannes de roulements industriels par Machine Learning  
> **NASA IMS Bearing Dataset** | Isolation Forest | Streamlit Dashboard

---

## 🎯 Objectif

Détecter en temps réel la **dégradation progressive** de roulements industriels avant qu'une panne survienne, en utilisant des données de vibrations réelles issues des laboratoires NASA.

---

## 🗂️ Structure du projet

```
📁 predictive-maintenance/
│
├── 📓 notebook.ipynb          # Analyse complète paso a paso
├── 🖥️  dashboard.py            # Dashboard Streamlit interactif
├── 📄 README.md
├── 📄 requirements.txt
│
└── 📁 src/
    ├── 1_generate_data.py
    ├── 2_eda.py
    ├── 3_preprocessing.py
    ├── 4_model.py
    └── 5_predict.py
```

---

## 🚀 Installation et lancement

### 1. Cloner le repo
```bash
git clone https://github.com/TON_USERNAME/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Télécharger le dataset
```python
import kagglehub
path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
```

### 4. Lancer le dashboard
```bash
streamlit run dashboard.py
```

---

## 🤖 Algorithme : Isolation Forest

| Paramètre | Valeur |
|---|---|
| Algorithme | Isolation Forest |
| Entraînement | 500 premiers fichiers (période saine) |
| Contamination | 5% |
| Nombre d'arbres | 200 |
| Features | RMS, STD, Peak, Kurtosis × 4 capteurs |

---

## 📈 Features extraites

| Feature | Description | Utilité |
|---|---|---|
| **RMS** | Racine carrée de la moyenne des carrés | Énergie globale de vibration |
| **STD** | Écart-type | Variabilité du signal |
| **Peak** | Valeur absolue maximale | Chocs ponctuels |
| **Kurtosis** | Aplatissement de la distribution | Détection de chocs (pannes) |

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Plotly](https://img.shields.io/badge/Plotly-5.17-purple)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)

---

## 📚 Dataset

**NASA IMS Bearing Dataset** — University of Cincinnati  
- 984 fichiers de mesures sur 7 jours (Feb 2004)
- 4 capteurs de vibrations (roulements)
- Fréquence : toutes les 10 minutes

[📥 Télécharger sur Kaggle](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)

---

## 👨‍💻 Auteur

**Abdo** — Étudiant en Génie Industriel @ ENSA Kénitra  

---

## 📄 Licence

MIT License
