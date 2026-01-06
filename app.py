# app.py
# Streamlit Web App: EDA + Clustering (K-Means from Scratch) + Classification (Decision Tree)

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Sistem Analisis & Prediksi Nilai Ujian",
    page_icon="📊",
    layout="wide",
)

NUMERIC_COLS = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
    "Exam_Score",
]

CLASS_FEATURES = [
    "Hours_Studied",
    "Attendance",
    "Sleep_Hours",
    "Previous_Scores",
    "Tutoring_Sessions",
    "Physical_Activity",
]


# =========================
# HELPERS
# =========================
@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    return df


@st.cache_data(show_spinner=False)
def prepare_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # convert numeric cols
    out = df.copy()
    missing_cols = [c for c in NUMERIC_COLS if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Kolom numerik tidak ditemukan: {missing_cols}")

    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out[NUMERIC_COLS] = out[NUMERIC_COLS].fillna(out[NUMERIC_COLS].median(numeric_only=True))
    return out


def make_status_lulus(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1 = Lulus, 0 = Tidak Lulus (mapping eksplisit, gak bisa kebalik)
    out["Status_Lulus"] = (out["Exam_Score"] >= 65).astype(int)
    out["Status_Lulus_Text"] = np.where(out["Status_Lulus"] == 1, "Lulus", "Tidak Lulus")
    return out



def fig_bar_counts(labels, counts, title, xlabel, ylabel):
    fig = plt.figure()
    plt.bar(labels, counts)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return fig


def fig_hist(series: pd.Series, bins: int = 20, title: str = ""):
    fig = plt.figure()
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.xlabel(series.name)
    plt.ylabel("Frekuensi")
    return fig


def fig_box(series: pd.Series, title: str = ""):
    fig = plt.figure()
    plt.boxplot(series.dropna(), vert=False)
    plt.title(title)
    plt.xlabel(series.name)
    return fig


def fig_corr_heatmap(corr: pd.DataFrame, title: str = "Heatmap Korelasi Variabel Numerik"):
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(corr.values, aspect="auto")
    plt.title(title)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)

    # annotate values (small matrix only)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            plt.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar()
    plt.tight_layout()
    return fig


def fig_scatter(x: pd.Series, y: pd.Series, title: str):
    fig = plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.title(title)
    return fig


# =========================
# K-MEANS FROM SCRATCH
# =========================
def assign_clusters(X, centroids):
    distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))
    return np.argmin(distances, axis=1)


def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        # guard empty cluster
        if len(cluster_points) == 0:
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)


@st.cache_data(show_spinner=False)
def run_kmeans_from_scratch(df_prepared: pd.DataFrame, k: int = 3, max_iter: int = 100, tolerance: float = 1e-4, seed: int = 42):
    # clustering uses numeric columns except Exam_Score
    X = df_prepared.drop(columns=["Exam_Score"])
    X_num = X.select_dtypes(include=["int64", "float64"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    np.random.seed(seed)
    random_indices = np.random.choice(len(X_scaled), k, replace=False)
    centroids = X_scaled[random_indices]

    converged_at = None
    for it in range(max_iter):
        labels = assign_clusters(X_scaled, centroids)
        new_centroids = update_centroids(X_scaled, labels, k)
        diff = np.linalg.norm(new_centroids - centroids)
        if diff < tolerance:
            converged_at = it
            centroids = new_centroids
            break
        centroids = new_centroids

    # output df with Cluster 1..k
    out = df_prepared.copy()
    out["Cluster"] = labels + 1
    return out, converged_at


# =========================
# DECISION TREE
# =========================
@st.cache_resource(show_spinner=False)
def train_decision_tree(df_prepared: pd.DataFrame):
    df2 = make_status_lulus(df_prepared)

    X = df2[CLASS_FEATURES]
    y = df2["Status_Lulus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=50,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Tidak Lulus", "Lulus"], output_dict=False
        ),
        "label_counts": df2["Status_Lulus"].value_counts().sort_index(),
    }
    return model, df2, metrics


# =========================
# UI
# =========================
st.title("📊 Sistem Analisis & Prediksi Nilai Ujian")

with st.sidebar:
    st.header("Menu")
    page = st.radio(
        "Navigasi",
        ["Beranda", "Analisis Data (EDA)", "Clustering (K-Means)", "Prediksi Kelulusan (Decision Tree)"],
        index=0
    )
    st.markdown("---")
    st.caption("Dataset: StudentPerformanceFactors.csv")


# =========================
# LOAD + PREPARE DATA
# =========================
try:
    df_raw = load_data("StudentPerformanceFactors.csv")
    df = prepare_numeric(df_raw)
    data_ok = True
except Exception as e:
    data_ok = False
    st.error(f"Gagal memuat/menyiapkan data: {e}")

if not data_ok:
    st.stop()


# =========================
# BERANDA
# =========================
if page == "Beranda":
    st.success("✅ Data Berhasil Dimuat")

    total = int(df.shape[0])
    avg_score = float(df["Exam_Score"].mean())
    pass_rate = float((df["Exam_Score"] >= 65).mean() * 100)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Siswa", f"{total:,}".replace(",", "."))
    c2.metric("Rata-rata Nilai Ujian", f"{avg_score:.2f}")
    c3.metric("Persentase Kelulusan (≥65)", f"{pass_rate:.2f}%")

    st.markdown("### Data (Semua Kolom)")
    st.write(f"Jumlah kolom: **{df.shape[1]}**")
    st.dataframe(df, use_container_width=True, height=420)

    # download full dataset (prepared)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data (CSV)",
        data=csv_bytes,
        file_name="student_performance_prepared.csv",
        mime="text/csv",
    )


# =========================
# EDA
# =========================
elif page == "Analisis Data (EDA)":
    st.subheader("🔎 Exploratory Data Analysis (EDA)")

    st.markdown("#### Statistik Deskriptif (Numerik)")
    st.dataframe(df[NUMERIC_COLS].describe(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### Visualisasi Distribusi & Outlier")

    col_left, col_right = st.columns(2)
    with col_left:
        feat = st.selectbox("Pilih fitur (Histogram)", NUMERIC_COLS, index=0)
        st.pyplot(fig_hist(df[feat], bins=20, title=f"Distribusi {feat}"), clear_figure=True)

    with col_right:
        feat2 = st.selectbox("Pilih fitur (Boxplot)", NUMERIC_COLS, index=0, key="box_feat")
        st.pyplot(fig_box(df[feat2], title=f"Boxplot {feat2}"), clear_figure=True)

    st.markdown("---")
    st.markdown("#### Korelasi (Numerik)")
    corr = df[NUMERIC_COLS].corr(numeric_only=True)
    st.pyplot(fig_corr_heatmap(corr), clear_figure=True)

    st.markdown("---")
    st.markdown("#### Scatter vs Exam_Score")
    scatter_feat = st.selectbox(
        "Pilih fitur untuk dibandingkan dengan Exam_Score",
        [c for c in NUMERIC_COLS if c != "Exam_Score"],
        index=0
    )
    st.pyplot(
        fig_scatter(df[scatter_feat], df["Exam_Score"], title=f"{scatter_feat} vs Exam_Score"),
        clear_figure=True
    )

    st.markdown("---")
    st.markdown("#### Data Lengkap (Semua Kolom)")
    st.dataframe(df, use_container_width=True, height=420)


# =========================
# CLUSTERING
# =========================
elif page == "Clustering (K-Means)":
    st.subheader("🧩 Clustering Siswa (K-Means from Scratch)")

    with st.expander("Pengaturan Clustering", expanded=True):
        k = st.slider("Jumlah Cluster (k)", min_value=2, max_value=6, value=3, step=1)
        max_iter = st.slider("Maksimum Iterasi", min_value=10, max_value=300, value=100, step=10)
        tol = st.number_input("Tolerance", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.6f")
        seed = st.number_input("Random Seed", min_value=0, max_value=9999, value=42, step=1)

    clustered_df, converged_at = run_kmeans_from_scratch(df, k=int(k), max_iter=int(max_iter), tolerance=float(tol), seed=int(seed))

    if converged_at is not None:
        st.info(f"K-Means konvergen pada iterasi ke-{converged_at}")
    else:
        st.warning("K-Means tidak konvergen dalam batas iterasi (hasil tetap ditampilkan).")

    st.markdown("#### Distribusi Cluster")
    dist = clustered_df["Cluster"].value_counts().sort_index()
    st.pyplot(
        fig_bar_counts(
            labels=[f"Cluster {i}" for i in dist.index],
            counts=dist.values,
            title="Distribusi Cluster",
            xlabel="Cluster",
            ylabel="Jumlah Siswa"
        ),
        clear_figure=True
    )
    st.dataframe(dist.rename("Jumlah").to_frame(), use_container_width=True)

    st.markdown("#### Profil Rata-rata per Cluster (Numerik)")
    profile = clustered_df.groupby("Cluster")[NUMERIC_COLS].mean(numeric_only=True)
    st.dataframe(profile, use_container_width=True)

    st.markdown("#### Data Lengkap + Cluster (Semua Kolom)")
    st.dataframe(clustered_df, use_container_width=True, height=420)

    csv_bytes = clustered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Hasil Clustering (CSV)",
        data=csv_bytes,
        file_name="student_performance_with_cluster.csv",
        mime="text/csv",
    )


# =========================
# CLASSIFICATION
# =========================
else:
    st.subheader("🎯 Prediksi Kelulusan (Decision Tree)")

    model, df_with_label, metrics = train_decision_tree(df)

    # label distribution
    st.markdown("#### Distribusi Label (Tidak Lulus vs Lulus)")
    label_counts = metrics["label_counts"]  # 0,1
    st.pyplot(
        fig_bar_counts(
            labels=["Tidak Lulus", "Lulus"],
            counts=[int(label_counts.get(0, 0)), int(label_counts.get(1, 0))],
            title="Distribusi Status Kelulusan Siswa",
            xlabel="Status Kelulusan",
            ylabel="Jumlah Siswa"
        ),
        clear_figure=True
    )

    st.markdown("#### Evaluasi Model")
    st.metric("Akurasi", f"{metrics['accuracy']:.4f}")

    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Aktual: Tidak Lulus", "Aktual: Lulus"], columns=["Pred: Tidak Lulus", "Pred: Lulus"])
    st.markdown("**Confusion Matrix**")
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("**Classification Report**")
    st.code(metrics["classification_report"])

    st.markdown("---")
    st.markdown("#### Visualisasi Decision Tree")
    fig = plt.figure(figsize=(22, 12))
    plot_tree(
        model,
        feature_names=CLASS_FEATURES,
        class_names=["Tidak Lulus", "Lulus"],
        filled=True
    )
    plt.title("Decision Tree - Klasifikasi Kelulusan Siswa")
    st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.markdown("#### Coba Prediksi (Input Manual)")
    c1, c2, c3 = st.columns(3)
    with c1:
        hs = st.number_input("Hours_Studied", min_value=0.0, max_value=float(df["Hours_Studied"].max()), value=float(df["Hours_Studied"].median()))
        att = st.number_input("Attendance", min_value=0.0, max_value=float(df["Attendance"].max()), value=float(df["Attendance"].median()))
    with c2:
        sl = st.number_input("Sleep_Hours", min_value=0.0, max_value=float(df["Sleep_Hours"].max()), value=float(df["Sleep_Hours"].median()))
        ps = st.number_input("Previous_Scores", min_value=0.0, max_value=float(df["Previous_Scores"].max()), value=float(df["Previous_Scores"].median()))
    with c3:
        ts = st.number_input("Tutoring_Sessions", min_value=0.0, max_value=float(df["Tutoring_Sessions"].max()), value=float(df["Tutoring_Sessions"].median()))
        pa = st.number_input("Physical_Activity", min_value=0.0, max_value=float(df["Physical_Activity"].max()), value=float(df["Physical_Activity"].median()))

    X_input = None

    if st.button("Prediksi Status Kelulusan"):
        X_input = pd.DataFrame([{
            "Hours_Studied": hs,
            "Attendance": att,
            "Sleep_Hours": sl,
            "Previous_Scores": ps,
            "Tutoring_Sessions": ts,
            "Physical_Activity": pa
        }])[CLASS_FEATURES]

        pred = int(model.predict(X_input)[0])
        st.success("Hasil: **Lulus** ✅" if pred == 1 else "Hasil: **Tidak Lulus** ❌")

    st.markdown("---")
    st.markdown("#### Data Lengkap + Label (Semua Kolom)")
    st.dataframe(df_with_label, use_container_width=True, height=420)

    csv_bytes = df_with_label.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data + Label (CSV)",
        data=csv_bytes,
        file_name="student_performance_with_label.csv",
        mime="text/csv",
    )

    # st.markdown("#### Contoh Input yang Diprediksi Lulus oleh Model")

    # pred_all = model.predict(df_with_label[CLASS_FEATURES])
    # contoh_lulus = df_with_label.loc[pred_all == 1, CLASS_FEATURES].head(10)

    # if len(contoh_lulus) == 0:
    #     st.warning("Model tidak menemukan contoh yang diprediksi Lulus (jarang terjadi).")
    # else:
    #     st.dataframe(contoh_lulus, use_container_width=True)
    #     st.caption("Copy salah satu baris di atas ke input manual untuk demo hasil 'Lulus'.")
