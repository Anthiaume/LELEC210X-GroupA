import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from student_fct import load_data, save_confusion_matrix




records = [("mcu13", "vinikot", "JBL Flip 5 - Auguste - spec_20_20"), # chainsaw, crackling fire, fireworks, gunshot
               ("mcu13", "fisher", "local speakers - spec_20_20")       ] # background
data, labels = load_data(records)
data  = data / np.linalg.norm(data, axis=1, keepdims=True)
print(data.shape, labels.shape)

df = pd.DataFrame(data)

# np.random.seed(0)

# n = 300

# x1 = np.random.normal(0, 1, n)
# x2 = 0.8*x1 #+ np.random.normal(0, 0.2, n)
# x3 = -0.5*x1 #+ np.random.normal(0, 0.3, n)
# x4 = np.random.normal(0, 1, n)
# x5 = (x4) * 0.01 + np.random.normal(0, 0.3, n)

# df = pd.DataFrame({
#     "f1": x1,
#     "f2": x2,
#     "f3": x3,
#     "f4": x4,
#     "f5": x5
# })
df.columns = [f"f{i}" for i in range(df.shape[1])]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(df["f1"], df["f2"], alpha=0.5)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Feature 1 vs Feature 2")
plt.subplot(1, 2, 2)
plt.scatter(df["f4"], df["f5"], alpha=0.5)
plt.xlabel("Feature 4")
plt.ylabel("Feature 5")
plt.title("Feature 4 vs Feature 5")
plt.tight_layout()
plt.show()

print(df)
print("pilou")
n = 20
df_tronc = df.copy()
for i in range(n):
    df_tronc.iloc[:, i*n] = 0
print(df_tronc)
print("pilou")
# df_tronc = df.iloc[:, :-n]  # toutes les lignes, toutes les colonnes sauf les n dernières


def analyze_pca_features(df, n_components=None, Normalize=True):
    """
    Analyse l'impact des features sur la variance du dataset via PCA.

    Parameters
    ----------
    df : pandas DataFrame
        Dataset contenant uniquement les features numériques
    n_components : int or None
        Nombre de composantes PCA
    Normalize : bool
        Si True, les données sont normalisées avant l'analyse PCA

    Returns
    -------
    pca, importance_df
    """

    # 1️⃣ Normalisation des données
    scaler = StandardScaler()
    if Normalize:
        X_scaled = df.values
    else:
        X_scaled = df.values

    # 2️⃣ PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # 3️⃣ Variance expliquée
    explained_variance = pca.explained_variance_ratio_

    # print("Variance expliquée par composante :")
    # for i, var in enumerate(explained_variance):
    #     print(f"PC{i+1}: {var:.4f}")

    # print("\nVariance cumulée :", explained_variance.cumsum())

    # 4️⃣ Contribution des features
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])],
        index=df.columns
    )

    # print("\nContribution des features aux composantes :")
    # print(loadings)

    # 5️⃣ Importance globale des features
    importance = np.sum(
        (pca.components_**2) * explained_variance[:, np.newaxis],
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": df.columns,
        "importance": importance
    }).sort_values("importance", ascending=False)

    # print("\nImportance globale des features :")
    # print(importance_df)

    # 6️⃣ Plot variance expliquée
    plt.figure()
    plt.plot(np.cumsum(explained_variance), marker='o')
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance cumulée expliquée")
    plt.title("Variance expliquée par la PCA")
    plt.grid()
    plt.show()

    # 7️⃣ Plot importance features
    plt.figure(figsize=(12, 12))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Importance")
    plt.title("Importance des features (PCA)")
    plt.gca().invert_yaxis()
    plt.show()

    return pca, importance_df

pca, importance = analyze_pca_features(df, Normalize=True)


def Remove_PC(df, pca, variance_threshold=1e-10, Normalize=True):
    """
    Supprime les composantes principales de variance faible et reconstruit les features.
    """

    # standardisation si besoin
    scaler = StandardScaler()
    if Normalize:
        X_scaled = scaler.fit_transform(df)
    else:
        X_scaled = df.values

    # projection dans l'espace PCA
    X_pca = pca.transform(X_scaled)
    # print("Shape PCA:", X_pca.shape)

    # mettre à zéro les PC de variance faible
    print(pca.explained_variance_ratio_)
    mask = pca.explained_variance_ratio_ >= variance_threshold
    pc_kept = np.where(mask)[0] + 1
    print("PC gardées :", pc_kept)

    X_pca[:, ~mask] = 0

    # reconstruction correcte
    X_reconstructed = pca.inverse_transform(X_pca)

    # si normalisation, revenir à l'échelle originale
    if Normalize:
        X_reconstructed = scaler.inverse_transform(X_reconstructed)

    df_reconstructed = pd.DataFrame(X_reconstructed, columns=df.columns)

    return df_reconstructed


df_reconstructed = Remove_PC(df, pca, variance_threshold=0.0002, Normalize=True)

print(df_reconstructed)

# error = np.sqrt(np.mean((df.values - df_reconstructed.values)**2))
# print("Reconstruction RMSE:", error)
rmse = np.sqrt(np.mean((df.values - df_reconstructed.values)**2))
print("RMSE :", rmse)

# Normalisé par la variance
rmse_norm = rmse / df.values.std()
print("RMSE / std :", rmse_norm)

print("\n\n")
print("Troncature\n")
pca, importance = analyze_pca_features(df_tronc, Normalize=True)

rmse = np.sqrt(np.mean((df_tronc.values - df.values)**2))
print("RMSE :", rmse)

# Normalisé par la variance
rmse_norm = rmse / df_tronc.values.std()
print("RMSE / std :", rmse_norm)
# rmse_norm = rmse / df.values.std()
# print("RMSE / std :", rmse_norm)

# print(df_tronc)
# print(df)