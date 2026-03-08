import numpy as np
import matplotlib.pyplot as plt


# # Première Température
# T1 = np.array([30, 32, 33, 34])  # Température en K
# T2 = np.array([30, 28, 27, 26])  # Température en K
# V1 = np.array([0.000162576234, -0.00120705432, -0.00192177529, -0.00262539534])  # Tension en V

# # Deuxième Température
# T3 = np.array([35, 37, 39, 41])  # Température en K
# T4 = np.array([35, 33, 21, 29])  # Température en K
# V2 = np.array([0.000199233726, -0.00119120378, -0.00247996519, -0.00376091783])  # Tension en V


# # Troisième Température
# T5 = np.array([37, 38, 39, 40])  # Température en K
# T6 = np.array([37, 36, 35, 34])  # Température en K
# V3 = np.array([0.000115832183, -0.000441555706, -0.00105878851, -0.0017925786])  # Tension en V


# # Calcul de la différence de température
# delta_T1 = T1 - T2
# delta_T2 = T3 - T4
# delta_T3 = T5 - T6

# Effet_Seebeck1 = V1 / delta_T1
# Effet_Seebeck2 = V2 / delta_T2
# Effet_Seebeck3 = V3 / delta_T3



import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PARAMÈTRES À ADAPTER
# ---------------------------------------------------------
fichier = r"C:\Users\ismae\Desktop\UCLouvain\Master\Codes Pythons\MAPR2471_Thermopower.txt"
   # ton fichier
col_setpoint = "Setpoint"              # nom de la colonne setpoint
col_V = "V"                            # nom de la colonne tension
col_Th = "SetpointTemperature1(�C)"                       # colonne température chaude
col_Tc = "SetpointTemperature2(�C)"                      # colonne température froide
n_points = 5                           # nombre de points à moyenner
# ---------------------------------------------------------

# Lecture du fichier (auto-détection séparateur)
df = pd.read_csv(fichier, sep=None, engine="python", encoding="cp1252")


# Calcul du Delta T
df["DeltaT"] = df[col_Th] - df[col_Tc]

# Regroupement par setpoint
grouped = df.groupby(col_setpoint)

# Pour chaque setpoint : moyenne des 5 dernières valeurs
results = []
for sp, g in grouped:
    g_sorted = g.sort_index()  # au cas où
    last_vals = g_sorted.tail(n_points)
    V_mean = last_vals[col_V].mean()
    DeltaT_mean = last_vals["DeltaT"].mean()
    results.append([sp, DeltaT_mean, V_mean])

# Conversion en DataFrame
res = pd.DataFrame(results, columns=["Setpoint", "DeltaT", "V_mean"])

# Tri par DeltaT pour un plot propre
res = res.sort_values("DeltaT")

# Plot
plt.figure(figsize=(7,5))
plt.plot(res["DeltaT"], res["V_mean"], "o-", label="V moyen (5 derniers points)")
plt.xlabel("ΔT (K)")
plt.ylabel("V (V)")
plt.title("V vs ΔT")
plt.grid(True)
plt.legend()
plt.show()