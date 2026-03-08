import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Graphes_Voltages_MCU.csv", comment=';', sep=',')

# Extraire les colonnes utiles
# time = df["Time(S)"]
# ch1 = df["CH1(V)"]
# ch2 = df["CH2(V)"]
# df_tronc = df[(df["Time(S)"] >= -2.0) & (df["Time(S)"] <= -1.5)]


df_tronc = df[(df["Time(S)"] >= -0.8) & (df["Time(S)"] <= 1.0)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]


plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1")
plt.plot(df_tronc["Time_shifted"], df_tronc["CH2(V)"], label="CH2")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()

R = 60
df_power = df_tronc.copy()
df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
plt.figure(figsize=(12, 6))
plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power")
plt.ylabel("Power (W)")
plt.grid(True)
plt.legend()
plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()

Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)


##########################################################################


# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Graphes_Voltages_MCU.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -0.8) & (df["Time(S)"] <= 1.0)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]


plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1")
plt.plot(df_tronc["Time_shifted"], df_tronc["CH2(V)"], label="CH2")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()

R = 60
df_power = df_tronc.copy()
df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
plt.figure(figsize=(12, 6))
plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power")
plt.ylabel("Power (W)")
plt.grid(True)
plt.legend()
plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()

Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)