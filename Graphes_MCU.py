import pandas as pd
import matplotlib.pyplot as plt

# # Charger le fichier CSV
# df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Graphes_Voltages_MCU.csv", comment=';', sep=',')

# # Extraire les colonnes utiles
# # time = df["Time(S)"]
# # ch1 = df["CH1(V)"]
# # ch2 = df["CH2(V)"]
# # df_tronc = df[(df["Time(S)"] >= -2.0) & (df["Time(S)"] <= -1.5)]


# df_tronc = df[(df["Time(S)"] >= -0.8) & (df["Time(S)"] <= 1.0)]
# t0 = df_tronc["Time(S)"].iloc[0]
# df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.002152
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.002152


# plt.figure(figsize=(12, 6))
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1")
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH2(V)"], label="CH2")
# plt.grid(True)
# plt.legend()
# plt.show()

# val_max_ch1 = df_tronc["CH1(V)"].max()
# val_min_ch1 = df_tronc["CH1(V)"].min()

# R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

# df_current = df_tronc.copy()
# df_current["Current(A)"] = df_current["CH1(V)"] / R

# df_votage_mcu = df_tronc.copy()
# df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
# df_power_mcu = df_votage_mcu.copy()
# df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
# plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU")
# plt.ylabel("Power MCU (mW)")
# plt.grid(True)
# plt.legend()
# plt.show()

# Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
# print("Mean Power MCU (mW):", Mean_power_mcu)


# ##########################################################################


# # Charger le fichier CSV
# df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Graphes_Voltages_MCU.csv", comment=';', sep=',')

# df_tronc = df[(df["Time(S)"] >= -0.8) & (df["Time(S)"] <= 1.0)]
# t0 = df_tronc["Time(S)"].iloc[0]
# df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.002152
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.002152



# plt.figure(figsize=(12, 6))
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1")
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH2(V)"], label="CH2")
# plt.grid(True)
# plt.legend()
# plt.show()

# val_max_ch1 = df_tronc["CH1(V)"].max()
# val_min_ch1 = df_tronc["CH1(V)"].min()

# R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

# df_current = df_tronc.copy()
# df_current["Current(A)"] = df_current["CH1(V)"] / R

# df_votage_mcu = df_tronc.copy()
# df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
# df_power_mcu = df_votage_mcu.copy()
# df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
# plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU")
# plt.ylabel("Power MCU (mW)")
# plt.grid(True)
# plt.legend()
# plt.show()

# Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
# Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
# print("Mean Power MCU (mW):", Mean_power_mcu)

# print("Duration of signal (s):", Durée_signal)

# df_power_mcu_high_frequency = df_power_mcu.copy()

# ############################ CLOCK REDUCED ##################################


# print("CLOCK")
# # Charger le fichier CSV
# df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_CLOCK_REDUCED.csv", comment=';', sep=',')

# df_tronc = df[(df["Time(S)"] >= -0.926) & (df["Time(S)"] <= 0.958)]
# t0 = df_tronc["Time(S)"].iloc[0]
# df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# # df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# # df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.002152
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.002152

# plt.figure(figsize=(12, 6))
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_clock_reduced")
# plt.title("Voltage CH1 with reduced clock")
# plt.grid(True)
# plt.legend()
# plt.show()

# val_max_ch1 = df_tronc["CH1(V)"].max()
# val_min_ch1 = df_tronc["CH1(V)"].min()
# print("min CH1:", val_min_ch1)
# print("max CH1:", val_max_ch1)

# R = 60
# # df_power = df_tronc.copy()
# # df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# # plt.figure(figsize=(12, 6))
# # plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# # plt.ylabel("Power (W)")
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# df_current = df_tronc.copy()
# df_current["Current(A)"] = df_current["CH1(V)"] / R

# df_votage_mcu = df_tronc.copy()
# df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
# df_power_mcu = df_votage_mcu.copy()
# df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
# plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_lower_clock")
# plt.title("Power MCU with reduced clock")
# plt.ylabel("Power MCU (mW)")
# plt.grid(True)
# plt.legend()
# plt.show()


# Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
# Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
# print("Mean Power MCU (mW):", Mean_power_mcu)
# print("Duration of signal (s):", Durée_signal)

# df_power_mcu_low_frequency = df_power_mcu.copy()




# ############################ RADIO REDUCED ##################################
# print("RADIO")
# # Charger le fichier CSV
# df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_radio_reduced.csv", comment=';', sep=',')

# df_tronc = df[(df["Time(S)"] >= -1.279) & (df["Time(S)"] <= 0.605)]
# t0 = df_tronc["Time(S)"].iloc[0]
# df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# # df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# # df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.002152
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.002152

# plt.figure(figsize=(12, 6))
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_radio_reduced")
# plt.title("Voltage CH1 with reduced radio")
# plt.grid(True)
# plt.legend()
# plt.show()

# val_max_ch1 = df_tronc["CH1(V)"].max()
# val_min_ch1 = df_tronc["CH1(V)"].min()
# print("min CH1:", val_min_ch1)
# print("max CH1:", val_max_ch1)

# R = 60
# # df_power = df_tronc.copy()
# # df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# # plt.figure(figsize=(12, 6))
# # plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# # plt.ylabel("Power (W)")
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# df_current = df_tronc.copy()
# df_current["Current(A)"] = df_current["CH1(V)"] / R

# df_votage_mcu = df_tronc.copy()
# df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
# df_power_mcu = df_votage_mcu.copy()
# df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
# plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_lower_radio_power(-8dB)")
# plt.title("Power MCU with reduced clock")
# plt.ylabel("Power MCU (mW)")
# plt.grid(True)
# plt.legend()
# plt.show()


# Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
# Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
# print("Mean Power MCU (mW):", Mean_power_mcu)
# print("Duration of signal (s):", Durée_signal)

# df_power_mcu_low_radio_power = df_power_mcu.copy()



# ############################ MATRIX REDUCED ##################################
# print("MATRIX")
# # Charger le fichier CSV
# df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_Matrix_REDUCED.csv", comment=';', sep=',')

# df_tronc = df[(df["Time(S)"] >= -1.392) & (df["Time(S)"] <= 0.505)]
# t0 = df_tronc["Time(S)"].iloc[0]
# df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# # df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# # df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

# plt.figure(figsize=(12, 6))
# plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_matrix_reduced")
# plt.title("Voltage CH1 with reduced matrix")
# plt.grid(True)
# plt.legend()
# plt.show()

# val_max_ch1 = df_tronc["CH1(V)"].max()
# val_min_ch1 = df_tronc["CH1(V)"].min()
# print("min CH1:", val_min_ch1)
# print("max CH1:", val_max_ch1)

# R = 60
# # df_power = df_tronc.copy()
# # df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# # plt.figure(figsize=(12, 6))
# # plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# # plt.ylabel("Power (W)")
# # plt.grid(True)
# # plt.legend()
# # plt.show()

# df_current = df_tronc.copy()
# df_current["Current(A)"] = df_current["CH1(V)"] / R

# df_votage_mcu = df_tronc.copy()
# df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
# df_power_mcu = df_votage_mcu.copy()
# df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
# plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_lower_radio_power(-8dB)")
# plt.title("Power MCU with reduced clock")
# plt.ylabel("Power MCU (mW)")
# plt.grid(True)
# plt.legend()
# plt.show()


# Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
# Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
# print("Mean Power MCU (mW):", Mean_power_mcu)
# print("Duration of signal (s):", Durée_signal)

# df_power_mcu_low_matrix_power = df_power_mcu.copy()







############################ UART DISABLED ##################################
print("UART DISABLED")
# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_UART_DISABLE.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -1.379) & (df["Time(S)"] <= -0.34)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_uart_disabled")
plt.title("Voltage CH1 with UART disabled")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()
print("min CH1:", val_min_ch1)
print("max CH1:", val_max_ch1)

R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_uart_disabled")
plt.title("Power MCU with UART disabled")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()


Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)
print("Duration of signal (s):", Durée_signal)

df_power_mcu_low_uart_power = df_power_mcu.copy()




############################ UART DISABLED ##################################
print("UART DISABLED")
# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Powe_MCU_CLOCK_REDUCED_2.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -0.552) & (df["Time(S)"] <= 0.758)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_clock_reduced_2")
plt.title("Voltage CH1 with clock reduced_2")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()
print("min CH1:", val_min_ch1)
print("max CH1:", val_max_ch1)

R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_clock_reduced_2")
plt.title("Power MCU with clock reduced_2")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()


Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)
print("Duration of signal (s):", Durée_signal)

df_power_mcu_low_clock_2 = df_power_mcu.copy()





############################ UART DISABLED ##################################
print("O3")
# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_O3.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -0.650) & (df["Time(S)"] <= 0.291)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_O3")
plt.title("Voltage CH1 with O3")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()
print("min CH1:", val_min_ch1)
print("max CH1:", val_max_ch1)

R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_O3")
plt.title("Power MCU with clock reduced_2")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()


Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)
print("Duration of signal (s):", Durée_signal)

df_power_mcu_O3 = df_power_mcu.copy()






############################ UART DISABLED ##################################
print("MAC")
# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_MAC.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -0.533) & (df["Time(S)"] <= 0.798)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_MAC")
plt.title("Voltage CH1 with O3")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()
print("min CH1:", val_min_ch1)
print("max CH1:", val_max_ch1)

R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power MCU_MAC")
plt.title("Power MCU with MAC reduced")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()


Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)
print("Duration of signal (s):", Durée_signal)

df_power_mcu_MAC = df_power_mcu.copy()








############################ UART DISABLED ##################################
print("MAC")
# Charger le fichier CSV
df = pd.read_csv("C:\\Users\\ismae\\Desktop\\UCLouvain\\Master\\Codes Pythons\\Power_MCU_CLOCK_2MHz.csv", comment=';', sep=',')

df_tronc = df[(df["Time(S)"] >= -1.384) & (df["Time(S)"] <= 0.134)]
t0 = df_tronc["Time(S)"].iloc[0]
df_tronc["Time_shifted"] = df_tronc["Time(S)"] - t0
# df_tronc["CH1(V)"] = df_tronc["CH1(V)"] - df_tronc["CH1(V)"].iloc[0]
# df_tronc["CH2(V)"] = df_tronc["CH2(V)"] - df_tronc["CH2(V)"].iloc[0]
df_tronc["CH1(V)"] = df_tronc["CH1(V)"] + 0.00161
df_tronc["CH2(V)"] = df_tronc["CH2(V)"] + 0.00161

plt.figure(figsize=(12, 6))
plt.plot(df_tronc["Time_shifted"], df_tronc["CH1(V)"], label="CH1_CLOCK_2MHz")
plt.title("Voltage CH1 with CLOCK_2MHz")
plt.grid(True)
plt.legend()
plt.show()

val_max_ch1 = df_tronc["CH1(V)"].max()
val_min_ch1 = df_tronc["CH1(V)"].min()
print("min CH1:", val_min_ch1)
print("max CH1:", val_max_ch1)

R = 60
# df_power = df_tronc.copy()
# df_power["Power_resistance(W)"] = df_power["CH1(V)"] * df_power["CH1(V)"]/ R
# plt.figure(figsize=(12, 6))
# plt.plot(df_power["Time_shifted"], df_power["Power_resistance(W)"], label="Power_resistance_clock_reduced")
# plt.ylabel("Power (W)")
# plt.grid(True)
# plt.legend()
# plt.show()

df_current = df_tronc.copy()
df_current["Current(A)"] = df_current["CH1(V)"] / R

df_votage_mcu = df_tronc.copy()
df_votage_mcu["Voltage_MCU(V)"] = 3.3 - df_tronc["CH1(V)"]
df_power_mcu = df_votage_mcu.copy()
df_power_mcu["Power_MCU(mW)"] = df_votage_mcu["Voltage_MCU(V)"] * df_current["Current(A)"] * 1000
plt.figure(figsize=(12, 6))
plt.plot(df_power_mcu["Time_shifted"], df_power_mcu["Power_MCU(mW)"], label="Power CLOCK_2MHz")
plt.title("Power MCU with MAC reduced")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()


Durée_signal = df_tronc["Time_shifted"].iloc[-1] - df_tronc["Time_shifted"].iloc[0]
Mean_power_mcu = df_power_mcu["Power_MCU(mW)"].mean()
print("Mean Power MCU (mW):", Mean_power_mcu)
print("Duration of signal (s):", Durée_signal)

df_power_mcu_CLOCK_2MHz = df_power_mcu.copy()



plt.figure(figsize=(12, 6))
# plt.plot(df_power_mcu_high_frequency["Time_shifted"], df_power_mcu_high_frequency["Power_MCU(mW)"], label="Power MCU_high_clock")
# plt.plot(df_power_mcu_low_frequency["Time_shifted"], df_power_mcu_low_frequency["Power_MCU(mW)"], label="Power MCU_low_clock")
# plt.plot(df_power_mcu_low_radio_power["Time_shifted"], df_power_mcu_low_radio_power["Power_MCU(mW)"], label="Power MCU_low_radio_power(-8dB)")
# plt.plot(df_power_mcu_low_matrix_power["Time_shifted"], df_power_mcu_low_matrix_power["Power_MCU(mW)"], label="Power MCU_low_matrix_power")
plt.plot(df_power_mcu_low_uart_power["Time_shifted"], df_power_mcu_low_uart_power["Power_MCU(mW)"], label="Power MCU_low_uart_power")
plt.plot(df_power_mcu_low_clock_2["Time_shifted"], df_power_mcu_low_clock_2["Power_MCU(mW)"], label="Power MCU_low_clock_2")
plt.plot(df_power_mcu_O3["Time_shifted"], df_power_mcu_O3["Power_MCU(mW)"], label="Power MCU_O3")
plt.plot(df_power_mcu_MAC["Time_shifted"], df_power_mcu_MAC["Power_MCU(mW)"], label="Power MCU_MAC")
plt.plot(df_power_mcu_CLOCK_2MHz["Time_shifted"], df_power_mcu_CLOCK_2MHz["Power_MCU(mW)"], label="Power MCU_CLOCk_2MHz")
plt.title("Power MCU with high and low clock")
plt.ylabel("Power MCU (mW)")
plt.grid(True)
plt.legend()
plt.show()




