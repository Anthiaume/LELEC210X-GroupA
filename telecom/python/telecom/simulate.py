# ruff: noqa: N806
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import firwin, freqz
from scipy.special import erfc

from .chain import Chain

TYPE = "ML"  # Type of STO estimation to use in the simulation (ML or corr)
PPD_algo = "DUALCORR"  # Algorithm to use for preamble detection (corr or dynamic_ppd)


def add_delay(chain: Chain, x: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """
    Apply the channel between TX and RX, handling the different oversampling factors
    and the addition of a delay.
    """
    fs = chain.bit_rate * chain.osr_rx  # Receiver sampling frequency
    sto_int = np.floor(tau * fs).astype(int)  # Integer delay (on the received samples)
    sto_frac = tau * fs - sto_int  # Fractional delay, remaining

    R = int(chain.osr_tx / chain.osr_rx)
    idx = np.floor(
        sto_frac * R
    ).astype(
        int
    )  # Index of best sample (available in transmitted signal oversampled at TX) depicting the fractional delay

    y = np.concatenate(
        (np.zeros(sto_int), x[idx::R])
    )  # Received signal with 0's from integer delay and sampling starting at the fractional delay

    return y, np.mod(sto_int, chain.osr_rx)


def add_cfo(chain: Chain, x: np.ndarray, cfo: float) -> np.ndarray:
    """Add a frequency offset on the signal x."""
    fs = chain.bit_rate * chain.osr_rx

    t = np.arange(len(x)) / fs  # Time vector
    y = np.exp(1j * 2 * np.pi * cfo * t) * x
    return y


@click.command()
@click.option(
    "-c",
    "--chain-name",
    default="telecom.chain.BasicChain",
    show_default=True,
    help="Chain to simulate, in the form 'module.ClassName'.",
)
@click.option(
    "-s",
    "--seed",
    default=1234,
    show_default=True,
    help="Random seed. Same seed => same results.",
)
@click.option(
    "--dest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path(__file__).parents[2] / "sim_outputs.csv",
    show_default=True,
    help="Write output to this file.",
)
def main(chain_name: str, seed: int, dest: Path):  # noqa: C901
    """
    Simulate the communication chain provided, for several SNRs.
    Compute and display the different metrics to evaluate the performances.
    """
    mod_path, class_name = chain_name.rsplit(".", 1)
    chain_mod = __import__(mod_path, fromlist=[class_name])
    chain: Chain = getattr(chain_mod, class_name)()

    EsN0s_dB = chain.EsN0_range  # Es is the energy of a symbol
    R = chain.osr_rx
    B = chain.bit_rate
    fs = B * R

    # Error counters/metric initialisation
    bit_errors = np.zeros(len(EsN0s_dB))
    packet_errors = np.zeros(len(EsN0s_dB))
    cfo_err = np.zeros(len(EsN0s_dB))
    sto_err = np.zeros(len(EsN0s_dB))
    preamble_misdetect = np.zeros(len(EsN0s_dB))  # Preamble miss detection (not found)
    preamble_false_detect = np.zeros(
        len(EsN0s_dB)
    )  # Preamble false detection (found in noise)

    # Transmitted signals that are independent of the payload bits

    #########################################################################
    # chain.preamble = chain.Viterbi_code_m2(chain.preamble)
    # chain.sync_word = chain.Viterbi_code_m2(chain.sync_word)
    
    #########################################################################

    x_pr = chain.modulate(chain.preamble)  # Modulated signal containing preamble
    x_sync = chain.modulate(chain.sync_word)  # Modulated signal containing sync_word
    hdr_len_byte = (len(chain.preamble) + len(chain.sync_word)) // 8
    x_noise = np.zeros(
        chain.payload_len * chain.osr_tx
    )  # Padding some zeros before the packets

    # Lowpass filter taps
    if chain.taps is None:
        if chain.numtaps != 0:
            taps = firwin(chain.numtaps, chain.cutoff, fs=fs)
    else:
        taps = chain.taps

    rng = np.random.default_rng(seed)

    # For loop on the number of packets to send
    # print(chain.n_packets)
    i=0
    for _ in range(chain.n_packets):
        print(i)
        i+=1
        # Random generation of payload bits
        bits = rng.integers(2, size=chain.payload_len)

        #########################################################
        # bits = chain.Viterbi_code_m2(bits)
        # R1, R0, symb_R1, symb_R0 = chain.poly2trellis(gn, gd)
        # bits = 
        #########################################################
        # Transmitted signal
        x_pay = chain.modulate(bits)  # Modulated signal with payload
        x = np.concatenate((x_noise, x_pr, x_sync, x_pay, np.zeros(chain.osr_tx)))

        # Channel application (without noise addition): delay and frequency offset
        if np.isnan(chain.sto_val):  # STO should be random
            tau = rng.random() * chain.sto_range
        else:
            tau = chain.sto_val

        y, sto_idx = add_delay(chain, x, tau)  # Delay addition
        start_idx = (
            int(tau * chain.osr_rx * B) + chain.payload_len * chain.osr_rx
        )  # Delay + noise in beginning, for STO metric

        if np.isnan(chain.cfo_val):  # CFO should be random
            cfo = rng.uniform(low=chain.cfo_range[0], high=chain.cfo_range[1])
        else:
            cfo = chain.cfo_val
        y_cfo = add_cfo(chain, y, cfo)  # Frequency offset addition

        # Normalized noise generation
        w = (rng.normal(size=y.size) + 1j * rng.normal(size=y.size)) / np.sqrt(
            2
        )  # Normalized complex normal vector CN(0, 1)

        # For loop on the SNRs

        for k, EsN0_dB in enumerate(EsN0s_dB):
            # print(k)
            # Add noise
            EsN0 = 10 ** (EsN0_dB / 10.0)
            SNR_input = EsN0 / chain.osr_rx
            noise = w * np.sqrt(1 / SNR_input)
            y_noisy = y_cfo + noise

            # Low-pass filtering
            if chain.numtaps != 0:
                y_filt = np.convolve(y_noisy, taps, mode="same")
            else:
                y_filt = y_noisy

            ## Preamble detection stage
            if chain.ideal_preamble_detect:
                detect_idx = start_idx
            else:
                if chain.use_dynamic_ppd:
                    detect_idx = chain.preamble_detect_ppd(y_filt, x_pr, PPD_algo)
                    # print(f"PPD detection index: {detect_idx}")
                  
                else:
                    detect_idx = chain.preamble_detect(y_filt)

            preamble_error = False
            if detect_idx is None:  # Misdetection of preamble
                preamble_misdetect[k] += 1
                errors = 0.5 * np.ones(len(bits))
                cfo_hat, tau_hat, detect_idx, start_frame = (
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                )
            else:  # Found a preamble, can demodulate packet
                # Preamble metrics
                if (
                    detect_idx < start_idx - 4 * R
                ):  # False detection of preamble (found in noise)
                    preamble_false_detect[k] += 1
                    preamble_error = True
                elif (
                    detect_idx > start_idx + len(chain.preamble) * chain.osr_rx
                ):  # Misdetection of preamble (found in packet)
                    preamble_misdetect[k] += 1
                    preamble_error = True
                y_detect = y_filt[detect_idx:]
                ## Synchronization stage
                # CFO estimation and correction
                if chain.ideal_cfo_estimation:
                    cfo_hat = cfo
                else:
                    cfo_hat = chain.cfo_estimation(
                        y_detect[: hdr_len_byte * 8 * chain.osr_rx]
                    )

                t = np.arange(len(y_detect)) / (B * R)
                y_sync = np.exp(-1j * 2 * np.pi * cfo_hat * t) * y_detect

                # STO estimation and correction
                if chain.ideal_sto_estimation:
                    if (
                        chain.ideal_preamble_detect
                    ):  # In this case, starting index of preamble already contains sto
                        tau_hat = 0
                    else:
                        tau_hat = chain.sto_estimation(
                        y_sync[: hdr_len_byte * 8 * chain.osr_rx], TYPE, x_pr
                    )
                else:
                    tau_hat = chain.sto_estimation(
                        y_sync[: hdr_len_byte * 8 * chain.osr_rx], TYPE, x_pr
                    )

                y_sync = y_sync[tau_hat:]    
                # print("viterbi") 
                # y_sync = chain.viterbi_decoder(y_sync, 0o7, 0o5, 2)
                # print("viterbi") 
                ## Demodulation and deframing stage
                bits_hat = chain.demodulate(y_sync)
                if (
                    chain.ideal_sto_estimation and chain.ideal_preamble_detect
                ):  # In this case, also assume perfect frame syncrhonization
                    start_frame = len(chain.preamble) + len(chain.sync_word)
                elif len(bits_hat) == 0:
                    preamble_error = True
                else:  # Frame synchronization
                    v = np.abs(
                        np.correlate(
                            bits_hat * 2 - 1,
                            np.array(chain.sync_word) * 2 - 1,
                            mode="full",
                        )
                    )
                    start_frame = np.argmax(v) + 1
                bits_hat_pay = bits_hat[
                    start_frame : start_frame + chain.payload_len
                ]  # Demodulated payload bits

                ## Computing performance metrics
                if len(bits) == len(bits_hat_pay) and not preamble_error:
                    errors = bits ^ bits_hat_pay
                    if np.sum(errors) > 0:
                        print(f"Errors: {np.sum(errors)} / {len(bits)}")
                        print(f"EsN0_dB: {EsN0_dB}", preamble_error)
                else:  # if the number of demodulated symbols is incorrect
                    errors = 0.5 * np.ones(len(bits))  # flag all bits as wrong
                    if(detect_idx < start_idx - 4 * R) : # False detection of preamble (found in noise)
                        errors = np.zeros(len(bits))  # flag all bits as correct, since the preamble was not found and thus no demodulation was done
                    print(f"EsN0_dB: {EsN0_dB}", preamble_error)
            # end if preamble
            if(not np.isnan(detect_idx)):
                bit_errors[k] += np.sum(errors)
                packet_errors[k] += 1 if any(errors) else 0
                cfo_err[k] += (cfo - cfo_hat) ** 2
                print(f"start_idx - detect_idx: {start_idx - detect_idx}")
                sto_err[k] += (
                    (
                        start_idx
                        + len(chain.preamble) * chain.osr_rx
                        + len(chain.sync_word) * chain.osr_rx
                    )
                    / (R * B)
                    - (detect_idx + np.int64(tau_hat) + start_frame * chain.osr_rx) / (R * B)
                ) ** 2
            else:
                bit_errors[k] += chain.payload_len/2
                packet_errors[k] += 1
                cfo_err[k] += np.nan
                sto_err[k] += np.nan

    # === Lecture de toutes les simulations ===
    sim_data_list = []
    if dest.exists():
        with open(dest, "r") as f:
            lines = f.readlines()

        current_data = []
        current_header = ""
        for line in lines:
            if line.startswith("#"):
                if current_data:  # Save previous simulation
                    sim_data_list.append((current_header, np.array(current_data)))
                    current_data = []
                current_header = line.strip()
            else:
                current_data.append([float(x) for x in line.split()])
        if current_data:
            sim_data_list.append((current_header, np.array(current_data)))

    # === Plot BER / PER ===
    EsN0_th = np.arange(EsN0s_dB[0], EsN0s_dB[-1])
    BER_th = 0.5 * erfc(np.sqrt(10 ** (EsN0_th / 10) / 2))
    BER_th_BPSK = 0.5 * erfc(np.sqrt(10 ** (EsN0_th / 10)))
    BER_th_noncoh = 0.5 * np.exp(-(10 ** (EsN0_th / 10)) / 2)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))
    fig_bis, ax_bis = plt.subplots(1, 2, constrained_layout=True, figsize=(10, 4))

    payload_len_sim = 0
    for i, (header, data) in enumerate(sim_data_list):
        EsN0s = data[:, 0]
        BERs = data[:, 1]
        PERs = data[:, 2]
        RMSE_cfos = data[:, 3]
        RMSE_stos = data[:, 4]

        # Extract payload_len and n_packets from header
        import re
        m_payload = re.search(r"payload_len=(\d+)", header)
        m_packets = re.search(r"n_packets=(\d+)", header)
        label = ""
        payload_len_sim = 1
        if i ==0:
            label= "100 payload bits, 100 pkts\n ref : ideal synchronisation"
        elif i==1:
            label= "100 payload bits, 100 pkts"
        elif i==2:
            label= "200 payload bits, 800 pkts"
        elif i==3:
            label= "800 payload bits, 200 pkts"
       
        if m_payload and m_packets:
           payload_len_sim = int(m_payload.group(1))
           payload_len_sim2 = payload_len_sim
           label = f"{m_payload.group(1)} bits, {m_packets.group(1)} pkts"

        color = colors[i % len(colors)]
        ax[0].plot(EsN0s, BERs, "-o", color=color, label=label)
        ax[1].plot(EsN0s, PERs, "-o", color=color, label=label)
        ax_bis[0].plot(EsN0s, RMSE_cfos, "-o", color=color, label=label)
        ax_bis[1].plot(EsN0s, RMSE_stos, "-o", color=color, label=label)

    # Theoretical BER
    ax[0].plot(EsN0_th, BER_th, "k--", label="AWGN Th. FSK")
    ax[0].plot(EsN0_th, BER_th_noncoh, "k-.", label="AWGN Th. FSK non-coh.")
    ax[0].plot(EsN0_th, BER_th_BPSK, "k:", label="AWGN Th. BPSK")
    ax[0].set_ylabel("BER")
    ax[0].set_xlabel("$E_s/N_0$ [dB]")
    ax[0].set_yscale("log")
    ax[0].set_ylim((1e-6, 1))
    ax[0].grid(True)
    ax[0].legend()
    ax[0].set_title("Average Bit Error Rate")

    ax_bis[0].set_ylabel("RMSE CFO [-]")
    ax_bis[0].set_xlabel("$E_s/N_0$ [dB]")
    ax_bis[0].grid(True)
    ax_bis[0].legend()
    ax_bis[0].set_title("RMSE CFO")


    # Theoretical PER
    ax[1].plot(EsN0_th, 1 - (1 - BER_th) ** payload_len_sim, "k--", label="AWGN Th. FSK")
    ax[1].plot(EsN0_th, 1 - (1 - BER_th_noncoh) ** payload_len_sim, "k-.", label="AWGN Th. FSK non-coh.")
    ax[1].plot(EsN0_th, 1 - (1 - BER_th_BPSK) ** payload_len_sim, "k:", label="AWGN Th. BPSK")
    ax[1].set_ylabel("PER")
    ax[1].set_xlabel("$E_s/N_0$ [dB]")
    ax[1].set_yscale("log")
    ax[1].set_ylim((1e-6, 1))
    ax[1].grid(True)
    ax[1].legend()
    ax[1].set_title("Average Packet Error Rate")

    ax_bis[1].set_ylabel("RMSE STO [-]")
    ax_bis[1].set_xlabel("$E_s/N_0$ [dB]")
    ax_bis[1].grid(True)
    ax_bis[1].legend()
    ax_bis[1].set_title("RMSE STO")

    plt.show()

    # Metrics
    BER = bit_errors / chain.payload_len / chain.n_packets
    PER = packet_errors / chain.n_packets
    RMSE_cfo = np.sqrt(cfo_err / chain.n_packets) / B
    RMSE_sto = np.sqrt(sto_err / chain.n_packets) * B
    preamble_mis = preamble_misdetect / chain.n_packets
    preamble_false = preamble_false_detect / chain.n_packets

    # FIR response plot
    if chain.numtaps != 0:
        w, h = freqz(taps)
        f = w * fs * 0.5 / np.pi
        _fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(7, 4))
        ax.set_title("FIR response")
        ax.plot(f, 20 * np.log10(abs(h)), "b")
        ax.set_ylabel("Amplitude (dB)", color="b")
        ax.set_xlabel("Frequency (Hz)")
        ax2 = ax.twinx()
        angles = np.unwrap(np.angle(h))
        ax2.plot(f, angles, "g")
        ax2.set_ylabel("Angle (radians)", color="g")
        ax2.grid(True)
        ax2.axis("tight")

    # Preamble / RMSE plots
    _fig, ax = plt.subplots(1, 3, constrained_layout=True, figsize=(10, 4))
    ax[0].plot(EsN0s_dB, preamble_mis * 100, "-s", label="Miss-detection")
    ax[0].plot(EsN0s_dB, preamble_false * 100, "-s", label="False-detection")
    ax[0].set_title("Preamble detection error")
    ax[0].set_ylabel("[%]")
    ax[0].set_xlabel("$E_s/N_0$ [dB]")
    ax[0].set_ylim([-1, 101])
    ax[0].grid()
    ax[0].legend()

    ax[1].semilogy(EsN0s_dB, RMSE_cfo, "-s")
    ax[1].set_title("RMSE CFO")
    ax[1].set_ylabel("RMSE [-]")
    ax[1].set_xlabel("$E_s/N_0$ [dB]")
    ax[1].grid()

    ax[2].semilogy(EsN0s_dB, RMSE_sto, "-s")
    ax[2].set_title("RMSE STO")
    ax[2].set_ylabel("RMSE [-]")
    ax[2].set_xlabel("$E_s/N_0$ [dB]")
    ax[2].grid()

    plt.show()

    # Save simulation outputs
    save_var = np.column_stack((
        EsN0s_dB,
        BER,
        PER,
        RMSE_cfo,
        RMSE_sto,
        preamble_mis,
        preamble_false
    ))
    with open(dest, "a") as f:
        f.write(f"# Simulation chain={chain_name}, payload_len={chain.payload_len}, n_packets={chain.n_packets}, seed={seed}\n")
        f.write("# EsN0_dB\tBER\tPER\tRMSE_cfo\tRMSE_sto\tpreamble_mis\tpreamble_false\n")
        np.savetxt(f, save_var, delimiter="\t")


    np.savetxt(dest, save_var, delimiter="\t")

    # Read file:
    #     data = np.loadtxt('telecom\python\telecom\sim_results.txt')
    #     SNRs_dB = data[:,0]
    #     BERs = data[:,1]



    def create_simulation_csv(base_path, chain_name, sto_method, p_len, n_pkts, snr_dB, ber, per, rmse_cfo, rmse_sto):
        # On s'assure que le dossier existe
        target_dir = Path(base_path)
        target_dir.mkdir(parents=True, exist_ok=True) 

        # Nom du fichier
        filename = f"sim_{sto_method}_len{p_len}_pkts{n_pkts}_freqdev{chain.freq_dev}_ppd{PPD_algo}_ppd{chain.use_dynamic_ppd}.csv"
        full_path = target_dir / filename

        # On prépare les données
        save_var = np.column_stack((snr_dB, ber, per, rmse_cfo, rmse_sto))

        # Écriture
        with open(full_path, "w") as f:
            f.write(f"# METHOD: {sto_method} | PAYLOAD: {p_len} | PACKETS: {n_pkts} | FREQ_DEV: {chain.freq_dev} | PPD: {chain.use_dynamic_ppd}\n")
            f.write("# EsN0_dB\tBER\tPER\tRMSE_cfo\tRMSE_sto\n")
            np.savetxt(f, save_var, delimiter="\t", fmt="%.8e")
        
        # ON IMPRIME LE CHEMIN ABSOLU POUR LE TROUVER
        print(f"\n🚀 FICHIER CRÉÉ ICI : {full_path.absolute()}")
        return full_path
    
    create_simulation_csv(
        base_path=r"C:\Users\ismae\Desktop\UCLouvain\Master\Projet_Elec\LELEC210X-GroupA\telecom\python\telecom",
        chain_name=chain_name,
        sto_method=TYPE,
        p_len=chain.payload_len,
        n_pkts=chain.n_packets,
        snr_dB=EsN0s_dB,
        ber=BER,
        per=PER,
        rmse_cfo=RMSE_cfo,
        rmse_sto=RMSE_sto
    )
    def plot_simulations(data_folder="."):
        pathlist = Path(data_folder).glob("sim_*.csv")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        
        # Couleurs et styles pour différencier
        styles = {"ML": "-", "corr": "--"}
        markers = ["o", "s", "d", "x", "^"]

        found_files = False

        for i, path in enumerate(pathlist):
            found_files = True
            # Extraction du nom pour la légende (ex: ML_len100)
            # On utilise regex pour isoler la méthode et la longueur du nom de fichier
            #extraction deviation de fréquence
            dev_frq = re.search(r"freqdev([\d\.]+)", path.name)
            freq_dev = dev_frq.group(1) if dev_frq else "N/A"
            match = re.search(r"sim_(.*)_len(\d+)", path.name)
            if match:
                method, p_len = match.groups()
                label = f"{method} (L={p_len}, FreqDev={freq_dev})"
            else:
                label = path.name

            # Chargement des données (en sautant les commentaires)
            data = np.loadtxt(path)
            # print(data)
            snr_dB   = data[:, 0]
            ber      = data[:, 1]
            per      = data[:, 2]
            rmse_cfo = data[:, 3]
            rmse_sto = data[:, 4]

            # 1. Plot BER
            axes[0].semilogy(snr_dB, ber, marker=markers[i % 5], 
                            linestyle=styles.get(method, "-"), label=label)
            
            # 2. Plot PER
            axes[1].semilogy(snr_dB, per, marker=markers[i % 5], 
                            linestyle=styles.get(method, "-"), label=label)
            
            # 3. Plot RMSE STO
            axes[2].semilogy(snr_dB, rmse_sto, marker=markers[i % 5], 
                            linestyle=styles.get(method, "-"), label=label)
    
        axes[0].plot(EsN0_th, 1 - (1 - BER_th_noncoh) ** payload_len_sim, "k-.", label="AWGN Th. FSK non-coh.")
        axes[1].plot(EsN0_th, 1 - (1 - BER_th_noncoh) ** payload_len_sim, "k-.", label="AWGN Th. FSK non-coh.")

        if not found_files:
            print("❌ Aucun fichier sim_*.csv trouvé dans le dossier.")
            return

        # --- Mise en forme ---
        axes[0].set_title("Bit Error Rate (BER)")
        axes[0].set_ylabel("BER")
        axes[0].set_ylim(1e-6, 1)

        axes[1].set_title("Packet Error Rate (PER)")
        axes[1].set_ylabel("PER")

        axes[2].set_title("RMSE STO")
        axes[2].set_ylabel("RMSE (normalisé)")

        for ax in axes:
            ax.set_xlabel("$E_s/N_0$ [dB]")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.legend()

        plt.suptitle("Comparaison des performances de synchronisation STO", fontsize=16)
        plt.show()
    plot_simulations(data_folder=r"C:\Users\ismae\Desktop\UCLouvain\Master\Projet_Elec\LELEC210X-GroupA\telecom\python\telecom")


if __name__ == "__main__":
    main()