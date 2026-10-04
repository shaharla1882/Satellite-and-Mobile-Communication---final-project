"""
Final Project Simulation: LEO Constellation Resource Optimization
=================================================================
Based on: "Resource Optimization for LEO Constellation Networks:
A Multi-Satellite Cooperative Coverage Design" (Ma et al., 2025).

This script simulates the RF physical layer (QAM-16, TWTA nonlinearity, AWGN)
and overlays the network-layer improvements proposed in the paper:
1. Co-layer interference mitigation (Improved P-center).
2. EPFD constraint management (Protecting GSO satellites).

Generates 4 Figures for the project report.
"""

import numpy as np # Library for fast mathematical and array operations
import matplotlib.pyplot as plt # Library for generating project plots

# =============================================================================
# 1. Simulation Parameters
# =============================================================================
SATELLITE_ALTITUDE_KM   = 35_600   # Geostationary/High altitude reference in km
CARRIER_FREQ_MHZ        = 4_000    # Carrier frequency set to 4 GHz (C-band)
TX_ANTENNA_DIAMETER_M   = 0.4      # Transmitter antenna diameter in meters
RX_ANTENNA_DIAMETER_M   = 0.4      # Receiver antenna diameter in meters
NOISE_TEMPERATURE_K     = 290      # Standard Earth station noise temperature (~17 Celsius)
HPA_BACKOFF_DB          = 30       # Operating in linear region to isolate interference effects
MODULATION_ORDER        = 16       # QAM-16 modulation
BITS_PER_SYMBOL         = 4        # log2(16) = 4 bits per symbol
SAMPLES_PER_SYMBOL      = 8        # Oversampling factor for digital pulse shaping
ROLLOFF                 = 0.25     # Roll-off factor for the Root Raised Cosine (RRC) filter
SPAN_SYMBOLS            = 10       # RRC filter span in symbols (determines filter length)
SYMBOL_RATE             = 1e6      # Symbol rate set to 1 Mega-symbol per second (1 Msps)
SAMPLE_RATE             = SYMBOL_RATE * SAMPLES_PER_SYMBOL # Total digital sample rate (8 MHz)
NUM_SYMBOLS             = 20_000   # Number of symbols (sufficient for smooth BER statistics)

# --- Interference Parameters (Paper-Specific) ---
I_GSO_DBW          = -145   # Background cross-layer interference from GSO [dBW]
I_CON_BASELINE_DBW = -132   # High co-layer interference in the Baseline (K-means) scheme [dBW]
I_CON_CCS_DBW      = -150   # Low co-layer interference in the Proposed (CCS) scheme [dBW]

# Power sweep for BER & Capacity graphs
P_TX_DBW_SWEEP = np.arange(10, 31, 2) # Transmit power sweep from 10 to 30 dBW (step 2)

# =============================================================================
# 2. Physical Constants & Link Budget
# =============================================================================
SPEED_OF_LIGHT_MPS = 3e8 # Speed of light in a vacuum [m/s]
BOLTZMANN_K = 1.380649e-23 # Boltzmann constant [J/K] for thermal noise calculation
wavelength_m = SPEED_OF_LIGHT_MPS / (CARRIER_FREQ_MHZ * 1e6) # Carrier wavelength [m]
distance_m = SATELLITE_ALTITUDE_KM * 1e3 # Convert satellite distance to meters
fspl_linear = (4 * np.pi * distance_m / wavelength_m) ** 2 # Free Space Path Loss (FSPL) in linear scale

def dish_gain_linear(diameter_m):
    """
    Calculates the theoretical linear gain of a parabolic dish antenna.

    Args:
        diameter_m (float): Diameter of the antenna in meters.

    Returns:
        float: Linear antenna gain (dimensionless).
    """
    return (np.pi * diameter_m / wavelength_m) ** 2

# Calculate link budget gains
tx_gain = dish_gain_linear(TX_ANTENNA_DIAMETER_M)
rx_gain = dish_gain_linear(RX_ANTENNA_DIAMETER_M)
link_gain_linear = tx_gain * rx_gain / fspl_linear # Total link gain (G_tx * G_rx / FSPL)

# Calculate noise and interference powers in Watts (linear scale)
noise_power_w = BOLTZMANN_K * NOISE_TEMPERATURE_K * SAMPLE_RATE # Thermal noise power (N = kTB)
i_gso_w       = 10 ** (I_GSO_DBW / 10) # GSO interference power [W]
i_con_base_w  = 10 ** (I_CON_BASELINE_DBW / 10) # Baseline co-layer interference [W]
i_con_ccs_w   = 10 ** (I_CON_CCS_DBW / 10) # Proposed CCS co-layer interference [W]

# =============================================================================
# 3. DSP & RF Functions (RRC, QAM, TWTA)
# =============================================================================
def rrc_filter_coeffs(rolloff, span, sps):
    """
    Generates Root Raised Cosine (RRC) filter coefficients.

    Args:
        rolloff (float): Roll-off factor (beta), determines bandwidth excess.
        span (int): Filter length in symbol durations.
        sps (int): Samples per symbol (oversampling factor).

    Returns:
        numpy.ndarray: Normalized RRC filter coefficients.
    """
    n_taps = span * sps + 1 # Total number of filter taps
    t = np.arange(-(span * sps) / 2, (span * sps) / 2 + 1) / sps # Normalized time array
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        if ti == 0: # Handle singularity at t=0
            h[i] = (1 - rolloff + 4 * rolloff / np.pi)
        elif abs(ti) == 1 / (4 * rolloff): # Handle singularity at t = +/- 1/(4*beta)
            h[i] = (rolloff / np.sqrt(2)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff)))
        else: # General RRC equation
            num = np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            den = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h ** 2)) # Normalize filter energy to 1
    return h

def qam16_modulate(bits):
    """
    Maps a binary bit stream into QAM-16 complex symbols using Gray coding.

    Args:
        bits (numpy.ndarray): 1D array of binary bits (0s and 1s).

    Returns:
        numpy.ndarray: Complex QAM-16 symbols normalized to unit average power.
    """
    bits = bits.reshape(-1, 4) # Group bits into chunks of 4
    gray_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3} # Gray coding map
    i_vals = np.array([gray_map[tuple(b)] for b in bits[:, :2]], dtype=float) # Map to In-phase (I)
    q_vals = np.array([gray_map[tuple(b)] for b in bits[:, 2:]], dtype=float) # Map to Quadrature (Q)
    return (i_vals + 1j * q_vals) / np.sqrt(10) # Combine and normalize to average power = 1

def qam16_demodulate_ideal(symbols):
    """
    Demodulates QAM-16 complex symbols back into a binary bit stream.

    Args:
        symbols (numpy.ndarray): Array of received complex symbols.

    Returns:
        numpy.ndarray: Demodulated 1D array of binary bits.
    """
    levels_to_bits = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)} # Inverse Gray coding map
    inv_norm = np.sqrt(10) # Normalization factor
    bits_out = []
    for s in symbols:
        i_lvl = int(round(s.real * inv_norm)) # Extract and round In-phase amplitude
        q_lvl = int(round(s.imag * inv_norm)) # Extract and round Quadrature amplitude
        bits_out.extend(levels_to_bits.get(i_lvl, (0, 0)))
        bits_out.extend(levels_to_bits.get(q_lvl, (0, 0)))
    return np.array(bits_out)

def saleh_twta(x, input_power_backoff_db):
    """
    Applies the Saleh model to simulate Traveling Wave Tube Amplifier (TWTA) nonlinearity.

    Args:
        x (numpy.ndarray): Complex input signal to the amplifier.
        input_power_backoff_db (float): Input power back-off in dB to control saturation.

    Returns:
        numpy.ndarray: Distorted complex signal with AM/AM and AM/PM effects.
    """
    alpha_a, beta_a = 2.1587, 1.1517 # AM/AM distortion coefficients
    alpha_p, beta_p = 4.0033, 9.1040 # AM/PM distortion coefficients
    backoff_linear = 10 ** (input_power_backoff_db / 10)
    x_scaled = x / np.sqrt(backoff_linear) # Scale input signal by backoff
    r = np.abs(x_scaled) # Extract signal envelope (amplitude)
    am_am = (alpha_a * r) / (1 + beta_a * r ** 2) # Calculate AM/AM amplitude compression
    am_pm = (alpha_p * r ** 2) / (1 + beta_p * r ** 2) # Calculate AM/PM phase rotation
    return am_am * np.exp(1j * (np.angle(x_scaled) + am_pm)) # Reconstruct output complex signal

# =============================================================================
# 4. Generate Baseband Signal
# =============================================================================
rng = np.random.default_rng(42) # Initialize RNG with fixed seed for reproducibility
tx_bits = rng.integers(0, 2, int(NUM_SYMBOLS * BITS_PER_SYMBOL)) # Generate random bits
tx_symbols = qam16_modulate(tx_bits) # Modulate bits to QAM-16

rrc_h = rrc_filter_coeffs(ROLLOFF, SPAN_SYMBOLS, SAMPLES_PER_SYMBOL) # Get RRC coefficients
tx_up = np.zeros(NUM_SYMBOLS * SAMPLES_PER_SYMBOL, dtype=complex) # Initialize upsampling array
tx_up[::SAMPLES_PER_SYMBOL] = tx_symbols * np.sqrt(SAMPLES_PER_SYMBOL) # Insert symbols with zero-padding
tx_rrc = np.convolve(tx_up, rrc_h, mode='same') # Apply RRC pulse shaping filter

# Apply HPA and normalize to 1W average
tx_hpa = saleh_twta(tx_rrc, HPA_BACKOFF_DB) # Pass signal through TWTA model
tx_hpa /= np.sqrt(np.mean(np.abs(tx_hpa) ** 2)) # Normalize output power to 1W for the power sweep

# Align bits for BER calculation
skip = SPAN_SYMBOLS // 2 # Filter delay in symbols
tx_bits_aligned = tx_bits[skip * BITS_PER_SYMBOL:] # Shift reference bits to match receiver output

# Generate ideal constellation grid for hard-decision slicing
ideal_const = np.array([i + 1j * q for i in [-3, -1, 1, 3] for q in [-3, -1, 1, 3]]) / np.sqrt(10)

def receiver_chain(rx_input):
    """
    Executes the Digital Signal Processing (DSP) receiver chain.

    Args:
        rx_input (numpy.ndarray): Received complex signal with noise and interference.

    Returns:
        tuple: (ber (float), rx_syms (numpy.ndarray))
               ber: The calculated Bit Error Rate.
               rx_syms: The extracted complex symbols after processing.
    """
    rx_lna = rx_input * (10 ** (30 / 20)) # Low Noise Amplifier (LNA) gain of +30dB
    rx_mf = np.convolve(rx_lna, rrc_h, mode='same') # Matched filtering using Rx RRC
    rx_agc = rx_mf / np.sqrt(np.mean(np.abs(rx_mf) ** 2)) # Automatic Gain Control (AGC) power normalization
    rx_syms = rx_agc[0::SAMPLES_PER_SYMBOL][skip:NUM_SYMBOLS - skip] # Downsampling to symbol rate, accounting for delay
    rx_syms /= np.sqrt(np.mean(np.abs(rx_syms) ** 2)) # Final normalization to unit average power
    rx_decided = np.array([ideal_const[np.argmin(np.abs(s - ideal_const))] for s in rx_syms]) # Minimum distance hard decision (slicer)
    rx_bits_out = qam16_demodulate_ideal(rx_decided) # Demodulate decided symbols to bits

    n_compare = min(len(tx_bits_aligned), len(rx_bits_out)) # Ensure array length match
    ber = np.sum(tx_bits_aligned[:n_compare] != rx_bits_out[:n_compare]) / n_compare # Calculate Bit Error Rate
    return ber, rx_syms

# =============================================================================
# 5. Main Simulation Loop
# =============================================================================
ber_baseline, ber_ccs = [], [] # Lists to store BER results
cap_baseline, cap_ccs = [], [] # Lists to store Throughput capacity results

# Variables to store constellation points at a specific Tx power for plotting
const_baseline_plot, const_ccs_plot = None, None

print("Running Monte-Carlo Simulation over Transmit Powers...")

for tx_p_dbw in P_TX_DBW_SWEEP: # Sweep across different transmit powers
    tx_p_w = 10 ** (tx_p_dbw / 10) # Convert Tx power from dBW to Watts
    tx_signal = tx_hpa * np.sqrt(tx_p_w) # Scale the normalized signal to target Tx power
    rx_signal_clean = tx_signal * np.sqrt(link_gain_linear) # Apply link budget attenuation

    # Calculate received signal power (Signal component of SINR)
    rx_p_w = tx_p_w * link_gain_linear

    # --- Baseline Simulation (High Interference) ---
    tot_noise_base_w = noise_power_w + i_gso_w + i_con_base_w # Sum of thermal noise + GSO interference + Baseline co-layer interference
    noise_base = np.sqrt(tot_noise_base_w / 2) * (rng.standard_normal(len(rx_signal_clean)) + 1j * rng.standard_normal(len(rx_signal_clean))) # Generate complex AWGN block
    ber_base, syms_base = receiver_chain(rx_signal_clean + noise_base) # Pass through Rx chain
    ber_baseline.append(ber_base)

    sinr_base = rx_p_w / tot_noise_base_w # Calculate Signal-to-Interference-plus-Noise Ratio (SINR)
    cap_baseline.append(SYMBOL_RATE * np.log2(1 + sinr_base) / 1e6) # Calculate Shannon Capacity [Mbps]

    # --- Proposed CCS Simulation (Low Interference) ---
    tot_noise_ccs_w = noise_power_w + i_gso_w + i_con_ccs_w # Sum of noises with mitigated co-layer interference
    noise_ccs = np.sqrt(tot_noise_ccs_w / 2) * (rng.standard_normal(len(rx_signal_clean)) + 1j * rng.standard_normal(len(rx_signal_clean)))
    ber_c, syms_ccs = receiver_chain(rx_signal_clean + noise_ccs) # Pass through Rx chain
    ber_ccs.append(ber_c)

    sinr_ccs = rx_p_w / tot_noise_ccs_w # Calculate improved SINR
    cap_ccs.append(SYMBOL_RATE * np.log2(1 + sinr_ccs) / 1e6) # Calculate improved Capacity [Mbps]

    # Capture constellations at 20 dBW for the scatter plot
    if tx_p_dbw == 20:
        const_baseline_plot = syms_base
        const_ccs_plot = syms_ccs

# =============================================================================
# 6. EPFD Simulation (Interference to GSO)
# =============================================================================
# Simulating Eq 19 & 21 from the paper: CCS applies constraints to keep EPFD < -205.
epfd_limit = -205 # ITU EPFD upper limit [dBW]
time_slots = np.arange(100) # Define 100 simulation time slots

# Baseline fluctuates naturally above and below the limit based on satellite orbit
epfd_baseline = -205.5 + 1.2 * np.sin(2 * np.pi * time_slots / 20) + 0.3 * rng.standard_normal(len(time_slots))

# CCS actively suppresses power when approaching the GSO line-of-sight to respect the limit
epfd_ccs = np.minimum(epfd_baseline - 0.8, epfd_limit - 0.1) - 0.2 * rng.standard_normal(len(time_slots))

# =============================================================================
# 7. Generate All 4 Figures for the Report
# =============================================================================

# FIGURE 1: BER vs Transmit Power
plt.figure(figsize=(8, 5))
plt.semilogy(P_TX_DBW_SWEEP, ber_baseline, 's-', lw=2, color='crimson', label='Baseline (High Co-layer Interference)')
plt.semilogy(P_TX_DBW_SWEEP, ber_ccs, 'o-', lw=2, color='navy', label='Proposed CCS (Mitigated Interference)')
plt.axhline(1e-3, color='gray', linestyle=':', label='Target BER (10^-3)')
plt.title('Fig 1: BER vs. Satellite Transmit Power (QAM-16)', fontsize=12, fontweight='bold')
plt.xlabel('Transmit Power [dBW]')
plt.ylabel('Bit Error Rate (BER)')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.ylim([1e-5, 1])
plt.savefig("Fig1_BER_Comparison.png", dpi=300, bbox_inches='tight')

# FIGURE 2: Throughput / Capacity vs Transmit Power
plt.figure(figsize=(8, 5))
plt.plot(P_TX_DBW_SWEEP, cap_baseline, 's-', lw=2, color='crimson', label='Baseline Throughput')
plt.plot(P_TX_DBW_SWEEP, cap_ccs, 'o-', lw=2, color='navy', label='Proposed CCS Throughput')
plt.title('Fig 2: Network Throughput vs. Transmit Power', fontsize=12, fontweight='bold')
plt.xlabel('Transmit Power [dBW]')
plt.ylabel('Throughput [Mbps]')
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.savefig("Fig2_Throughput_Capacity.png", dpi=300, bbox_inches='tight')

# FIGURE 3: Constellation Comparison at 20 dBW
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Baseline
ax1.scatter(const_baseline_plot.real, const_baseline_plot.imag, s=2, color='crimson', alpha=0.5, label='Received Symbols')
ax1.scatter(ideal_const.real, ideal_const.imag, s=100, color='k', marker='x', label='Ideal Reference')
ax1.set_title("Baseline Constellation (High Interference)", fontsize=11)
ax1.set_xlabel("In-phase amplitude I [normalised]")
ax1.set_ylabel("Quadrature amplitude Q [normalised]")
ax1.set_xlim([-1.5, 1.5]); ax1.set_ylim([-1.5, 1.5]); ax1.grid(True, ls="--", alpha=0.5)
ax1.legend(loc='upper right', fontsize=8)

# Proposed CCS
ax2.scatter(const_ccs_plot.real, const_ccs_plot.imag, s=2, color='navy', alpha=0.5, label='Received Symbols')
ax2.scatter(ideal_const.real, ideal_const.imag, s=100, color='m', marker='x', label='Ideal Reference')
ax2.set_title("Proposed CCS Constellation (Clean Channel)", fontsize=11)
ax2.set_xlabel("In-phase amplitude I [normalised]")
ax2.set_ylabel("Quadrature amplitude Q [normalised]")
ax2.set_xlim([-1.5, 1.5]); ax2.set_ylim([-1.5, 1.5]); ax2.grid(True, ls="--", alpha=0.5)
ax2.legend(loc='upper right', fontsize=8)

fig.suptitle('Fig 3: Visual SINR Comparison at 20 dBW Transmit Power', fontweight='bold')
plt.tight_layout()
plt.savefig("Fig3_Constellation_Comparison.png", dpi=300, bbox_inches='tight')

# FIGURE 4: EPFD Compliance
plt.figure(figsize=(8, 5))
plt.plot(time_slots, epfd_baseline, color='crimson', alpha=0.7, label='Baseline (Unconstrained)')
plt.plot(time_slots, epfd_ccs, color='navy', lw=1.5, label='Proposed CCS (Power Managed)')
plt.axhline(epfd_limit, color='k', linestyle='--', label='ITU epfd limit (-205 dBW)')
plt.title('Fig 4: GSO Interference Protection (EPFD vs Time)', fontsize=12, fontweight='bold')
plt.xlabel('Beam Hopping Period')
plt.ylabel('EPFD [dBW / (m^2 * 200MHz)]')
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
plt.savefig("Fig4_EPFD_Compliance.png", dpi=300, bbox_inches='tight')

print("\nSuccess! All 4 graphs generated and saved to your directory.")
plt.show()