"""
RF Satellite Link Simulation
=============================
Converted from MathWorks MATLAB Communications Toolbox example 'commrfsatlink'.

Models the following RF impairments end-to-end:
  - HPA nonlinearity (Saleh TWTA), Free-space path loss, Doppler,
    Receiver thermal noise (AWGN), Phase noise (1/f),
    I/Q amplitude & phase imbalance, DC offsets.

HOW TO VERIFY THE RESULTS ARE CORRECT
---------------------------------------
1. FSPL        : printed value must match 20*log10(4*pi*d/lambda).
                 At 4 GHz / 35 600 km  →  expected ≈ 195.5 dB.
2. Antenna gain: G = (pi*D/lambda)^2.
                 For D=0.4 m, f=4 GHz  →  expected ≈ 24.4 dBi.
3. Eb/N0       : printed.  At T=20 K the link Eb/N0 ≈ 3 dB (marginal).
                 NOISE_TEMPERATURE_K=0  →  BER must be exactly 0.
4. BER sanity  : with T=0, backoff=30 dB, IQ_IMPAIRMENT='none' → BER = 0.
5. Power spectrum (Fig 1): Tx and Rx band shapes must match; Rx noise floor
                 elevated by kTB above Tx out-of-band floor.
6. AM/AM (Fig 2): low-amplitude region must be linear; saturates at high drive.
7. Constellation before/after HPA (Fig 3): at 30 dB backoff the two clouds
                 must overlap almost perfectly.
8. End-to-end constellation (Fig 4): at T=0, clusters sit on red crosses → BER=0.

Dependencies:  pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq, fftshift

# =============================================================================
# SECTION 0 – Simulation Parameters
# =============================================================================

# Satellite orbital altitude above the ground station [km]
SATELLITE_ALTITUDE_KM   = 35_600

# Downlink carrier frequency [MHz]
CARRIER_FREQ_MHZ        = 4_000

# Transmit and receive parabolic dish diameters [m]
TX_ANTENNA_DIAMETER_M   = 0.4
RX_ANTENNA_DIAMETER_M   = 0.4

# Effective system noise temperature [K].
# 0 = no noise (ideal), 20 = very low, 290 = room temperature, 500 = high.
# NOTE: at T=20 K the physical link Eb/N0 is only ~3 dB (marginal link).
# The simulation correctly reproduces this; use T=0 to verify BER=0.
NOISE_TEMPERATURE_K     = 0

# HPA input back-off from saturation [dB].
# 30 dB = negligible distortion, 7 dB = moderate, 1 dB = severe.
HPA_BACKOFF_DB          = 1

# Carrier frequency offset due to Doppler [Hz].  0 or 3.
DOPPLER_ERROR_HZ        = 0

# Phase noise PSD at 100 Hz offset [dBc/Hz].  -100=negligible, -55=low, -48=high.
PHASE_NOISE_DBCHZ       = -100

# I/Q impairment mode at the receiver.
# Options: "none", "amplitude_imbalance", "phase_imbalance",
#          "dc_inphase", "dc_quadrature"
IQ_IMPAIRMENT           = "none"

# Compensation block enable flags
ENABLE_DPD              = False   # Digital Pre-Distortion before the HPA
ENABLE_DC_CORRECTION    = False   # DC Blocker after matched filter
ENABLE_DOPPLER_CORRECTION = False # Carrier synchroniser (PLL)
ENABLE_IQ_CORRECTION    = False   # LMS I/Q imbalance compensator

# QAM-16: 4 bits per symbol
MODULATION_ORDER        = 16
BITS_PER_SYMBOL         = 4

# Oversampling factor: samples per QAM symbol in the waveform
SAMPLES_PER_SYMBOL      = 8

# RRC roll-off factor β (0–1); controls spectral width vs pulse smoothness
ROLLOFF                 = 0.25

# RRC filter length in symbol periods; longer → better ISI suppression
SPAN_SYMBOLS            = 10

# QAM symbol rate [symbols/s]
SYMBOL_RATE             = 1e6

# Waveform sample rate [samples/s]
SAMPLE_RATE             = SYMBOL_RATE * SAMPLES_PER_SYMBOL

# Reproducible random seed
RNG_SEED                = 42

# Number of QAM symbols to simulate
NUM_SYMBOLS             = 4_000

# =============================================================================
# SECTION 1 – Physical Constants and Link-Budget Calculations
# =============================================================================

# Speed of light [m/s]
SPEED_OF_LIGHT_MPS = 3e8

# Boltzmann constant [J/K]
BOLTZMANN_K = 1.380649e-23

# Carrier frequency in Hz
carrier_freq_hz = CARRIER_FREQ_MHZ * 1e6

# Wavelength λ = c / f  [m]
wavelength_m = SPEED_OF_LIGHT_MPS / carrier_freq_hz

# Satellite distance in metres
distance_m = SATELLITE_ALTITUDE_KM * 1e3

# Free-Space Path Loss (linear): FSPL = (4*pi*d/lambda)^2
# VERIFICATION: 10*log10(fspl_linear) must equal 195.5 dB at 4 GHz / 35600 km
fspl_linear = (4 * np.pi * distance_m / wavelength_m) ** 2


def dish_gain_linear(diameter_m):
    """
    Linear gain of a parabolic dish: G = (pi * D / lambda)^2.
    VERIFICATION: D=0.4 m, f=4 GHz → expected 24.4 dBi.
    """
    # Standard parabolic aperture gain formula (ideal aperture, no efficiency factor)
    return (np.pi * diameter_m / wavelength_m) ** 2


# Linear gains of Tx and Rx dishes
tx_gain = dish_gain_linear(TX_ANTENNA_DIAMETER_M)
rx_gain = dish_gain_linear(RX_ANTENNA_DIAMETER_M)

# Received signal power relative to transmitted (linear, dimensionless)
# = Tx_gain * Rx_gain / FSPL  (Friis transmission equation)
link_gain_linear = tx_gain * rx_gain / fspl_linear

# Eb/N0 at the receiver input (before matched filter gain):
# Eb/N0 = (P_rx / N0) / Rb  =  link_gain * Fs / (kT * Rb)
# where Rb = bit rate = SYMBOL_RATE * BITS_PER_SYMBOL
noise_psd_physical = BOLTZMANN_K * max(NOISE_TEMPERATURE_K, 1e-30)   # N0 = kT [W/Hz]
Rb = SYMBOL_RATE * BITS_PER_SYMBOL
EbN0_linear = link_gain_linear / (noise_psd_physical * Rb)

# Print link budget for manual verification
print("=" * 62)
print("RF Satellite Link – Python Simulation")
print("=" * 62)
print(f"  Carrier frequency    : {CARRIER_FREQ_MHZ} MHz")
print(f"  Wavelength           : {wavelength_m*100:.2f} cm")
print(f"  Satellite altitude   : {SATELLITE_ALTITUDE_KM} km")
print(f"  Free-space path loss : {10*np.log10(fspl_linear):.1f} dB  (expected ≈ 195.5 dB)")
print(f"  Tx antenna gain      : {10*np.log10(tx_gain):.1f} dBi  (expected ≈ 24.4 dBi for D=0.4 m)")
print(f"  Rx antenna gain      : {10*np.log10(rx_gain):.1f} dBi")
print(f"  Net link gain/loss   : {10*np.log10(link_gain_linear):.1f} dB")
if NOISE_TEMPERATURE_K > 0:
    print(f"  Eb/N0 at receiver    : {10*np.log10(EbN0_linear):.1f} dB  (QAM-16 needs ~15 dB for low BER)")
print(f"  Noise temperature    : {NOISE_TEMPERATURE_K} K")
print(f"  HPA back-off         : {HPA_BACKOFF_DB} dB")
print()

# =============================================================================
# SECTION 2 – Signal Processing Functions
# =============================================================================

def rrc_filter_coeffs(rolloff, span, sps):
    """
    Square-Root Raised Cosine (RRC) FIR filter coefficients.
    Two cascaded RRC filters (Tx + Rx) form a full Raised Cosine response
    which achieves zero ISI at the correct sampling instant.
    """
    # Total number of filter taps (symmetric around centre)
    n_taps = span * sps + 1
    # Time axis in symbol periods
    t = np.arange(-(span * sps) / 2, (span * sps) / 2 + 1) / sps
    h = np.zeros(n_taps)
    for i, ti in enumerate(t):
        # Special case t=0: avoid 0/0
        if ti == 0:
            h[i] = (1 - rolloff + 4 * rolloff / np.pi)
        # Special case |t|=1/(4*beta): avoid 0/0
        elif abs(ti) == 1 / (4 * rolloff):
            h[i] = (rolloff / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * rolloff))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * rolloff))
            )
        else:
            # General RRC formula
            num = np.sin(np.pi * ti * (1 - rolloff)) + 4 * rolloff * ti * np.cos(
                np.pi * ti * (1 + rolloff))
            den = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            h[i] = num / den
    # Normalise to unit energy so the filter preserves signal power
    h /= np.sqrt(np.sum(h ** 2))
    return h


def qam16_modulate(bits):
    """
    Map groups of 4 bits to Gray-coded QAM-16 complex symbols (unit average power).
    Gray coding: adjacent constellation points differ by exactly 1 bit.
    """
    # Reshape flat bit stream into groups of 4: [I_msb, I_lsb, Q_msb, Q_lsb]
    bits = bits.reshape(-1, 4)
    # 2-bit Gray code → amplitude level on each axis {-3, -1, +1, +3}
    gray_map = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}
    # Map first 2 bits to in-phase amplitude
    i_vals = np.array([gray_map[tuple(b)] for b in bits[:, :2]], dtype=float)
    # Map last 2 bits to quadrature amplitude
    q_vals = np.array([gray_map[tuple(b)] for b in bits[:, 2:]], dtype=float)
    # Combine and normalise: average symbol power = (9+9+1+1)/4 * 2 / 10 = 1
    return (i_vals + 1j * q_vals) / np.sqrt(10)


def qam16_ideal_constellation():
    """16 ideal QAM-16 reference positions at unit average power."""
    # Four equally spaced levels per axis, normalised to unit average power
    levels = np.array([-3, -1, 1, 3]) / np.sqrt(10)
    # All 16 I+jQ combinations
    return np.array([i + 1j * q for i in levels for q in levels])


def saleh_twta(x, input_power_backoff_db):
    """
    Saleh (1981) model for a Travelling Wave Tube Amplifier (TWTA).
    Applies AM/AM (amplitude compression) and AM/PM (phase rotation).
    """
    # Saleh model coefficients from the 1981 paper
    alpha_a, beta_a = 2.1587, 1.1517   # AM/AM
    alpha_p, beta_p = 4.0033, 9.1040   # AM/PM

    # Scale input so that average power equals the back-off point below saturation
    backoff_linear = 10 ** (input_power_backoff_db / 10)
    x_scaled = x / np.sqrt(backoff_linear)

    # Instantaneous envelope of the complex input signal
    r = np.abs(x_scaled)
    # AM/AM: output amplitude saturates at large r (nonlinear gain compression)
    am_am = (alpha_a * r) / (1 + beta_a * r ** 2)
    # AM/PM: phase rotation proportional to instantaneous power
    am_pm = (alpha_p * r ** 2) / (1 + beta_p * r ** 2)

    # Reconstruct complex output with new amplitude and added phase distortion
    return am_am * np.exp(1j * (np.angle(x_scaled) + am_pm))


def digital_predistortion(x, input_power_backoff_db):
    """
    Approximate inverse of the Saleh AM/AM and AM/PM functions.
    Pre-distorts the signal so the HPA output is more linear.
    Effective only at moderate back-off (≥7 dB).
    """
    alpha_a, beta_a = 2.1587, 1.1517
    alpha_p, beta_p = 4.0033, 9.1040
    backoff_linear = 10 ** (input_power_backoff_db / 10)
    x_scaled = x / np.sqrt(backoff_linear)
    r = np.abs(x_scaled)
    # Forward AM/AM gain at this amplitude
    gain_fwd = alpha_a / (1 + beta_a * r ** 2)
    # Inverse gain: boost what the HPA will compress
    dpd_gain = 1.0 / np.maximum(gain_fwd, 1e-6)
    # Forward AM/PM to pre-cancel
    phase_shift = (alpha_p * r ** 2) / (1 + beta_p * r ** 2)
    # Apply inverse amplitude scaling and negative phase rotation
    return x_scaled * dpd_gain * np.exp(-1j * phase_shift) * np.sqrt(backoff_linear)


def apply_phase_noise(signal, noise_level_dbchz, sample_rate, ref_freq=100.0):
    """
    Add 1/f (flicker / pink) phase noise to model oscillator instability.
    Generation method: Kasdin, N.J., Proc. IEEE, Vol. 83, No. 5, May 1995.
    """
    n = len(signal)
    # Convert dBc/Hz to linear power density [rad²/Hz]
    psd_linear = 10 ** (noise_level_dbchz / 10)
    # Total phase noise variance over the bandwidth
    variance = psd_linear * sample_rate / 2
    # Frequency axis (avoid DC singularity)
    freqs = fftfreq(n, d=1.0 / sample_rate)
    freqs[0] = 1e-6
    # White Gaussian noise in frequency domain
    white_fft = np.fft.fft(np.random.randn(n))
    # 1/f shaping: PSD ∝ 1/|f|
    shape = 1.0 / np.sqrt(np.abs(freqs) / ref_freq)

    shape[0] = 0.0
    # Transform back to time domain
    pink_noise = np.real(np.fft.ifft(white_fft * shape))
    # Scale to the desired variance
    pink_noise *= np.sqrt(variance / max(np.var(pink_noise), 1e-30))
    # Apply as a multiplicative phase rotation e^(j*phi(t))
    return signal * np.exp(1j * pink_noise)


def apply_iq_impairment(signal, mode):
    """
    Apply I/Q hardware impairment:
      'none'               – no impairment
      'amplitude_imbalance'– ±1.5 dB gain mismatch
      'phase_imbalance'    – ±10° phase mismatch
      'dc_inphase'         – 1e-8 DC on I branch
      'dc_quadrature'      – 5e-8 DC on Q branch
    """
    # Separate I and Q components
    I = np.real(signal)
    Q = np.imag(signal)
    if mode == "amplitude_imbalance":
        # Boost I by +1.5 dB, attenuate Q by -1.5 dB (linear amplitude)
        I = I * 10 ** ( 1.5 / 20)
        Q = Q * 10 ** (-1.5 / 20)
    elif mode == "phase_imbalance":
        # Rotate I by +10°, Q by -10°
        th = np.radians(10)
        I, Q = I * np.cos(th) - Q * np.sin(th), I * np.sin(th) + Q * np.cos(th)
    elif mode == "dc_inphase":
        I = I + 1e-8    # small DC bias on I (may not cause errors alone)
    elif mode == "dc_quadrature":
        Q = Q + 5e-8    # larger DC bias on Q (causes errors without compensation)
    return I + 1j * Q


def dc_blocker(signal, alpha=0.99):
    """
    First-order IIR high-pass filter to remove DC: H(z) = (1-z^-1)/(1-α*z^-1).
    Applied independently to I and Q.
    """
    b, a = [1, -1], [1, -alpha]
    return lfilter(b, a, signal.real) + 1j * lfilter(b, a, signal.imag)


def carrier_synchroniser(signal, samples_per_symbol, loop_bw=0.01):
    """
    Decision-directed PLL for carrier frequency/phase recovery.
    Tracks and removes Doppler-induced frequency offset.
    """
    corrected = np.zeros_like(signal)
    phase_est = 0.0
    freq_est  = 0.0
    alpha = loop_bw           # proportional gain
    beta  = loop_bw ** 2 / 4 # integral gain
    for k in range(len(signal)):
        # Correct current sample by the estimated phase
        corrected[k] = signal[k] * np.exp(-1j * phase_est)
        # Hard decision (sign-based slicer)
        s = corrected[k]
        decision = np.sign(s.real) + 1j * np.sign(s.imag)
        # Phase error between received sample and hard decision
        error = np.angle(s * np.conj(decision))
        # Update loop filter: integral tracks frequency, proportional tracks phase
        freq_est  += beta  * error
        phase_est += freq_est + alpha * error
    return corrected


def iq_imbalance_compensator(signal, mu=1e-5, n_iter=None):
    """
    Blind LMS I/Q imbalance compensator using a conjugate tap.
    Minimises the constant-modulus error to drive output to unit amplitude.
    """
    n_iter = n_iter or len(signal)
    # Weights: [direct, conjugate]; initialise as pass-through
    w = np.array([1.0 + 0j, 0.0 + 0j])
    corrected = np.zeros_like(signal)
    for k in range(len(signal)):
        # Two-tap input: direct and complex-conjugate
        x = np.array([signal[k], np.conj(signal[k])])
        y = np.dot(w, x)
        corrected[k] = y
        if k < n_iter:
            # CMA-like update: drive |y|² → 1
            error = y * (1 - np.abs(y) ** 2)
            w += mu * np.conj(error) * x
    return corrected


def compute_power_spectrum(signal, sample_rate, n_fft=2048):
    """
    Welch-method PSD estimate [dBW/Hz] vs frequency [MHz].
    Hanning window applied per block to reduce spectral leakage.
    """
    window   = np.hanning(n_fft)
    n_frames = len(signal) // n_fft
    psd      = np.zeros(n_fft)
    for i in range(n_frames):
        frame = signal[i * n_fft:(i + 1) * n_fft] * window
        psd  += np.abs(fftshift(fft(frame))) ** 2
    # Normalise by frame count and window energy
    psd /= n_frames * np.sum(window ** 2)
    freqs = fftshift(fftfreq(n_fft, d=1.0 / sample_rate))
    return freqs / 1e6, 10 * np.log10(psd + 1e-30)


# =============================================================================
# SECTION 3 – Satellite Downlink Transmitter
# =============================================================================

# Initialise RNG with fixed seed for reproducible results
rng = np.random.default_rng(RNG_SEED)

# Generate random binary bit stream (4 bits per QAM-16 symbol)
num_bits = NUM_SYMBOLS * BITS_PER_SYMBOL
tx_bits  = rng.integers(0, 2, size=num_bits)

# Map 4-bit groups to Gray-coded QAM-16 complex symbols
tx_symbols = qam16_modulate(tx_bits)

# Compute RRC filter coefficients (shared by Tx and Rx)
rrc_h = rrc_filter_coeffs(ROLLOFF, SPAN_SYMBOLS, SAMPLES_PER_SYMBOL)

# Upsample to waveform rate by inserting zeros between symbols
tx_up = np.zeros(NUM_SYMBOLS * SAMPLES_PER_SYMBOL, dtype=complex)
tx_up[::SAMPLES_PER_SYMBOL] = tx_symbols * np.sqrt(SAMPLES_PER_SYMBOL)

# Apply RRC transmit filter: pulse-shapes the signal to the required bandwidth
tx_rrc = np.convolve(tx_up, rrc_h, mode='same')

# Optionally pre-distort before the HPA to compensate AM/AM and AM/PM
tx_dpd = digital_predistortion(tx_rrc, HPA_BACKOFF_DB) if ENABLE_DPD else tx_rrc

# Save signal before HPA for constellation comparison (Plot 3)
signal_before_hpa = tx_dpd.copy()

# Apply Saleh TWTA nonlinearity (AM/AM + AM/PM distortion)
tx_hpa = saleh_twta(tx_dpd, HPA_BACKOFF_DB)
# Normalize the HPA output power back to 1 W (0 dBW)
# This prevents the TWTA backoff scaling from destroying the link budget
tx_hpa /= np.sqrt(np.mean(np.abs(tx_hpa) ** 2))
# Tx dish antenna gain: scale amplitude by sqrt(linear gain)
tx_after_antenna = tx_hpa * np.sqrt(tx_gain)

# =============================================================================
# SECTION 4 – Downlink Path
# =============================================================================

# Free-space path loss: attenuate amplitude by sqrt(FSPL) over 35 600 km
rx_after_fspl = tx_after_antenna / np.sqrt(fspl_linear)

# Doppler shift: multiply by complex phase ramp e^(j*2*pi*f_d*t)
t_samples  = np.arange(len(rx_after_fspl)) / SAMPLE_RATE
rx_doppler = rx_after_fspl * np.exp(1j * 2 * np.pi * DOPPLER_ERROR_HZ * t_samples)

# =============================================================================
# SECTION 5 – Ground Station Downlink Receiver
# =============================================================================

# Rx dish antenna gain: scale amplitude by sqrt(linear gain)
rx_after_rx_antenna = rx_doppler * np.sqrt(rx_gain)

# Thermal noise: AWGN added relative to the RECEIVED SIGNAL POWER.
# This matches the MATLAB Receiver Thermal Noise block behaviour.
# The noise power is computed from kTB and compared to the actual received
# signal power; the SNR is preserved correctly through subsequent AGC.
if NOISE_TEMPERATURE_K > 0:
    # Actual received signal power [W] (time-averaged over the waveform)
    rx_signal_power = np.mean(np.abs(rx_after_rx_antenna) ** 2)
    # Physical noise PSD N0 = k*T [W/Hz]; total noise power = N0 * sample_rate
    noise_power = BOLTZMANN_K * NOISE_TEMPERATURE_K * SAMPLE_RATE
    # Noise standard deviation per quadrature (I and Q each carry half the noise power)
    noise_std = np.sqrt(noise_power / 2)
    # Generate independent complex AWGN with the correct physical power
    thermal_noise = noise_std * (
        rng.standard_normal(len(rx_after_rx_antenna))
        + 1j * rng.standard_normal(len(rx_after_rx_antenna))
    )
    rx_noisy = rx_after_rx_antenna + thermal_noise
    # Print the actual SNR at this point for verification
    snr_db = 10 * np.log10(rx_signal_power / noise_power)
    print(f"  SNR at Rx input (before LNA) : {snr_db:.1f} dB")
else:
    # No noise: pass signal unchanged (use for BER=0 sanity check)
    rx_noisy = rx_after_rx_antenna.copy()
    print(f"  SNR at Rx input              : ∞ (no noise)")

# 1/f phase noise from the local oscillator
rx_phase_noise = apply_phase_noise(rx_noisy, PHASE_NOISE_DBCHZ, SAMPLE_RATE)

# I/Q hardware impairment (amplitude/phase mismatch or DC offset)
rx_iq_impaired = apply_iq_impairment(rx_phase_noise, IQ_IMPAIRMENT)

# LNA: 30 dB gain to raise the signal for baseband processing
# Amplitude gain = 10^(30/20) = 31.62×; does not change SNR
LNA_gain_lin = 10 ** (30 / 20)
rx_lna = rx_iq_impaired * LNA_gain_lin

# RRC receive filter (matched filter): maximises SNR at sampling instant,
# combined with Tx RRC gives full Raised Cosine → zero ISI
rx_mf = np.convolve(rx_lna, rrc_h, mode='same')

# Optional DC blocker: first-order IIR high-pass to remove DC offset
rx_dc_blocked = dc_blocker(rx_mf) if ENABLE_DC_CORRECTION else rx_mf.copy()

# AGC: normalise waveform to unit average power so decision thresholds are fixed
signal_power = np.mean(np.abs(rx_dc_blocked) ** 2)
rx_agc = rx_dc_blocked / np.sqrt(signal_power)

# Optional I/Q imbalance compensator (blind LMS)
rx_iq_comp = iq_imbalance_compensator(rx_agc) if ENABLE_IQ_CORRECTION else rx_agc.copy()

# Optional Doppler carrier synchroniser (decision-directed PLL)
rx_synced = carrier_synchroniser(rx_iq_comp, SAMPLES_PER_SYMBOL) if ENABLE_DOPPLER_CORRECTION else rx_iq_comp.copy()

# Downsample to symbol rate: take one sample per symbol.
# Set delay to 0 because np.convolve(mode='same') already centers the sequence
delay   = 0
skip = SPAN_SYMBOLS // 2
rx_syms = rx_synced[delay::SAMPLES_PER_SYMBOL][skip:NUM_SYMBOLS - skip]

# Final symbol-power normalisation so the constellation spans the correct range
rx_syms /= np.sqrt(np.mean(np.abs(rx_syms) ** 2))

# Hard decision: assign each received symbol to the nearest ideal constellation point
ideal_const = qam16_ideal_constellation()
rx_decided  = np.array(
    [ideal_const[np.argmin(np.abs(s - ideal_const))] for s in rx_syms]
)

# =============================================================================
# SECTION 6 – Bit Error Rate (BER)
# =============================================================================

def qam16_demodulate_ideal(symbols):
    """Inverse Gray-code demap: ideal QAM-16 symbols → bits."""
    levels_to_bits = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}
    inv_norm = np.sqrt(10)
    bits_out = []
    for s in symbols:
        # De-normalise and round to nearest amplitude level {-3,-1,+1,+3}
        i_lvl = int(round(s.real * inv_norm))
        q_lvl = int(round(s.imag * inv_norm))
        bits_out.extend(levels_to_bits.get(i_lvl, (0, 0)))
        bits_out.extend(levels_to_bits.get(q_lvl, (0, 0)))
    return np.array(bits_out)

# Decode hard-decision symbols back to bits
rx_bits = qam16_demodulate_ideal(rx_decided)

# The Tx and Rx bit streams must be aligned: the filter delay shifts the Rx
# Set delay_bits to 0 because delay is handled by mode='same'
delay_bits = skip * BITS_PER_SYMBOL
tx_bits_aligned = tx_bits[delay_bits: delay_bits + len(rx_bits)]
n_compare  = min(len(tx_bits_aligned), len(rx_bits))
bit_errors = np.sum(tx_bits_aligned[:n_compare] != rx_bits[:n_compare])
ber        = bit_errors / n_compare

print()
print("Simulation Results")
print("------------------")
print(f"  Symbols simulated  : {NUM_SYMBOLS}")
print(f"  Bits compared      : {n_compare}")
print(f"  Bit errors         : {bit_errors}")
print(f"  BER                : {ber:.2e}")
if NOISE_TEMPERATURE_K == 0 and HPA_BACKOFF_DB >= 30:
    print(f"  Expected BER       : 0.00e+00  ✓" if ber == 0 else
          f"  Expected BER       : 0.00e+00  ✗ (alignment problem)")
print()

# =============================================================================
# SECTION 7 – Plots (one figure per MATLAB scope)
# =============================================================================

# One-line parameter summary for every figure footer
params_str = (
    f"Altitude={SATELLITE_ALTITUDE_KM} km | Freq={CARRIER_FREQ_MHZ} MHz | "
    f"Noise T={NOISE_TEMPERATURE_K} K | HPA back-off={HPA_BACKOFF_DB} dB | "
    f"BER={ber:.2e} | I/Q={IQ_IMPAIRMENT}"
)

# ── FIGURE 1: Tx vs Rx Power Spectrum ─────────────────────────────────────
fig1, ax = plt.subplots(figsize=(9, 5))

# Compute PSD of the transmitted waveform (after RRC, before HPA)
f_tx, psd_tx = compute_power_spectrum(tx_rrc, SAMPLE_RATE)
# Compute PSD of the received waveform (after all impairments, before downsampling)
f_rx, psd_rx = compute_power_spectrum(rx_synced, SAMPLE_RATE)

# Gold = Tx, steel blue = Rx (matches MATLAB scope colours)
ax.plot(f_tx, psd_tx, color='gold',      linewidth=1.2,
        label='Tx — after RRC pulse shaping, before HPA')
ax.plot(f_rx, psd_rx, color='steelblue', linewidth=1.2,
        label='Rx — after all RF impairments, before demodulation')

ax.set_title("Power Spectrum: Transmitted vs Received Signal", fontsize=12, fontweight='bold')
ax.set_xlabel("Frequency offset from carrier [MHz]", fontsize=10)
ax.set_ylabel("Power Spectral Density [dBW/Hz]", fontsize=10)
ax.set_xlim([-2, 2])
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
fig1.text(0.5, 0.01, params_str, ha='center', fontsize=7, color='gray')
fig1.tight_layout(rect=[0, 0.04, 1, 1])
fig1.savefig("plot1_power_spectrum.png", dpi=150, bbox_inches='tight')

# ── FIGURE 2: HPA AM/AM and AM/PM ─────────────────────────────────────────
fig2, ax_am = plt.subplots(figsize=(9, 5))

# Sweep normalised input amplitude from 0 to 1.5
r_in = np.linspace(0, 1.5, 500)
# Scale to the operating point given the back-off
x_sw = r_in / np.sqrt(10 ** (HPA_BACKOFF_DB / 10))

# Saleh model coefficients
alpha_a, beta_a = 2.1587, 1.1517
alpha_p, beta_p = 4.0033, 9.1040

# AM/AM: output amplitude vs input amplitude (saturates at high drive)
am_am = (alpha_a * x_sw) / (1 + beta_a * x_sw ** 2)
# AM/PM: phase rotation [degrees] vs input amplitude
am_pm = np.degrees((alpha_p * x_sw ** 2) / (1 + beta_p * x_sw ** 2))

# Second y-axis (right) for the phase curve
ax_pm = ax_am.twinx()
ax_am.plot(r_in, am_am, color='tab:blue',   linewidth=2, label='AM/AM')
ax_pm.plot(r_in, am_pm, color='tab:orange', linewidth=2, linestyle='--', label='AM/PM')

ax_am.set_title(f"HPA AM/AM and AM/PM Characteristics  (back-off = {HPA_BACKOFF_DB} dB)",
                fontsize=12, fontweight='bold')
ax_am.set_xlabel("Input amplitude [normalised, 1.0 = saturation]", fontsize=10)
ax_am.set_ylabel("Output amplitude [normalised]", color='tab:blue', fontsize=10)
ax_pm.set_ylabel("Phase shift [degrees]", color='tab:orange', fontsize=10)
ax_am.legend(loc='upper left',  fontsize=9)
ax_pm.legend(loc='lower right', fontsize=9)
ax_am.grid(True, linestyle='--', alpha=0.5)
fig2.text(0.5, 0.01, params_str, ha='center', fontsize=7, color='gray')
fig2.tight_layout(rect=[0, 0.04, 1, 1])
fig2.savefig("plot2_hpa_am_am_am_pm.png", dpi=150, bbox_inches='tight')

# ── FIGURE 3: Constellation Before and After HPA ──────────────────────────
fig3, ax = plt.subplots(figsize=(7, 7))

# Set plot downsampling delay to 0 as well
pre_syms  = signal_before_hpa[0::SAMPLES_PER_SYMBOL][:NUM_SYMBOLS]
post_syms = tx_hpa[0::SAMPLES_PER_SYMBOL][:NUM_SYMBOLS]
# Normalise both to unit average power for a fair visual comparison
pre_syms  /= np.sqrt(np.mean(np.abs(pre_syms)  ** 2))
post_syms /= np.sqrt(np.mean(np.abs(post_syms) ** 2))

ax.scatter(pre_syms.real,  pre_syms.imag,
           s=5, color='gold',      alpha=0.5, label='Before HPA (Tx RRC output)')
ax.scatter(post_syms.real, post_syms.imag,
           s=5, color='steelblue', alpha=0.5, label='After HPA (TWTA output)')

ax.set_title(f"Constellation Before and After HPA  (back-off = {HPA_BACKOFF_DB} dB)",
             fontsize=12, fontweight='bold')
ax.set_xlabel("In-phase amplitude I [normalised]", fontsize=10)
ax.set_ylabel("Quadrature amplitude Q [normalised]", fontsize=10)
ax.legend(fontsize=9, markerscale=4)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')
fig3.text(0.5, 0.01, params_str, ha='center', fontsize=7, color='gray')
fig3.tight_layout(rect=[0, 0.04, 1, 1])
fig3.savefig("plot3_constellation_hpa.png", dpi=150, bbox_inches='tight')

# ── FIGURE 4: End-to-End Received Constellation ───────────────────────────
fig4, ax = plt.subplots(figsize=(7, 7))

# Received symbol cloud after all impairments and receiver processing
ax.scatter(rx_syms.real, rx_syms.imag,
           s=6, color='gold', alpha=0.5,
           label=f'Received symbols  (BER = {ber:.2e})')
# Ideal QAM-16 reference grid
ax.scatter(ideal_const.real, ideal_const.imag,
           s=150, color='red', zorder=5, marker='x', linewidths=2.5,
           label='Ideal QAM-16 reference (decision centres)')

ax.set_title(f"End-to-End Received QAM-16 Constellation  (BER = {ber:.2e})",
             fontsize=12, fontweight='bold')
ax.set_xlabel("In-phase amplitude I [normalised]", fontsize=10)
ax.set_ylabel("Quadrature amplitude Q [normalised]", fontsize=10)
ax.legend(fontsize=9, markerscale=2)
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_aspect('equal')
fig4.text(0.5, 0.01, params_str, ha='center', fontsize=7, color='gray')
fig4.tight_layout(rect=[0, 0.04, 1, 1])
fig4.savefig("plot4_received_constellation.png", dpi=150, bbox_inches='tight')

print("Figures saved:")
print("  plot1_power_spectrum.png")
print("  plot2_hpa_am_am_am_pm.png")
print("  plot3_constellation_hpa.png")
print("  plot4_received_constellation.png")

plt.show()