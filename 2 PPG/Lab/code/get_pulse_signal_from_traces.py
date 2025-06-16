# START OF FILE get_pulse_signal_from_traces.py ---

# Yannick Benezeth - yannick.benezeth@u-bourgogne.fr
# M1 Vibot - Univ Bourgogne
# 2024 - 2025

import argparse
import os
import time
import cv2
import numpy as np
from scipy.io import savemat, loadmat
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from utils import load_ppg

def normalize_trace(trace):
    """Applies zero-mean, unit-variance normalization to a trace."""
    mean_val = np.mean(trace)
    std_val = np.std(trace)
    epsilon = 1e-9
    if std_val > epsilon:
        normalized = (trace - mean_val) / std_val
    else:
        # Avoid division by zero, just center the trace
        normalized = trace - mean_val
    return normalized

def apply_bandpass_filter(trace, low_hz, high_hz, fs, order):
    """Applies a Butterworth bandpass filter to a trace."""
    nyquist = 0.5 * fs
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist
    filtered_trace = trace

    # Check frequency validity
    if low_norm >= high_norm:
        print(f"Error: Invalid frequency range for filter ({low_hz}-{high_hz} Hz relative to fs={fs} Hz). Low >= High. Returning unfiltered trace.")
        return filtered_trace

    try:
        b, a = signal.butter(order, [low_norm, high_norm], btype='bandpass')
        filtered_trace = signal.filtfilt(b, a, trace)
    except ValueError as e:
         print(f"Error applying bandpass filter: {e}. Returning unfiltered trace.")
         print(f"  Filter params: order={order}, low_norm={low_norm:.3f}, high_norm={high_norm:.3f}, fs={fs:.2f}")

    return filtered_trace

def get_pulse_signal_from_traces(vid_folder, out_folder, verbose, method='GREEN'):
    print(f"Call get_pulse_signal_from_trace:")
    print(f"  VIDFOLDER: {vid_folder}")
    print(f"  OUTFOLDER: {out_folder}")
    print(f"  METHOD:    {method}")
    print(f"  VERBOSE:   {verbose}")
    print("-" * 20)

    start_time_func = time.time()

    pulse_trace_final = np.array([]) # Initialize

    # --- Parameters
    mat_file_name = 'rgbTraces.mat'
    pulse_out_file_name = 'pulseTrace.mat'

    # --- Load Ground Truth PPG ---
    gt_trace, gt_hr, gt_time = load_ppg(vid_folder)

    # --- Load RGB Traces ---
    in_mat_path = os.path.join(out_folder, mat_file_name)
    try:
        mat_data = loadmat(in_mat_path)
        if 'rgbTraces' not in mat_data:
             print(f"Error: 'rgbTraces' key not found in {in_mat_path}")
             return 
        rgb_traces = mat_data['rgbTraces']
        print(f"Loaded rgbTraces with shape: {rgb_traces.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {in_mat_path}")
        return
    except Exception as e:
        print(f"Error loading {in_mat_path}: {e}")
        return

    # Check shape after loading
    if rgb_traces.shape[0] < 4:
        print(f"Error: rgbTraces expected at least 4 rows, found {rgb_traces.shape[0]}")
        return

    # Get traces and time vector
    R_trace = rgb_traces[0, :]
    G_trace = rgb_traces[1, :]
    B_trace = rgb_traces[2, :]
    time_trace = rgb_traces[3, :] / 1000.0 # Convert ms to seconds
    trace_size = R_trace.shape[0] # Use length of one trace


    # --- Common Parameters ---
    low_cutoff_hz = 0.7  # Hz (approx 42 bpm)
    high_cutoff_hz = 3.5 # Hz (approx 210 bpm)
    filter_order = 5
    epsilon = 1e-9

    # Calculate Sampling Frequency
    time_diffs = np.diff(time_trace)
    valid_diffs = time_diffs[time_diffs > epsilon]
    if len(valid_diffs) < 1:
        print("Warning: Cannot reliably determine sampling frequency. Assuming 30Hz.")
        mean_interval = 1/30.0 # Fallback guess
        fs = 1.0 / mean_interval
    else:
        fs = 1.0 / np.mean(valid_diffs)
    print(f"Calculated sampling frequency (Fs): {fs:.2f} Hz")

    # --- Preprocessing & Method Execution ---

    processed_signal = None 
    signal_for_plot1 = None 
    signal_for_plot2 = None 

    if method == 'GREEN':
        # Normalizing raw G trace
        G_norm_raw = normalize_trace(G_trace)
        # Filtering normalized G trace
        processed_signal = apply_bandpass_filter(G_norm_raw, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        signal_for_plot1 = G_norm_raw
        signal_for_plot2 = processed_signal # Signal after filtering

    elif method == 'G-R':
        # Normalizing raw G and R traces
        G_norm_raw = normalize_trace(G_trace)
        R_norm_raw = normalize_trace(R_trace)
        # Filtering normalized G and R traces
        G_filt_norm = apply_bandpass_filter(G_norm_raw, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        R_filt_norm = apply_bandpass_filter(R_norm_raw, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        # Combining filtered traces
        processed_signal = G_filt_norm - R_filt_norm
        signal_for_plot1 = G_norm_raw - R_norm_raw # Combine *before* filtering for plot 1
        signal_for_plot2 = processed_signal # Signal after combination

    elif method == 'CHROM':
        # --- CHROM Specific Parameters from Lab/Paper ---
        norm_win_sec = 1.0 # For Eq. 2 normalization
        chrom_win_sec = 1.0 # For Eq. 15/16 alpha calculation window

        # --- Normalization Method ---
        print(f"CHROM Step a: Applying normalization using {norm_win_sec}s sliding window average...")
        norm_win_samp = int(round(norm_win_sec * fs))
        if norm_win_samp < 1: norm_win_samp = 1

        # take the arithmetic average of each pixel with its neighbor.
        # size is the size of the sub-array to calculate arithmetic average.
        # The standard for pixels without enough neighbors is to reflect
        # mode : ‘nearest’ (a a a a | a b c d | d d d d)
        mean_R_sliding = uniform_filter1d(R_trace, size=norm_win_samp, mode='nearest')
        mean_G_sliding = uniform_filter1d(G_trace, size=norm_win_samp, mode='nearest')
        mean_B_sliding = uniform_filter1d(B_trace, size=norm_win_samp, mode='nearest')

        # Calculate normalized traces Cni = Ci / µ(Ci)
        Rn = R_trace / (mean_R_sliding + epsilon)
        Gn = G_trace / (mean_G_sliding + epsilon)
        Bn = B_trace / (mean_B_sliding + epsilon)
        print("CHROM Normalization complete.")

        Rf = apply_bandpass_filter(Rn, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        Gf = apply_bandpass_filter(Gn, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        Bf = apply_bandpass_filter(Bn, low_cutoff_hz, high_cutoff_hz, fs, filter_order)

        # Implement Equation 16/14 using a new sliding window ---
        # Calculate Xs and Ys signals (based on Eq. 9, using Eq.2 normalized traces)
        X = 3*Rn - 2*Gn
        Y = 1.5*Rn + Gn - 1.5*Bn

        # Filter X and Y to get Xf and Yf
        print("CHROM Filtering intermediate X and Y signals...")
        Xf = apply_bandpass_filter(X, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        Yf = apply_bandpass_filter(Y, low_cutoff_hz, high_cutoff_hz, fs, filter_order)
        print("CHROM Xf and Yf filtering complete.")

        # Calculate alpha using a sliding window on Xf, Yf (Eq. 15)
        print(f"CHROM Step b: Calculating alpha using {chrom_win_sec}s sliding window...")
        chrom_win_samp = int(round(chrom_win_sec * fs))
        if chrom_win_samp < 2: chrom_win_samp = 2
        half_win_chrom = chrom_win_samp // 2

        alpha_values = np.zeros(trace_size)

        # Pad Xf, Yf for calculating std dev in window
        Xf_padded = np.pad(Xf, half_win_chrom, mode='edge')
        Yf_padded = np.pad(Yf, half_win_chrom, mode='edge')

        for i in range(trace_size):
            Xf_win = Xf_padded[i : i + chrom_win_samp]
            Yf_win = Yf_padded[i : i + chrom_win_samp]
            std_Xf = np.std(Xf_win)
            std_Yf = np.std(Yf_win)
            alpha_values[i] = std_Xf / (std_Yf + epsilon)
        print("CHROM Alpha calculation complete.")

        # Calculate final pulse signal S (Eq. 14)
        # processed_signal = Xf - alpha_values * Yf

        # Calculate final pulse signal S using (Eq. 16)
        print("Calculating final signal S")
        processed_signal = 3 * (1 - alpha_values / 2) * Rf - 2 * (1 + alpha_values / 2) * Gf + (3 * alpha_values / 2) * Bf

        # Prepare signals for plotting
        signal_for_plot1 = Rn # Show one of the normalized signals as input example
        signal_for_plot2 = processed_signal # Show the signal before final normalization

    else:
        print(f"Error: Unknown method '{method}'. Choose 'GREEN', 'G-R', or 'CHROM'.")
        return

    if processed_signal is None:
        print("Error: Method processing failed to generate a signal.")
        return
    print(f"Method '{method}' processing complete.")


    # --- Final Step: Normalize the result from the chosen method ---
    print("Applying final normalization to the processed signal...")
    pulse_trace_final = normalize_trace(processed_signal)
    print("Final normalization complete.")


    # --- Visualization ---
    if verbose >= 1:
        print("Generating plots...")
        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        # Plot 1: Show the relevant initially normalized trace(s) / combination
        title1 = f'Initially Normalized Trace(s) ({method})'
        axs[0].plot(time_trace, signal_for_plot1, label=f'Norm Input ({method})')
        if method == 'GREEN': title1 = 'Initially Normalized Green Trace'
        elif method == 'G-R': title1 = 'Initially Normalized (G-R)'
        elif method == 'CHROM': title1 = 'Raw CHROM Combination (Using Norm Traces)' # Approx
        axs[0].set_title(title1)
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Show the signal after filtering and method-specific combination (before final norm)
        title2 = f'Processed Signal ({method}, Filtered, Before Final Norm)'
        axs[1].plot(time_trace, signal_for_plot2, 'b-', label=f'Filtered Processed ({method})')
        axs[1].set_title(title2)
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Show the final normalized pulse trace
        axs[2].plot(time_trace, pulse_trace_final, 'k-', label='Final Normalized rPPG Trace')
        axs[2].set_title(f'Final Normalized rPPG Signal (Method: {method})')
        axs[2].set_xlabel('Time (seconds)')
        axs[2].legend()
        axs[2].grid(True)

        # Overlay ground truth on the final plot
        if gt_trace.size > 0 and gt_time.size == gt_trace.size:
            try:
                interp_func = interp1d(gt_time, gt_trace, kind='linear', bounds_error=False, fill_value=np.nan)
                gt_trace_interp = interp_func(time_trace)
                valid_interp = ~np.isnan(gt_trace_interp)
                if np.any(valid_interp):
                     axs[2].plot(time_trace[valid_interp], gt_trace_interp[valid_interp], 'r--', alpha=0.6, label='Ground Truth PPG (Interp.)')
                     axs[2].legend()
            except Exception as e:
                print(f"Warning: Could not interpolate/plot ground truth trace: {e}")

        plt.tight_layout()
        plt.show()

    # --- Save Results ---
    if pulse_trace_final.size > 0:
        pulse_out_path = os.path.join(out_folder, pulse_out_file_name)
        try:
            savemat(pulse_out_path, {
                'pulseTrace': pulse_trace_final.astype(np.float64),
                'timeTrace': time_trace.astype(np.float64)
            })
            print(f"\nSaved final pulse trace ({method} method) to: {pulse_out_path}")
        except Exception as e:
            print(f"\nError saving {pulse_out_file_name}: {e}")
    else:
        print("\nWarning: No final pulse trace generated to save.")