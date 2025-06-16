# START OF FILE get_hr_from_pulse.py ---

# Yannick Benezeth - yannick.benezeth@u-bourgogne.fr
# M1 Vibot - Univ Bourgogne
# 2024 - 2025

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt
import os
import time
from scipy.fft import fft, fftfreq

from utils import load_ppg

def estimate_hr_from_fft(signal_window, fs, low_hz, high_hz, n_fft, min_prominence=0.5, verbose_level=0, window_info=""):
    n_samples = len(signal_window)
    epsilon = 1e-9

    # Basic checks
    if n_samples < 20:
        if verbose_level >= 3: print(f"  [FFT Debug {window_info}] NaN: Window too short ({n_samples} samples)")
        return np.nan
    if fs <= epsilon or np.isnan(fs):
        if verbose_level >= 3: print(f"  [FFT Debug {window_info}] NaN: Invalid fs ({fs})")
        return np.nan

    # Ensure n_fft is at least n_samples
    if n_fft < n_samples:
        n_fft = n_samples # No padding if n_fft is too small

    # Apply a window function
    windowed_signal = signal_window * np.hanning(n_samples)

    # Compute Zero-Padded FFT
    # Use n=n_fft to specify the FFT length
    yf = fft(windowed_signal, n=n_fft)
    xf = fftfreq(n_fft, d=1.0/fs) # Use n_fft here as well

    # Get positive frequency magnitude spectrum
    positive_freq_mask = xf >= 0
    xf_pos = xf[positive_freq_mask]
    yf_mag = np.abs(yf[positive_freq_mask])
    if np.max(yf_mag) > epsilon:
         yf_mag_norm = yf_mag / np.max(yf_mag)
    else:
         yf_mag_norm = yf_mag

    # Find peaks within the valid HR frequency range
    valid_hr_mask = (xf_pos >= low_hz) & (xf_pos <= high_hz)
    if not np.any(valid_hr_mask):
        if verbose_level >= 3: print(f"  [FFT Debug {window_info}] NaN: No FFT bins in valid HR range ({low_hz}-{high_hz} Hz)")
        return np.nan

    freqs_in_range = xf_pos[valid_hr_mask]
    mags_in_range_norm = yf_mag_norm[valid_hr_mask]

    if len(mags_in_range_norm) == 0:
         if verbose_level >= 3: print(f"  [FFT Debug {window_info}] NaN: Mags in range array is empty")
         return np.nan

    # Use find_peaks
    peaks, properties = scipy.signal.find_peaks(mags_in_range_norm, prominence=min_prominence)

    if len(peaks) == 0:
        if verbose_level >= 3: print(f"  [FFT Debug {window_info}] NaN: No peak found with prominence >= {min_prominence}")
        return np.nan

    # Choose highest magnitude peak among prominent ones
    highest_peak_idx_in_peaks = np.argmax(mags_in_range_norm[peaks])
    chosen_peak_index_in_range = peaks[highest_peak_idx_in_peaks]
    peak_freq_hz = freqs_in_range[chosen_peak_index_in_range]

    # Convert peak frequency to BPM
    hr_bpm = peak_freq_hz * 60.0

    if verbose_level >= 3: print(f"  [FFT Debug {window_info}] OK: Found peak freq {peak_freq_hz:.2f} Hz ({hr_bpm:.1f} BPM)")

    return hr_bpm


def get_hr_from_pulse(vid_folder, out_folder, win_length_sec, verbose):
    print(f"Call get_hr_from_pulse (Task 3 - Zero-Padded FFT):")
    print(f"  VIDFOLDER: {vid_folder}")
    print(f"  OUTFOLDER: {out_folder}")
    print(f"  WINLENGTHSEC: {win_length_sec}")
    print(f"  VERBOSE: {verbose}")
    print("-" * 20)

    start_time_proc = time.time()

    # --- Parameters ---
    step_sec = 1.0
    mat_file_name = 'pulseTrace.mat'
    pulse_file = os.path.join(out_folder, f"{mat_file_name}")
    low_hr_hz = 0.7
    high_hr_hz = 3.5
    peak_prominence_threshold = 0.3
    epsilon = 1e-9

    # --- FFT Parameters ---
    # Define the desired FFT length (for zero-padding)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
    # Should be >= window length in samples. Larger values give finer freq resolution.
    N_FFT = 8192 # Example: Use 8192 points for FFT calculation
    # 8192 gives much finer frequency resolution (Δf).
    # 8192 is a power of 2 (computationally efficient for FFTs) and is significantly larger than the typical window length
    # It uses zero-padding to artificially increase the FFT length.
    # This dramatically increases the number of frequency bins calculated by the FFT.
    print(f"Using FFT length (N_FFT): {N_FFT}")

    # --- Load rPPG Data ---
    if not os.path.isfile(pulse_file):
        print(f"Error: Input file not found at '{pulse_file}'")
        return [], []
    try:
        mat_data = scipy.io.loadmat(pulse_file)
        pulse_trace = mat_data['pulseTrace'].squeeze()
        time_trace = mat_data['timeTrace'].squeeze()
        print(f"Loaded rPPG trace (length {len(pulse_trace)}) and time (length {len(time_trace)})")
    except Exception as e:
        print(f"Error loading {pulse_file}: {e}")
        return [], []


    print("Processing...")

    trace_size = len(time_trace)
    if trace_size <= 1:
        print("Error: timeTrace from pulseTrace.mat is too short.")
        return [], []

    # --- Load PPG ground truth ---
    gt_trace, gt_hr, gt_time = load_ppg(vid_folder)
    gt_available = gt_trace.size > 0 and gt_time.size > 1 and gt_hr.size > 0
    if not gt_available:
        print("Warning: Ground truth PPG data is missing or incomplete. Cannot estimate PPG HR or get sensor HR.")


    # --- Calculate rPPG Sampling Frequency ---
    fs_rppg = np.nan
    if trace_size > 1:
        time_diffs_rppg = np.diff(time_trace)
        valid_diffs_rppg = time_diffs_rppg[time_diffs_rppg > epsilon]
        if len(valid_diffs_rppg) > 0:
            mean_interval_rppg = np.mean(valid_diffs_rppg)
            if mean_interval_rppg > epsilon: fs_rppg = 1.0 / mean_interval_rppg
            else: print("Warning: Mean rPPG time interval is near zero.")
        else: print("Warning: No valid time differences found for rPPG trace.")
    if np.isnan(fs_rppg):
        print("Error: Could not determine rPPG sampling frequency. Aborting.")
        return [],[]
    print(f"Detected rPPG Fs: {fs_rppg:.2f} Hz")


    # --- Sliding Window Setup ---
    win_length = int(round(win_length_sec * fs_rppg))
    step = int(round(step_sec * fs_rppg))
    if win_length < 20 or step < 1:
        print(f"Error: Window length ({win_length}) or step ({step}) is too small.")
        return [], []
    half_win = win_length // 2

    # --- Sliding window loop ---
    window_indices = range(half_win, trace_size - half_win, step)
    time_list = []
    hr_ppg_estimated = []
    hr_rppg_estimated = []
    hr_sensor_ref = []

    print(f"Starting sliding window: {len(window_indices)} windows total, WinLenSamples={win_length}, StepSamples={step}")

    for idx, i in enumerate(window_indices):
        start_idx_rppg = max(0, i - half_win)
        end_idx_rppg = min(trace_size, i + half_win)
        current_win_len = end_idx_rppg - start_idx_rppg
        if current_win_len < win_length:
             if start_idx_rppg == 0: end_idx_rppg = min(trace_size, start_idx_rppg + win_length)
             elif end_idx_rppg == trace_size: start_idx_rppg = max(0, end_idx_rppg - win_length)
        current_win_len = end_idx_rppg - start_idx_rppg
        if start_idx_rppg >= end_idx_rppg or current_win_len < 10:
            time_list.append(np.nan)
            hr_rppg_estimated.append(np.nan)
            hr_ppg_estimated.append(np.nan)
            hr_sensor_ref.append(np.nan)
            if verbose >= 3: print(f"  Skipping invalid window index {idx}")
            continue

        center_idx_rppg = (start_idx_rppg + end_idx_rppg) // 2
        crt_time = time_trace[center_idx_rppg]
        time_list.append(crt_time)
        start_time = time_trace[start_idx_rppg]
        end_time = time_trace[end_idx_rppg-1]

        # --- Estimate HR from rPPG ---
        crt_pulse_win_rppg = pulse_trace[start_idx_rppg:end_idx_rppg]
        
        hr_rppg_current = estimate_hr_from_fft(
            crt_pulse_win_rppg, fs_rppg, low_hr_hz, high_hr_hz, n_fft=N_FFT,
            min_prominence=peak_prominence_threshold,
            verbose_level=verbose, window_info=f"rPPG @ {crt_time:.1f}s"
        )
        hr_rppg_estimated.append(hr_rppg_current)

        # --- Estimate HR from PPG ---
        hr_ppg_current = np.nan
        fs_ppg = np.nan
        if gt_available:
            start_idx_gt = np.argmin(np.abs(gt_time - start_time))
            end_idx_gt = np.argmin(np.abs(gt_time - end_time))
            if start_idx_gt < end_idx_gt and end_idx_gt < len(gt_trace):
                crt_ppg_win = gt_trace[start_idx_gt : end_idx_gt + 1]
                crt_time_ppg_win = gt_time[start_idx_gt : end_idx_gt + 1]
                n_ppg_samples = len(crt_time_ppg_win)
                if n_ppg_samples > 1:
                    time_diffs_ppg = np.diff(crt_time_ppg_win)
                    valid_diffs_ppg = time_diffs_ppg[time_diffs_ppg > epsilon]
                    if len(valid_diffs_ppg) > 0:
                        mean_interval_ppg = np.mean(valid_diffs_ppg)
                        if mean_interval_ppg > epsilon: fs_ppg = 1.0 / mean_interval_ppg
                if not np.isnan(fs_ppg):
                     hr_ppg_current = estimate_hr_from_fft(
                         crt_ppg_win, fs_ppg, low_hr_hz, high_hr_hz, n_fft=N_FFT,
                         min_prominence=peak_prominence_threshold,
                         verbose_level=verbose, window_info=f"PPG @ {crt_time:.1f}s"
                     )
        hr_ppg_estimated.append(hr_ppg_current)

        # --- Get Sensor HR Reference ---
        hr_sensor_current = np.nan
        if gt_available:
             try:
                 sensor_time_idx = np.argmin(np.abs(gt_time - crt_time))
                 hr_sensor_current = gt_hr[sensor_time_idx]
             except IndexError: hr_sensor_current = np.nan
        hr_sensor_ref.append(hr_sensor_current)


        if verbose >= 2 :
             print(f"  Win {idx} @{crt_time:.2f}s: rPPG HR={hr_rppg_current:.1f}, PPG HR={hr_ppg_current:.1f}, Sensor HR={hr_sensor_current:.1f}")


    print("\nProcessing finished.")
    end_time_proc = time.time()
    print(f"Processing done in {round(end_time_proc - start_time_proc)} seconds.")

    # Convert lists to numpy arrays
    hr_ppg_estimated = np.array(hr_ppg_estimated)
    hr_rppg_estimated = np.array(hr_rppg_estimated)
    hr_sensor_ref = np.array(hr_sensor_ref)
    time_list = np.array(time_list)

    # Filter out NaN times for analysis and plotting
    valid_time_mask = ~np.isnan(time_list)
    time_plot = time_list[valid_time_mask]
    hr_ppg_plot = hr_ppg_estimated[valid_time_mask]
    hr_rppg_plot = hr_rppg_estimated[valid_time_mask]
    hr_sensor_plot = hr_sensor_ref[valid_time_mask]

    if not np.all(valid_time_mask):
         print(f"Warning: {np.sum(~valid_time_mask)} windows were skipped due to index issues.")

    # Calculate Error and MAE
    valid_error_mask = ~np.isnan(hr_rppg_plot) & ~np.isnan(hr_ppg_plot)
    time_for_error = time_plot[valid_error_mask]
    error_rppg_vs_ppg = hr_rppg_plot[valid_error_mask] - hr_ppg_plot[valid_error_mask]
    if len(error_rppg_vs_ppg) > 0:
        mae_rppg_vs_ppg = np.mean(np.abs(error_rppg_vs_ppg))
        print(f"MAE (rPPG Estimate vs PPG Estimate): {mae_rppg_vs_ppg:.2f} bpm")
    else:
        mae_rppg_vs_ppg = np.nan
        print("MAE (rPPG Estimate vs PPG Estimate): N/A (no overlapping valid estimates)")
    valid_sensor_mask = ~np.isnan(hr_sensor_plot)
    valid_comp_mask = valid_sensor_mask & ~np.isnan(hr_rppg_plot)
    if np.any(valid_comp_mask):
         mae_rppg_vs_sensor = np.mean(np.abs(hr_rppg_plot[valid_comp_mask] - hr_sensor_plot[valid_comp_mask]))
         print(f"MAE (rPPG Estimate vs Sensor Ref):   {mae_rppg_vs_sensor:.2f} bpm")
    else:
        print("MAE (rPPG Estimate vs Sensor Ref):   N/A (no overlapping valid estimates)")

    # Plotting
    if verbose >= 1:
        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        ax_hr = axs[0]
        ax_hr.grid(True)
        valid_sensor = ~np.isnan(hr_sensor_plot)
        if np.any(valid_sensor): ax_hr.plot(time_plot[valid_sensor], hr_sensor_plot[valid_sensor], 'b-*', markersize=5, linewidth=1.5, label='HR Sensor (Reference)')
        valid_ppg = ~np.isnan(hr_ppg_plot)
        if np.any(valid_ppg): ax_hr.plot(time_plot[valid_ppg], hr_ppg_plot[valid_ppg], 'g-^', markersize=5, linewidth=1.5, label='HR PPG (FFT Estimate)')
        valid_rppg = ~np.isnan(hr_rppg_plot)
        if np.any(valid_rppg): ax_hr.plot(time_plot[valid_rppg], hr_rppg_plot[valid_rppg], 'c-o', markersize=5, alpha=0.8, linewidth=1.5, label='HR rPPG (FFT Estimate)')
        ax_hr.set_title(f'Heart Rate Estimation (Window: {win_length_sec}s, Step: {step_sec}s, N_FFT: {N_FFT})') # Added N_FFT to title
        ax_hr.set_ylabel('Heart Rate (bpm)')
        all_valid_hr = np.concatenate(
            [arr[~np.isnan(arr)] for arr in [hr_sensor_plot, hr_ppg_plot, hr_rppg_plot] if np.any(~np.isnan(arr))]
        ) if np.any(~np.isnan(hr_sensor_plot)) or np.any(~np.isnan(hr_ppg_plot)) or np.any(~np.isnan(hr_rppg_plot)) else np.array([60, 100])
        y_min_hr = max(30, np.min(all_valid_hr) - 15)
        y_max_hr = min(220, np.max(all_valid_hr) + 15)
        ax_hr.set_ylim([y_min_hr, y_max_hr])
        ax_hr.legend()

        ax_err = axs[1]
        ax_err.grid(True)
        if len(time_for_error) > 0:
            ax_err.plot(time_for_error, error_rppg_vs_ppg, 'r.-', markersize=6, linewidth=1.0, label='Error (rPPG Est - PPG Est)')
            ax_err.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax_err.legend(title=f'MAE = {mae_rppg_vs_ppg:.2f} bpm' if not np.isnan(mae_rppg_vs_ppg) else 'MAE = N/A')
            max_abs_err = np.max(np.abs(error_rppg_vs_ppg)) if len(error_rppg_vs_ppg) > 0 else 10
            err_limit = max(5, np.ceil(max_abs_err * 1.2))
            ax_err.set_ylim([-err_limit, err_limit])
        else:
            ax_err.text(0.5, 0.5, 'No overlapping valid estimates\nto calculate error', horizontalalignment='center', verticalalignment='center', transform=ax_err.transAxes)
            ax_err.legend()
        ax_err.set_xlabel('Time (seconds)')
        ax_err.set_ylabel('HR Error (bpm)')

        plt.tight_layout(h_pad=2.0)
        plt.show()

    return list(hr_ppg_estimated), list(hr_rppg_estimated)
