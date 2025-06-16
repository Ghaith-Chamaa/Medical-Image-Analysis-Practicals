# Yannick Benezeth - yannick.benezeth@u-bourgogne.fr
# M1 Vibot - Univ Bourgogne
# 2024 - 2025


import numpy as np
import os
import numpy as np
import scipy.signal
import warnings # Pour afficher des avertissements comme la fonction warning de MATLAB


def load_ppg(vid_folder):
    gt_trace = np.array([])
    gt_hr = np.array([])
    gt_time = np.array([])
    found_file = False

    # --- Try format 1: gtdump.xmp ---
    gt_filename_xmp = os.path.join(vid_folder, 'gtdump.xmp')
    if not found_file and os.path.exists(gt_filename_xmp):
        print(f"Found ground truth file: {gt_filename_xmp}")
        try:
            # Use numpy.loadtxt, assuming comma delimiter like csvread might expect
            # Skip header rows if necessary (e.g., skiprows=1) - adjust if needed
            gt_data = np.loadtxt(gt_filename_xmp, delimiter=',') # Try comma first

            if gt_data.ndim == 1: # Handle case where file might have only one row of data
                 gt_data = gt_data.reshape(1, -1)

            if gt_data.shape[1] >= 4: # Need at least 4 columns
                gt_time = gt_data[:, 0] / 1000.0 # Col 1 (index 0) is time in ms
                gt_hr = gt_data[:, 1] # Col 2 (index 1) is HR
                gt_trace = gt_data[:, 3] # Col 4 (index 3) is trace

                # Normalize trace (zero mean, unit variance)
                trace_mean = np.mean(gt_trace)
                trace_std = np.std(gt_trace)
                if trace_std > 1e-9: # Avoid division by zero
                    gt_trace = (gt_trace - trace_mean) / trace_std
                else:
                    gt_trace = gt_trace - trace_mean

                found_file = True
                print("  -> Loaded data from gtdump.xmp format.")
            else:
                print(f"  -> Warning: {gt_filename_xmp} has fewer than 4 columns.")

        except Exception as e:
            print(f"  -> Error reading {gt_filename_xmp}: {e}")
            # Reset arrays if reading failed partially
            gt_trace, gt_hr, gt_time = np.array([]), np.array([]), np.array([])


    # --- Try format 2: ground_truth.txt ---
    gt_filename_txt = os.path.join(vid_folder, 'ground_truth.txt')
    if not found_file and os.path.exists(gt_filename_txt):
        print(f"Found ground truth file: {gt_filename_txt}")
        try:
            # Use numpy.loadtxt, assuming space/tab delimiter like dlmread
            gt_data = np.loadtxt(gt_filename_txt)

            if gt_data.ndim == 1: # Handle case with only one column/row read
                 # This format expects multiple rows, so ndim=1 is likely an error or unexpected format
                 print(f"  -> Warning: Unexpected format in {gt_filename_txt} (expected multiple rows).")
            elif gt_data.shape[0] >= 3: # Need at least 3 rows
                gt_trace = gt_data[0, :] # Row 1 (index 0) is trace
                gt_hr = gt_data[1, :] # Row 2 (index 1) is HR
                gt_time_raw = gt_data[2, :] # Row 3 (index 2) is time

                # Make time relative to start
                if gt_time_raw.size > 0:
                    gt_time = gt_time_raw - gt_time_raw[0]
                else:
                    gt_time = np.array([])

                # Normalize trace (zero mean, unit variance)
                trace_mean = np.mean(gt_trace)
                trace_std = np.std(gt_trace)
                if trace_std > 1e-9: # Avoid division by zero
                    gt_trace = (gt_trace - trace_mean) / trace_std
                else:
                    gt_trace = gt_trace - trace_mean

                found_file = True
                print("  -> Loaded data from ground_truth.txt format.")
            else:
                 print(f"  -> Warning: {gt_filename_txt} has fewer than 3 rows.")

        except Exception as e:
            print(f"  -> Error reading {gt_filename_txt}: {e}")
            gt_trace, gt_hr, gt_time = np.array([]), np.array([]), np.array([])


    # --- Try format 3: Pulse Rate_BPM.txt / BP_MMHG.txt ---
    gt_filename_hr = os.path.join(vid_folder, 'Pulse Rate_BPM.txt')
    gt_filename_bp = os.path.join(vid_folder, 'BP_MMHG.txt')
    if not found_file and os.path.exists(gt_filename_hr):
        print(f"Found ground truth file: {gt_filename_hr}")
        try:
            # Load HR data
            gt_hr = np.loadtxt(gt_filename_hr)
            if gt_hr.ndim == 0: gt_hr = np.array([gt_hr]) # Handle single value file

            # Generate time vector (assuming indices are milliseconds as per MATLAB code)
            num_hr_samples = len(gt_hr)
            if num_hr_samples > 0:
                 # MATLAB code generates 0:(N-1) and divides by 1000.
                 # This implies the index corresponds to milliseconds.
                 gt_time = np.arange(num_hr_samples) / 1000.0
                 print(f"  -> Loaded HR data. Generated time vector assuming 1 KHz sampling (0 to {(num_hr_samples-1)/1000:.3f}s).")
            else:
                 gt_time = np.array([])
                 print("  -> Warning: HR file is empty.")


            # Optionally load trace data from BP_MMHG.txt
            if os.path.exists(gt_filename_bp):
                print(f"Found optional trace file: {gt_filename_bp}")
                try:
                    gt_trace = np.loadtxt(gt_filename_bp)
                    if gt_trace.ndim == 0: gt_trace = np.array([gt_trace]) # Handle single value file

                    # Normalize trace
                    trace_mean = np.mean(gt_trace)
                    trace_std = np.std(gt_trace)
                    if trace_std > 1e-9:
                        gt_trace = (gt_trace - trace_mean) / trace_std
                    else:
                        gt_trace = gt_trace - trace_mean
                    print("  -> Loaded and normalized trace data from BP_MMHG.txt.")

                    # Basic check for length consistency (optional)
                    if gt_trace.size != gt_hr.size:
                         print(f"  -> Warning: Trace length ({gt_trace.size}) differs from HR length ({gt_hr.size}).")
                         # Decide how to handle: truncate, error, etc.
                         # Truncating longer array for now:
                         min_len = min(gt_trace.size, gt_hr.size)
                         gt_trace = gt_trace[:min_len]
                         gt_hr = gt_hr[:min_len]
                         gt_time = gt_time[:min_len]
                         print(f"     -> Truncated both to length {min_len}.")


                except Exception as e_bp:
                    print(f"  -> Error reading {gt_filename_bp}: {e_bp}")
                    gt_trace = np.array([]) # Reset trace if BP file reading fails
            else:
                 print("  -> Optional trace file BP_MMHG.txt not found.")
                 gt_trace = np.array([]) # Ensure trace is empty if file not found


            found_file = True # Mark as found even if only HR was loaded

        except Exception as e_hr:
            print(f"  -> Error reading {gt_filename_hr}: {e_hr}")
            gt_trace, gt_hr, gt_time = np.array([]), np.array([]), np.array([])


    # --- Final Check ---
    if not found_file:
        print('Warning: No valid PPG ground truth file found.')

    # Ensure all returned arrays are 1D
    #gt_trace = gt_trace.flatten()
    #gt_hr = gt_hr.flatten()
    #gt_time = gt_time.flatten()

    print(f"load_ppg finished. Trace size: {gt_trace.size}")
    return gt_trace, gt_hr, gt_time

