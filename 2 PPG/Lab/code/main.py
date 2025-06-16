# Yannick Benezeth - yannick.benezeth@u-bourgogne.fr
# M1 Vibot - Univ Bourgogne
# 2024 - 2025

import argparse
import os

from get_trace_from_vid_file import get_trace_from_vid_file
from get_pulse_signal_from_traces import get_pulse_signal_from_traces
from get_hr_from_pulse import get_hr_from_pulse


def launcher(vid_folder_arg=None, out_folder_arg=None, win_length_sec_arg=None, channel_comb_method_arg=None, verbose_arg=None):
    # Default values
    VIDFOLDER = vid_folder_arg if vid_folder_arg is not None else r'D:\VIBOT Master\Semester 2\Medical Image Analysis\Lectures\Y.Benezeth\Lab\video'
    OUTFOLDER = out_folder_arg if out_folder_arg is not None else r'D:\VIBOT Master\Semester 2\Medical Image Analysis\Lectures\Y.Benezeth\Lab\out'
    WINLENGTHSEC = win_length_sec_arg if win_length_sec_arg is not None else 15
    CHANNELMETHOD = channel_comb_method_arg if channel_comb_method_arg is not None else 'GREEN'
    VERBOSE = verbose_arg if verbose_arg is not None else 1

    print("--- Configuration ---")
    print(f"VIDFOLDER: {VIDFOLDER}")
    print(f"OUTFOLDER: {OUTFOLDER}")
    print(f"WINLENGTHSEC: {WINLENGTHSEC}")
    print(f"VERBOSE: {VERBOSE}")

    # %%%%%%
    # Get RGB traces from a video file
    get_trace_from_vid_file(vid_folder=VIDFOLDER,
                            out_folder=OUTFOLDER,
                            verbose=VERBOSE) 

    # %%%%%%
    # Get rPPG signal from RGB traces
    get_pulse_signal_from_traces(vid_folder=VIDFOLDER,
                                out_folder=OUTFOLDER,
                                method=CHANNELMETHOD,
                                verbose=VERBOSE)

    # %%%%%%
    # Get Heart Rate from rPPG signal
    hr_ppg, hr_rppg = get_hr_from_pulse(vid_folder=VIDFOLDER,
                              out_folder=OUTFOLDER,
                               win_length_sec=WINLENGTHSEC,
                               verbose=VERBOSE) 

    # Calculate accuracy, correlation plots, Bland Altman and show results
    print(f"\n--- Results for the video: {VIDFOLDER} ---")
    

# --- Command-line Argument Handling ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python script for heart rate estimation from videos.')

    # Define expected arguments
    parser.add_argument('--VIDFOLDER', type=str, help='Folder containing the video or data.')
    parser.add_argument('--OUTFOLDER', type=str, help='Folder to save intermediate/final results.')
    parser.add_argument('--WINLENGTHSEC', type=int, help='Analysis window length in seconds.')
    parser.add_argument('--CHANNELCOMBMETHOD', type=str, help='Used Channel Combination Method for Pulse Extraction.')
    parser.add_argument('--VERBOSE', type=int, default=1, help='Verbosity level (0: silent, 1: normal, 2: detailed).')

    # Parse the provided arguments
    args = parser.parse_args()

    # Call the main function with provided arguments (or defaults if not provided)
    launcher(vid_folder_arg=args.VIDFOLDER,
                out_folder_arg=args.OUTFOLDER,
                win_length_sec_arg=args.WINLENGTHSEC,
                channel_comb_method_arg=args.CHANNELCOMBMETHOD,
                verbose_arg=args.VERBOSE)
