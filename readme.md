# Medical Image Analysis - Course Practicals

This repository contains the completed practicals and lab assignments for the Medical Image Analysis course. It covers topics from fundamental image color theory and segmentation to advanced signal processing for physiological measurement from video.

## 📋 Table of Contents

- [Project Goal](#-project-goal)
- [Key Topics Covered](#-key-topics-covered)
- [Setup and Installation](#️-setup-and-installation)
- [Usage Guide](#-usage-guide)
- [Practicals Overview](#-practicals-overview)
  - [Practical 1: Color, Spectral Imaging, and Segmentation](#practical-1-color-spectral-imaging-and-segmentation)
  - [Practical 2: Remote Photoplethysmography (rPPG) from Video](#practical-2-remote-photoplethysmography-rppg-from-video)
- [Acknowledgements](#-acknowledgements)

## 🎯 Project Goal

The goal of this repository is to document and showcase the hands-on work completed for the Medical Image Analysis course. Each practical implements a core concept, demonstrating a progression from basic image manipulation to the development of a complete pipeline for a real-world biomedical application.

## 💡 Key Topics Covered

- **Image Processing:** Color space transformations (RGB, HSV, CIEL*a*b*), filtering, and histogram analysis.
- **Image Segmentation:** Color-based thresholding and object isolation.
- **Machine Learning:** Unsupervised learning (K-Means) for color quantization and image compression.
- **Signal Processing:** Digital filtering (band-pass), signal normalization, and frequency analysis using the Fast Fourier Transform (FFT).
- **Computer Vision:** Face detection (Haar Cascades), feature tracking (Kanade-Lucas-Tomasi optical flow), and skin pixel detection (histogram backprojection).
- **Physiological Computing:** Implementation of a complete remote photoplethysmography (rPPG) pipeline to estimate heart rate from video.

## ⚙️ Setup and Installation

Follow these steps to set up the project environment.

1.  **Prerequisites:**
    *   Python 3.8+

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Ghaith-Chamaa/Medical-Image-Analysis-Practicals.git
    cd Medical-Image-Analysis-Practicals
    ```

## 🚀 Usage Guide

### Running Practical 1 (Jupyter Notebook)

1.  Launch Jupyter Notebook or JupyterLab.
    ```bash
    jupyter notebook
    ```
2.  Open the `spectral_mia.ipynb` file and run the cells sequentially.

### Running Practical 2 (Python Scripts)

The main entry point for the second practical is `main.py`. It can be executed from the command line within the directory.

1.  Run the script using python. You can specify arguments to change its behavior.

    **Example command:**
    ```bash
    python main.py --VIDFOLDER ../data/video/ --OUTFOLDER ../data/out/ --CHANNELCOMBMETHOD CHROM --VERBOSE 1
    ```

    **Key Arguments:**
    *   `--VIDFOLDER`: Path to the folder containing the video and ground truth files.
    *   `--OUTFOLDER`: Path to the folder where intermediate `.mat` files will be saved.
    *   `--CHANNELCOMBMETHOD`: The rPPG extraction method. Options: `GREEN`, `G-R`, `CHROM`.
    *   `--WINLENGTHSEC`: The length of the sliding window in seconds for HR analysis.
    *   `--VERBOSE`: Controls console output and plot visibility.

## 🔬 Practicals Overview

### Practical 1: Color, Spectral Imaging, and Segmentation

*   **Associated Files:** `spectral_mia.ipynb`

This practical explores the fundamentals of digital color and its application in image analysis. The notebook is divided into three parts: segmentation, compression, and spectral synthesis.

First, an image is loaded and converted to the HSV and CIEL*a*b* color spaces. The HSV space proves highly effective for **color-based segmentation**, where simple thresholds on the Hue channel are used to create a binary mask of a target object. This mask enables creative applications like swapping the object's color while preserving its texture and lighting.

Next, the lab demonstrates **image compression via color quantization**. The K-Means clustering algorithm is used to find a reduced palette of representative colors. Each pixel in the image is then mapped to the nearest color in the palette, effectively reducing the image's data size.

Finally, the practical introduces **spectral imaging**. A "spectral cube" is loaded from a series of images taken at different wavelengths. This data is then used to synthesize two different views of the same scene: one as perceived by a standardized human eye (CIE Standard Observer) and another as captured by a specific camera sensor (Nikon D50). This process highlights the differences between human and machine vision.

### Practical 2: Remote Photoplethysmography (rPPG) from Video

*   **Associated Files:** `main.py`, `get_trace_from_vid_file.py`, `get_pulse_signal_from_traces.py`, `get_hr_from_pulse.py`, `utils.py`

This practical implements a complete, end-to-end system for estimating heart rate from a simple facial video pointed by `Medical-Image-Analysis-Practicals-PPG-Lab.gdrive` file. The pipeline is broken into three main scripts orchestrated by `main.py`.

1.  **ROI Tracking & Trace Extraction:** The process begins in `get_trace_from_vid_file.py`, which analyzes the input video frame by frame. It uses a Haar Cascade for initial face detection and the robust Kanade-Lucas-Tomasi (KLT) algorithm for continuous optical flow tracking. To isolate the physiological signal, a specific Region of Interest (ROI) on the forehead is targeted. A **skin detection** method based on histogram backprojection ensures that only skin pixels within the ROI are used, improving signal-to-noise ratio. The average RGB values from this ROI are recorded over time, producing the raw `rgbTraces.mat` file.

2.  **Pulse Signal Extraction:** The raw RGB traces are noisy. `get_pulse_signal_from_traces.py` processes them to extract a clean rPPG signal. It implements three different methods of increasing complexity: **GREEN** (using only the green channel), **G-R** (using the difference of green and red), and **CHROM**, a state-of-the-art method robust to motion artifacts. These methods apply normalization and band-pass filtering to isolate the pulse wave, saving the result in `pulseTrace.mat`.

3.  **Heart Rate Estimation:** Finally, `get_hr_from_pulse.py` takes the clean pulse signal and calculates the heart rate. It uses a **sliding window approach**, analyzing short, overlapping segments of the signal. Within each window, a Fast Fourier Transform (FFT) with zero-padding is applied to obtain a high-resolution frequency spectrum. The dominant peak in this spectrum corresponds to the heart rate in Hz, which is converted to Beats Per Minute (BPM). The script compares its own estimation against a ground-truth signal and calculates the Mean Absolute Error (MAE) to evaluate performance.

## 🙏 Acknowledgements

A special thanks to the instructor, **Franck Marzani** & **Antony Madaleno** for practical 1 & **Yannick Benezeth** for practical 2, for providing the foundational code and guidance for these practicals.