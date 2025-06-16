# Yannick Benezeth - yannick.benezeth@u-bourgogne.fr
# M1 Vibot - Univ Bourgogne
# 2024 - 2025
# Enhanced with Histogram Backprojection using a Reference Image

import argparse
import os
import time
import cv2
import numpy as np
from scipy.io import savemat

def convolve_backprojection(B, r):
    """Applies elliptical convolution to the backprojection map."""
    try:
        D = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(r,r))
        # Use border replication to handle edges better during convolution
        B = cv2.filter2D(B, -1, D, borderType=cv2.BORDER_REPLICATE)
    except cv2.error as e:
        print(f"Warning: Convolution failed - {e}. Returning original map.")
    return B


def get_trace_from_vid_file(vid_folder, out_folder, verbose):
    print(f"Call get_trace_from_vid_file() [with Ref Image Hist Backprojection]:")
    print(f"  VIDFOLDER: {vid_folder}")
    print(f"  OUTFOLDER: {out_folder}")
    print(f"  VERBOSE: {verbose}")

    start_time_func = time.time()

    vid_file_name = 'vid-001.avi'
    out_file_name = 'rgbTraces.mat'

    vid_full_path = os.path.join(vid_folder, vid_file_name)
    out_full_path = os.path.join(out_folder, out_file_name)

    # Check if output file already exists - SKIP if it does
    if os.path.exists(out_full_path):
        print(f'rgbTraces file exists ({out_full_path}), skipping getTraceFromVidFile()...')
        return

    # --- Face Detection Setup ---
    haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(haar_cascade_path):
         print(f"Error: Haar Cascade file not found at {haar_cascade_path}")
         return
    face_detector = cv2.CascadeClassifier(haar_cascade_path)

    # --- Tracking Parameters ---
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # --- Histogram Backprojection Parameters ---
    hist_channels = [0, 1]  # Use Hue and Saturation
    hist_size = [60, 64]
    hist_ranges = [0, 180, 0, 256]
    convolution_radius = 5
    backprojection_threshold = 30 # Adjust this threshold based on results
    min_skin_pixel_count_threshold = 30
    print(f"Using Histogram Backprojection (HSV) with Reference Image.")

    # --- Define path to reference skin image ---
    reference_image_path = r"D:\VIBOT Master\Semester 2\Medical Image Analysis\Lectures\Y.Benezeth\Lab\video\image-Photoroom.png"

    # --- Calculate Reference Model Histogram ---
    model_histogram = None
    print(f"Attempting to load reference skin image: {reference_image_path}")
    ref_image = cv2.imread(reference_image_path)

    if ref_image is None:
        print(f"Error: Could not load reference image '{reference_image_path}'.")
        print("Histogram backprojection disabled. Falling back to full ROI average.")
    else:
        print("Calculating model histogram from reference image...")
        try:
            # --- Create Mask for Reference Image (using YCrCb) ---
            # Thresholds for identifying skin in the reference image
            ref_min_YCrCb = np.array([0, 133, 77], np.uint8)
            ref_max_YCrCb = np.array([255, 173, 127], np.uint8)
            ref_ycrcb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2YCR_CB)
            ref_skin_mask = cv2.inRange(ref_ycrcb, ref_min_YCrCb, ref_max_YCrCb)

            ref_skin_pixel_count = cv2.countNonZero(ref_skin_mask)
            if ref_skin_pixel_count < 100: # Check if enough skin pixels are found
                print(f"Warning: Found only {ref_skin_pixel_count} skin pixels in reference image using YCrCb mask. Histogram might be poor.")
                # Consider falling back or stopping if too few pixels? For now, proceed.

            # --- Calculate Histogram from Masked Reference ---
            ref_hsv = cv2.cvtColor(ref_image, cv2.COLOR_BGR2HSV)
            # Use the ref_skin_mask to calculate histogram only from skin areas
            model_histogram = cv2.calcHist([ref_hsv], hist_channels, mask=ref_skin_mask,
                                             histSize=hist_size, ranges=hist_ranges)

            if model_histogram is not None and np.sum(model_histogram) > 0:
                 cv2.normalize(model_histogram, model_histogram, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                 print("Model histogram calculated successfully from reference image.")
            else:
                 print("Error: Histogram calculated from reference is empty or invalid.")
                 model_histogram = None # Disable backprojection

        except cv2.error as e:
            print(f"Error processing reference image or calculating histogram: {e}")
            model_histogram = None # Disable backprojection

    # --- ROI Smoothing Parameter ---
    roi_smoothing_alpha = 0.4

    # --- Video Handling ---
    vid_obj = cv2.VideoCapture(vid_full_path)
    if not vid_obj.isOpened(): print(f"Error opening video: {vid_full_path}"); return
    fps = vid_obj.get(cv2.CAP_PROP_FPS); nb_frame_prop = vid_obj.get(cv2.CAP_PROP_FRAME_COUNT)
    nb_frame = int(nb_frame_prop) if nb_frame_prop > 0 else 20000
    print(f"Video Info: FPS={fps:.2f}, Est. Frames={nb_frame}")
    rgb_traces = np.zeros((4, nb_frame))

    # --- Tracking & ROI Variables ---
    old_gray = None; p0 = None; bbox = None; bbox_points = None
    smoothed_forehead_roi_coords = None

    # --- Processing Loop ---
    n = 0; start_time_loop = time.time(); processed_frame_count = 0

    while True:
        ret, frame = vid_obj.read()
        if not ret or frame is None: break

        frame_copy = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        try: gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error as e: continue

        # --- Face Localisation ---
        # Updates 'bbox' (raw face box)
        if p0 is None or len(p0) < 10: # Detection
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) > 0:
                x_f, y_f, w_f, h_f = faces[0]; bbox = (x_f, y_f, w_f, h_f)
                track_roi_x1=max(0, int(x_f+w_f*0.1)); track_roi_y1=max(0, int(y_f))
                track_roi_x2=min(frame_width, int(x_f+w_f*0.9)); track_roi_y2=min(frame_height, int(y_f+h_f*0.8))
                if track_roi_x1 < track_roi_x2 and track_roi_y1 < track_roi_y2:
                    roi_gray_track = gray[track_roi_y1:track_roi_y2, track_roi_x1:track_roi_x2]
                    p0 = cv2.goodFeaturesToTrack(roi_gray_track, mask=None, **feature_params)
                    if p0 is not None:
                        p0[:, 0, 0] += track_roi_x1; p0[:, 0, 1] += track_roi_y1
                        bbox_points = np.array([[x_f,y_f],[x_f+w_f,y_f],[x_f+w_f,y_f+h_f],[x_f,y_f+h_f]], dtype=float)
                    else: bbox = None; p0 = None; bbox_points = None
                else: bbox = None; p0 = None; bbox_points = None
            else: p0 = None; bbox = None; bbox_points = None
        else: # Tracking
            if old_gray is None: old_gray = gray.copy()
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray, p0, None, **lk_params)
            good_new = p1[st==1]; good_old = p0[st==1]
            if len(good_new) >= 10:
                xform, _ = cv2.estimateAffinePartial2D(good_old, good_new)
                if xform is not None:
                    bbox_points = cv2.transform(bbox_points.reshape(-1,1,2), xform).reshape(-1,2)
                    min_c=np.min(bbox_points, axis=0); max_c=np.max(bbox_points, axis=0)
                    bbox = (int(min_c[0]), int(min_c[1]), int(max_c[0]-min_c[0]), int(max_c[1]-min_c[1]))
                    p0 = good_new.reshape(-1,1,2)
                else: p0 = None; bbox = None; bbox_points = None
            else: p0 = None; bbox = None; bbox_points = None


        # --- Define Target Forehead ROI & Smooth Coordinates ---
        current_forehead_roi_coords = None
        if bbox is not None:
            x, y, w, h = bbox
            target_x1 = x + w * 0.25; target_y1 = y + h * 0.10
            target_x2 = x + w * 0.75; target_y2 = y + h * 0.40
            target_coords = np.array([target_x1, target_y1, target_x2, target_y2])
            if smoothed_forehead_roi_coords is None: smoothed_forehead_roi_coords = target_coords
            else: smoothed_forehead_roi_coords = (roi_smoothing_alpha*target_coords + (1-roi_smoothing_alpha)*smoothed_forehead_roi_coords)
            current_forehead_roi_coords = smoothed_forehead_roi_coords
        else: smoothed_forehead_roi_coords = None


        # --- Extract RGB using Backprojection with Reference Histogram ---
        avg = np.zeros(3) # Default to black
        skin_mask_display = None # For optional display

        if current_forehead_roi_coords is not None:
            x1 = max(0, int(current_forehead_roi_coords[0]))
            y1 = max(0, int(current_forehead_roi_coords[1]))
            x2 = min(frame_width, int(current_forehead_roi_coords[2]))
            y2 = min(frame_height, int(current_forehead_roi_coords[3]))

            if x1 < x2 and y1 < y2:
                img_roi = frame_copy[y1:y2, x1:x2]

                # --- Apply Backprojection (if model exists) ---
                if model_histogram is not None:
                    try:
                        roi_hsv = cv2.cvtColor(img_roi, cv2.COLOR_BGR2HSV)
                        backproj_map = cv2.calcBackProject([roi_hsv], hist_channels, model_histogram, hist_ranges, scale=1)
                        backproj_map = convolve_backprojection(backproj_map, convolution_radius)
                        _, skin_mask = cv2.threshold(backproj_map, backprojection_threshold, 255, cv2.THRESH_BINARY)
                        skin_mask_display = skin_mask # Store for potential display

                        num_skin_pixels = cv2.countNonZero(skin_mask)
                        if num_skin_pixels > min_skin_pixel_count_threshold:
                            avg_bgr = cv2.mean(img_roi, mask=skin_mask)[:3]
                            avg = np.array([avg_bgr[2], avg_bgr[1], avg_bgr[0]])
                        else: # Fallback to mean of the smoothed forehead ROI
                            avg_bgr = np.mean(img_roi, axis=(0, 1))
                            avg = np.array([avg_bgr[2], avg_bgr[1], avg_bgr[0]]) if not np.isnan(avg_bgr).any() else np.zeros(3)

                    except cv2.error as e:
                        print(f"Frame {n}: OpenCV error during backprojection/averaging: {e}. Using full ROI mean.")
                        avg_bgr = np.mean(img_roi, axis=(0, 1))
                        avg = np.array([avg_bgr[2], avg_bgr[1], avg_bgr[0]]) if not np.isnan(avg_bgr).any() else np.zeros(3)
                else:
                    # Model histogram not available, use full ROI average
                    avg_bgr = np.mean(img_roi, axis=(0, 1))
                    avg = np.array([avg_bgr[2], avg_bgr[1], avg_bgr[0]]) if not np.isnan(avg_bgr).any() else np.zeros(3)


        # --- Store RGB traces and timestamp ---
        if n < nb_frame:
            time_ms = vid_obj.get(cv2.CAP_PROP_POS_MSEC)
            rgb_traces[0:3, n] = avg
            rgb_traces[3, n] = time_ms
            processed_frame_count += 1
        else:
            print(f"Warning: Frame index {n} exceeds preallocated size {nb_frame}. Stopping.")
            break

        # --- Verbose output and visualization ---
        if verbose >= 1 and n % 50 == 0:
            
            current_time_loop = time.time(); elapsed = current_time_loop-start_time_loop
            fps_proc = processed_frame_count / elapsed if elapsed > 0 else 0
            print(f'\nProcessed frame {n+1}/{int(nb_frame_prop if nb_frame_prop > 0 else n+1)} ({fps_proc:.1f} fps_proc)')


            # Draw the smoothed forehead ROI
            if current_forehead_roi_coords is not None:
                x1, y1, x2, y2 = map(int, current_forehead_roi_coords)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display frame & optional mask
            cv2.imshow('Frame', frame)
            if verbose >= 2 and skin_mask_display is not None:
                cv2.imshow('Skin Mask (Ref Hist Backprojection)', skin_mask_display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): break

        elif verbose == 0 and n % 200 == 0: print('.', end='', flush=True)

        old_gray = gray.copy()
        n += 1

    # --- Cleanup and Save ---
    vid_obj.release()
    cv2.destroyAllWindows()

    # Trim unused pre-allocated space
    if processed_frame_count < nb_frame:
        print(f"\nTrimming rgb_traces array from {nb_frame} to {processed_frame_count} stored frames.")
        rgb_traces = rgb_traces[:, :processed_frame_count]

    if processed_frame_count > 0:
        try:
             savemat(out_full_path, {'rgbTraces': rgb_traces})
             print(f'\nSaved RGB traces (Ref Image Hist Backprojection) to: {out_full_path}')
        except Exception as e: print(f"\nError saving MAT file: {e}")
    else: print("\nWarning: No trace data collected, MAT file not saved.")

    end_time_func = time.time()
    print(f'get_trace_from_vid_file done in {round(end_time_func - start_time_func)} seconds')

