import pandas as pd
import mne
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import math
import json


def load_and_preprocess_eeg(path_eeg, filename, marker_file, SFREQ, L_FREQ, H_FREQ):
    df = pd.read_csv(path_eeg / filename)
    df_markers = pd.read_csv(path_eeg / marker_file)

    eeg_data = df.drop(columns=["TimeStamp"]).values.T * 1e-6  # uV -> V
    ch_names = [col for col in df.columns if col != "TimeStamp"]
    sfreq = SFREQ

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(eeg_data, info)

    try:
        raw.set_montage("standard_1020")
    except ValueError:
        print("! Note: Problems with channel installation.")

    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose=False)

    return raw, df, df_markers

def extract_segments(raw, df_time, df_markers, segments_list):
    """
    Returns a list of Raw objects (snippets) and their names.
    """
    extracted_raws = []
    valid_labels = []

    rec_start_time = df_time["TimeStamp"].iloc[0]

    for start_code, end_code, label in segments_list:
        if (
            start_code not in df_markers["Code"].values
            or end_code not in df_markers["Code"].values
        ):
            print(f"! Omitted: {label} (no markers) {start_code}/{end_code})")
            continue

        t_start_abs = df_markers.loc[
            df_markers["Code"] == start_code, "TimeStamp"
        ].values[0]
        t_end_abs = df_markers.loc[df_markers["Code"] == end_code, "TimeStamp"].values[
            0
        ]

        t_min = max(0, t_start_abs - rec_start_time)
        t_max = min(raw.times[-1], t_end_abs - rec_start_time)

        crop = raw.copy().crop(tmin=t_min, tmax=t_max)
        extracted_raws.append(crop)
        valid_labels.append(label)

    return extracted_raws, valid_labels

def plot_psd_comparison(raw_list, labels):
    """
    Draws PSD graphs in a grid layout.
    """
    n = len(raw_list)
    if n == 0:
        return

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows), sharey=True)

    if n > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    print(f">>>  Drawing PSD for {n} segments (Layout {rows}x{cols}).")

    for i, (raw_seg, label) in enumerate(zip(raw_list, labels)):
        ax = axes_flat[i]

        spectrum = raw_seg.compute_psd(fmin=1, fmax=45)
        spectrum.plot(axes=ax, picks="eeg", show=False, spatial_colors=True)

        ax.set_title(f"PSD: {label}")
        ax.grid(True)

    for i in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    plt.tight_layout()
    plt.show()

def plot_topomap_comparison(
    raw_list,
    labels,
    band_name="Alpha",
    bands={
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Alpha": (8, 12),
        "Beta": (13, 30),
        "Gamma": (30, 45),
    },
):
    """
    Draws Topomaps in a grid layout.
    """
    n = len(raw_list)
    if n == 0:
        return

    if band_name not in bands:
        print(f"Error: Unknown band '{band_name}'.")
        return

    fmin, fmax = bands[band_name]

    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))

    if n > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    print(f">>> Topomap {band_name} ({n} seg)...")

    all_powers = []
    spectra_data = []
    for raw_seg in raw_list:
        spect = raw_seg.compute_psd(fmin=fmin, fmax=fmax)
        data = spect.get_data().mean(axis=1)
        all_powers.extend(data)
        spectra_data.append(data)

    vmax = np.max(all_powers)
    vmin = np.min(all_powers)

    for i, (data, raw_seg, label) in enumerate(zip(spectra_data, raw_list, labels)):
        ax = axes_flat[i]
        im, _ = mne.viz.plot_topomap(
            data,
            raw_seg.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(vmin, vmax),
            names=raw_seg.ch_names,
        )
        ax.set_title(f"{label}\n{band_name}")

    for i in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.colorbar(im, ax=axes_flat[:n], fraction=0.05)

    plt.show()

def plot_band_power_over_time(raw_list, labels, band=(8, 12), band_name="Alpha"):
    plt.figure(figsize=(12, 6))

    for raw_seg, label in zip(raw_list, labels):
        raw_band = raw_seg.copy().filter(band[0], band[1], verbose=False)

        raw_band.apply_hilbert(envelope=True)

        data = raw_band.get_data().mean(axis=0) * 1e6
        times = raw_band.times

        window = int(raw_seg.info["sfreq"])
        data_smooth = np.convolve(data, np.ones(window) / window, mode="valid")
        times_smooth = times[: len(data_smooth)]

        plt.plot(times_smooth, data_smooth, label=label)

    plt.title(f"The evolution of {band_name} over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_asymmetry(raw_list, labels, ch_left="F3", ch_right="F4"):
    results = []

    for raw_seg in raw_list:
        psd = raw_seg.compute_psd(fmin=8, fmax=12, picks=[ch_left, ch_right])
        data = psd.get_data()

        alpha_left = data[0].mean()
        alpha_right = data[1].mean()

        faa = np.log(alpha_right) - np.log(alpha_left)
        results.append(faa)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        labels, results, color=["green" if x > 0 else "red" for x in results]
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title(
        f"Alpha asymmetry ({ch_right} vs {ch_left})\n>0: Positive/Approach | <0: Negative/Withdrawal"
    )
    plt.ylabel("Index FAA")
    plt.show()

def get_user_metadata(file_name, json_path):
    """
    Retrieves user ID and Name from the user_map.json file.
    """
    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return None

    with open(json_path, "r") as f:
        user_map = json.load(f).get("user_map", {})

    file_key = file_name if file_name in user_map else file_name.lower()
    return user_map.get(file_key)

def extract_scroll_events(eeg_filename, interactions_csv, users_csv, json_path):
    """
    Correlates behavioral scroll events with the EEG recording timeline.
    Filters specifically for 'brainrot_scroll' and 'smart_scroll' string IDs.
    """
    
    import os
    
    print(os.listdir(json_path.parent))
    
    TARGET_STAGES = ["brainrot_scroll", "smart_scroll"]

    user_info = get_user_metadata(eeg_filename, json_path)
    if not user_info:
        print("User not found in user_map.json.")
        return None

    participant_id = user_info["id"]
    user_name = user_info["name"]
    print(f"Processing data for: {user_name} (ID: {participant_id})")

    try:
        df_interactions = pd.read_csv(interactions_csv)
        df_users = pd.read_csv(users_csv)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return None

    # Standardize Timestamps
    df_interactions["created_at"] = pd.to_datetime(
        df_interactions["created_at"], format="mixed", utc=True
    )
    df_users["created_at"] = pd.to_datetime(
        df_users["created_at"], format="mixed", utc=True
    )

    user_interactions = df_interactions[df_interactions["participant_id"] == participant_id]
    
    if user_interactions.empty:
        print(f"No interactions found for ID {participant_id}.")
        return None

    br_static_summary = user_interactions[
        (user_interactions["stage_id"] == "brainrot_static") & 
        (user_interactions["interaction_type"] == "session_summary")
    ]

    if not br_static_summary.empty:
        end_time = br_static_summary.iloc[0]["created_at"]
        duration_sec = br_static_summary.iloc[0]["duration_ms"] / 1000.0
        
        session_start_time = end_time - pd.to_timedelta(duration_sec, unit='s')
        print(f"Session Start Time (T0) calculated from brainrot_static summary: {session_start_time}")
    else:
        print("WARNING: No session_summary for 'brainrot_static' found. Trying fallback.")
        br_static_events = user_interactions[user_interactions["stage_id"] == "brainrot_static"]
        if not br_static_events.empty:
            session_start_time = br_static_events["created_at"].min()
            print(f"Session Start Time (T0) from first brainrot_static event (fallback): {session_start_time}")
        else:
            print("ERROR: Could not find any 'brainrot_static' events to determine T0.")
            return None


    # Filter Scroll Gestures AND Target Stages
    scrolls = df_interactions[
        (df_interactions["participant_id"] == participant_id)
        & (df_interactions["interaction_type"] == "scroll_gesture")
        & (df_interactions["stage_id"].isin(TARGET_STAGES))
    ].copy()

    if scrolls.empty:
        print(f"No scroll gestures found for stages: {TARGET_STAGES}")
        return None

    # Calculate Relative Onset
    scrolls["onset"] = (scrolls["created_at"] - session_start_time).dt.total_seconds()
    scrolls["duration"] = scrolls["duration_ms"] / 1000.0

    # Create clear descriptions: "Scroll_brainrot_scroll"
    scrolls["description"] = "Scroll_" + scrolls["stage_id"].astype(str)

    result = scrolls[["onset", "duration", "description"]]
    result = result[result["onset"] >= 0]

    print(f"Successfully extracted {len(result)} scroll events.")
    print(f"Sample descriptions: {result['description'].unique()}")
    return result

def plot_scrolls_with_band(
    raw_obj,
    band_name="Beta",
    fmin=13,
    fmax=30,
    channel="C3",
    t_start=0,
    t_duration=20,
    title_suffix="",
):
    """
    Plots the Hilbert envelope (power) of a specific frequency band overlaying scroll events.
    """

    # Validation
    if channel not in raw_obj.ch_names:
        print(f"Channel {channel} not found. Using: {raw_obj.ch_names[0]}")
        channel = raw_obj.ch_names[0]

    try:
        raw_crop = (
            raw_obj.copy().pick([channel]).crop(tmin=t_start, tmax=t_start + t_duration)
        )
    except ValueError:
        print(f"Time window {t_start}-{t_start+t_duration}s is out of bounds.")
        return

    # Processing
    raw_crop.filter(fmin, fmax, verbose=False)
    raw_crop.apply_hilbert(envelope=True)

    data = raw_crop.get_data()[0] * 1e6
    times = raw_crop.times

    # Smoothing
    sfreq = raw_crop.info["sfreq"]
    window_size = int(sfreq * 0.15)
    data_series = pd.Series(data)
    data_smooth = (
        data_series.rolling(window=window_size, center=True).mean().fillna(data_series)
    )
    mean_power = np.mean(data)

    # Plotting Configuration
    plt.figure(figsize=(15, 6))

    # Colors
    if "Alpha" in band_name:
        main_color = "#6A0DAD"  # Purple
    elif "Beta" in band_name:
        main_color = "#D35400"  # Orange
    else:
        main_color = "#2E4053"  # Gray

    plt.plot(times, data, color=main_color, alpha=0.2, lw=0.5, label="Raw Envelope")
    plt.plot(times, data_smooth, color=main_color, lw=2, label=f"{band_name} Trend")
    plt.axhline(mean_power, color="gray", linestyle="--", alpha=0.7, label="Mean Power")

    # Overlay Scroll Events
    events = raw_crop.annotations
    legend_added = False

    for i in range(len(events)):
        desc = events.description[i]
        rel_onset = events.onset[i] - raw_crop.first_time
        duration = events.duration[i]
        rel_end = rel_onset + duration

        # Check if event is visible in window
        if "Scroll" in desc and rel_end > 0 and rel_onset < t_duration:
            label = "Scroll Gesture" if not legend_added else None

            # Dynamic coloring based on stage name
            color = "#E74C3C"  # Default Red
            if "brainrot" in desc.lower():
                color = "#C0392B"  # Dark Red
            elif "smart" in desc.lower():
                color = "#2980B9"  # Blue

            plt.axvspan(rel_onset, rel_end, color=color, alpha=0.2, label=label)
            plt.axvline(rel_end, color=color, linestyle=":", linewidth=1.5, alpha=0.8)
            legend_added = True

    final_title = f"{band_name} Band ({fmin}-{fmax} Hz) - {channel}"
    if title_suffix:
        final_title += f" | {title_suffix}"

    plt.title(final_title, fontsize=14)
    plt.xlabel("Time (s) relative to window start", fontsize=12)
    plt.ylabel("Power (µV)", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def compare_stages(raw_obj, band="Beta", fmin=13, fmax=30, channel="C3", duration=20):
    """
    Automatically finds the first scroll event in Brainrot and Smart Scroll
    and plots them for comparison.
    """
    print(f"--- COMPARATIVE ANALYSIS: {band} Band on {channel} ---")

    t_brainrot = None
    t_smart = None

    if raw_obj.annotations:
        for annot in raw_obj.annotations:
            desc = annot["description"]
            onset = annot["onset"]

            # --- FIX IS HERE: Looking for new string names ---
            if "brainrot_scroll" in desc and t_brainrot is None:
                t_brainrot = onset
            elif "smart_scroll" in desc and t_smart is None:
                t_smart = onset

            if t_brainrot is not None and t_smart is not None:
                break
    else:
        print("Error: No annotations found in raw object.")
        return

    # Plot Brainrot
    if t_brainrot is not None:
        plot_scrolls_with_band(
            raw_obj,
            band_name=band,
            fmin=fmin,
            fmax=fmax,
            channel=channel,
            t_start=t_brainrot - 2,
            t_duration=duration,
            title_suffix="BRAINROT SCROLL",
        )
    else:
        print("No events found for Brainrot Scroll ('Scroll_brainrot_scroll').")

    # Plot Smart
    if t_smart is not None:
        plot_scrolls_with_band(
            raw_obj,
            band_name=band,
            fmin=fmin,
            fmax=fmax,
            channel=channel,
            t_start=t_smart - 2,
            t_duration=duration,
            title_suffix="SMART SCROLL",
        )
    else:
        print("No events found for Smart Scroll ('Scroll_smart_scroll').")
