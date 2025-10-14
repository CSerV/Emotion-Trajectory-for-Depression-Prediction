#!/usr/bin/env python3
"""
Emotion Trajectory Feature Extraction Module

This module provides enhanced feature extraction for emotional trajectory analysis,
including linear statistics and second-order statistics for various emotion features:
- Emotional Inertia
- Rising/Falling Rates  
- Recovery Times
- Zero Crossing Rate

Date: September 2025
"""

import numpy as np
import pandas as pd
import torch


def _compute_within_window_second_order(x_window, y_vec, eps=1e-8):
    """Second-order stats within a single window.

    Returns a flattened vector in order:
    [mean_x(Dx), var_x(Dx), E[x^2](Dx), y^2(Dy), E[x]*y (Dx*Dy), E[Xi*Xj] (i<j), Yi*Yj (i<j)]
    """
    X = np.asarray(x_window, dtype=float)
    y = np.asarray(y_vec, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("x_window must be 2D (T, Dx)")
    T, Dx = X.shape
    Dy = y.shape[0]
    if T == 0:
        return np.zeros(Dx + Dx + Dx + Dy + Dx * Dy + (Dx * (Dx - 1)) // 2 + (Dy * (Dy - 1)) // 2)

    mean_x = X.mean(axis=0)
    var_x = X.var(axis=0) + eps
    ex2 = (X ** 2).mean(axis=0)
    y2 = y ** 2
    exy = (mean_x[:, None] * y[None, :]).flatten()

    exx_pairs = []
    for i in range(Dx):
        for j in range(i + 1, Dx):
            exx_pairs.append((X[:, i] * X[:, j]).mean())
    exx_pairs = np.array(exx_pairs, dtype=float) if exx_pairs else np.array([], dtype=float)

    yy_pairs = []
    for i in range(Dy):
        for j in range(i + 1, Dy):
            yy_pairs.append(y[i] * y[j])
    yy_pairs = np.array(yy_pairs, dtype=float) if yy_pairs else np.array([], dtype=float)

    return np.concatenate([mean_x, var_x, ex2, y2, exy, exx_pairs, yy_pairs])


def _compute_cross_window_linear_stats_with_std(X_summary, Y, eps=1e-8):
    """Compute cross-window linear stats and their standard errors.

    Returns (mean_flat, std_flat) where mean_flat order is
    [corr (Dx*Dy), slopes (Dx*Dy), cov (Dx*Dy)], and std_flat order is
    [se_corr, se_slope, se_cov].
    """
    X = np.asarray(X_summary, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2:
        raise ValueError("X_summary must be 2D (n_windows, Dx)")
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    if Y.ndim != 2:
        raise ValueError("Y must be 1D/2D (n_windows, Dy)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X_summary and Y must have the same number of windows")

    nW, Dx = X.shape
    Dy = Y.shape[1]
    if nW < 3:
        zeros = np.zeros(Dx * Dy, dtype=float)
        zeros3 = np.concatenate([zeros, zeros, zeros])
        return zeros3, zeros3

    # Center
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    var_x = Xc.var(axis=0, ddof=0) + eps
    var_y = Yc.var(axis=0, ddof=0) + eps

    cov_xy = (Xc.T @ Yc) / nW  # (Dx, Dy)
    corr_xy = cov_xy / (np.sqrt(var_x)[:, None] * np.sqrt(var_y)[None, :])
    slopes_xy = cov_xy / var_x[:, None]

    mean_flat = np.concatenate([
        corr_xy.flatten(),
        slopes_xy.flatten(),
        cov_xy.flatten(),
    ])

    # se_cov: std of per-window products divided by sqrt(nW)
    prod = (Xc[:, :, None] * Yc[:, None, :])  # (nW, Dx, Dy)
    se_cov = prod.reshape(nW, -1).std(axis=0, ddof=1) / np.sqrt(nW)

    # se_corr: Fisher z transform approximation
    r = corr_xy.flatten()
    se_r = (1 - r ** 2) / np.sqrt(max(nW - 3, 1))

    # se_slope: from simple linear regression
    E = Yc[:, None, :] - (Xc[:, :, None] * slopes_xy[None, :, :])  # (nW, Dx, Dy)
    var_e = E.reshape(nW, -1).var(axis=0, ddof=0) + eps
    var_x_rep = np.repeat(var_x, Dy)
    se_slope = np.sqrt(var_e / (max(nW - 2, 1) * var_x_rep + eps))

    std_flat = np.concatenate([se_r, se_slope, se_cov])
    return mean_flat, std_flat


def compute_emotional_inertia(emotion_list, time_lag=1):
    """Compute emotional inertia using autocorrelation
    
    Args:
        emotion_list: List or array of emotion values
        time_lag: Time lag for autocorrelation computation
        
    Returns:
        float: Emotional inertia value
    """
    
    # Changed to use mean squared difference instead of autocorrelation
    x = np.array(emotion_list)
    diffs = np.diff(x)
    MASD = np.mean(np.abs(diffs))
    # monotone decreasing transform
    inertia = 1 / (1 + MASD)
    # min-max normalize to [0,1]
    return inertia

    # return np.mean(np.abs(diffs))


def compute_session_emotional_inertia_enhanced(session_sequences, window_size=5, stride=2):
    """
    Compute enhanced emotional inertia with linear and second-order statistics
    
    Args:
        session_sequences: List of session emotion sequences
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (whole_inertia_list, avg_inertia_list, std_inertia_list, 
                linear_stats_list, second_order_stats_list)
    """
    whole_inertia_list = []
    avg_inertia_list = []
    std_inertia_list = []
    linear_stats_list = []
    second_order_stats_list = []
    
    for session in session_sequences:
        window_inertia_list = []
        window_second_order_stats = []
        X_summaries = []  # per-window mean x
        Y_targets = []    # per-window inertia vector

        window_size = min(window_size, session.shape[0])
        for i in range(0, len(session) - window_size + 1, stride):
            window = session[i:i + window_size]
            # Compute inertia values (Dy=3)
            arousal = window[:, 0]
            valence = window[:, 1]
            dominance = window[:, 2]
            arousal_inertia = compute_emotional_inertia(arousal)
            valence_inertia = compute_emotional_inertia(valence)
            dominance_inertia = compute_emotional_inertia(dominance)
            y_vec = np.array([arousal_inertia, valence_inertia, dominance_inertia])

            window_inertia_list.append(y_vec)

            # within-window second order
            second_vec = _compute_within_window_second_order(window, y_vec)
            window_second_order_stats.append(second_vec)

            # cross-window linear summaries
            X_summaries.append(window.mean(axis=0))
            Y_targets.append(y_vec)

        window_inertia_list = np.array(window_inertia_list)
        window_second_order_stats = np.array(window_second_order_stats)
        X_summaries = np.array(X_summaries)
        Y_targets = np.array(Y_targets)

        # original avg/std of inertia per session
        avg_inertia = window_inertia_list.mean(axis=0)
        std_inertia = window_inertia_list.std(axis=0)

        # Cross-window linear stats with std
        cross_linear, cross_linear_std = _compute_cross_window_linear_stats_with_std(np.array(X_summaries), np.array(Y_targets))

        # within-window second-order aggregated
        avg_second_order_stats = window_second_order_stats.mean(axis=0)
        std_second_order_stats = window_second_order_stats.std(axis=0)

        # Append to lists
        avg_inertia_list.append(avg_inertia)
        std_inertia_list.append(std_inertia)
        whole_inertia_list.append(window_inertia_list)

        linear_stats_list.append({
        'avg_linear_stats': cross_linear,
        'std_linear_stats': cross_linear_std,
            'raw_linear_stats': np.array([])
        })

        second_order_stats_list.append({
            'avg_second_order_stats': avg_second_order_stats,
            'std_second_order_stats': std_second_order_stats,
            'raw_second_order_stats': window_second_order_stats
        })
    # else:
    #     print(f"Session too short for windowing, use smaller window size than: {session.shape}")
    
    return (whole_inertia_list, avg_inertia_list, std_inertia_list, 
            linear_stats_list, second_order_stats_list)


def compute_emotion_rising_falling_rate_enhanced(session_sequences, window_size=5, stride=1):
    """
    Compute enhanced rising/falling rates with linear and second-order statistics
    
    Args:
        session_sequences: List of session emotion sequences
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (whole_rf_list, avg_rf_list, std_rf_list, linear_stats_list, second_order_stats_list)
    """
    whole_rising_falling_rates = []
    avg_rising_falling_rates = []
    std_rising_falling_rates = []
    linear_stats_list = []
    second_order_stats_list = []

    for session in session_sequences:
        window_rising_falling_rates = []
        window_second_order_stats = []
        X_summaries = []
        Y_targets = []

        window_size = min(window_size, session.shape[0])
        for i in range(0, len(session) - window_size + 1, stride):
            window = session[i:i + window_size]
            valence = window[:, 1]
            arousal = window[:, 0]
            dominance = window[:, 2]
            
            # Compute rising/falling rates (Dy=6)
            valence_rising_rate = np.sum(np.diff(valence) > 0) / len(valence)
            valence_falling_rate = np.sum(np.diff(valence) < 0) / len(valence)
            arousal_rising_rate = np.sum(np.diff(arousal) > 0) / len(arousal)
            arousal_falling_rate = np.sum(np.diff(arousal) < 0) / len(arousal)
            dominance_rising_rate = np.sum(np.diff(dominance) > 0) / len(dominance)
            dominance_falling_rate = np.sum(np.diff(dominance) < 0) / len(dominance)

            y_vec = np.array([
                valence_rising_rate, valence_falling_rate,
                arousal_rising_rate, arousal_falling_rate,
                dominance_rising_rate, dominance_falling_rate
            ])
            window_rising_falling_rates.append(y_vec)

            # within-window second order
            second_vec = _compute_within_window_second_order(window, y_vec)
            window_second_order_stats.append(second_vec)

            # cross-window summaries
            X_summaries.append(window.mean(axis=0))
            Y_targets.append(y_vec)
    
        # Aggregate
        window_rising_falling_rates = np.array(window_rising_falling_rates)
        window_second_order_stats = np.array(window_second_order_stats)
        X_summaries = np.array(X_summaries)
        Y_targets = np.array(Y_targets)
        
        avg_window_rising_falling_rates = np.mean(window_rising_falling_rates, axis=0)
        std_window_rising_falling_rates = np.std(window_rising_falling_rates, axis=0)
        
        cross_linear, cross_linear_std = _compute_cross_window_linear_stats_with_std(X_summaries, Y_targets)
        
        avg_second_order_stats = window_second_order_stats.mean(axis=0)
        std_second_order_stats = window_second_order_stats.std(axis=0)
        
        # Append to lists
        whole_rising_falling_rates.append(window_rising_falling_rates)
        avg_rising_falling_rates.append(avg_window_rising_falling_rates)
        std_rising_falling_rates.append(std_window_rising_falling_rates)
        
        linear_stats_list.append({
            'avg_linear_stats': cross_linear,
            'std_linear_stats': cross_linear_std,
            'raw_linear_stats': np.array([])
        })
        
        second_order_stats_list.append({
            'avg_second_order_stats': avg_second_order_stats,
            'std_second_order_stats': std_second_order_stats,
            'raw_second_order_stats': window_second_order_stats
        })

    return (whole_rising_falling_rates, avg_rising_falling_rates, std_rising_falling_rates,
            linear_stats_list, second_order_stats_list)


def compute_single_recovery_time(emotion_series, low_t, rec_t):
    """Compute the recovery time for a single emotion series using adaptive thresholds"""
    recovery_times = []
    i = 0
    n = len(emotion_series)

    while i < n:
        if emotion_series[i] < low_t:
            start = i
            i += 1
            while i < n and emotion_series[i] < rec_t:
                i += 1
            if i < n:
                recovery_times.append(i - start)
        else:
            i += 1

    return np.mean(recovery_times) if recovery_times else 0


def compute_multi_recovery_times_enhanced(session_sequences, window_size=10, stride=2):
    """
    Compute enhanced recovery times with linear and second-order statistics
    
    Args:
        session_sequences: List of session emotion sequences
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (whole_recovery_time, mean_recovery_time, std_recovery_time, 
                linear_stats_list, second_order_stats_list)
    """
    whole_recovery_time = []
    mean_recovery_time = []
    std_recovery_time = []
    linear_stats_list = []
    second_order_stats_list = []

    for session in session_sequences:
        window_recovery_time = []
        window_second_order_stats = []
        X_summaries = []
        Y_targets = []
        
        valence_session = session[:, 1]
        arousal_session = session[:, 0]
        dominance_session = session[:, 2]
        
        # Compute adaptive thresholds
        va_low_threshold = np.percentile(valence_session, 25)
        va_rec_threshold = np.percentile(valence_session, 50)
        ar_low_threshold = np.percentile(arousal_session, 25)
        ar_rec_threshold = np.percentile(arousal_session, 50)
        do_low_threshold = np.percentile(dominance_session, 25)
        do_rec_threshold = np.percentile(dominance_session, 50)

        window_size = min(window_size, session.shape[0])
        for i in range(0, session.shape[0] - window_size + 1, stride):
            end = i + window_size
            window = session[i:end]
            valence = window[:, 1]
            arousal = window[:, 0]
            dominance = window[:, 2]

            # Compute recovery times (Dy=3)
            valence_recovery = compute_single_recovery_time(valence, va_low_threshold, va_rec_threshold)
            arousal_recovery = compute_single_recovery_time(arousal, ar_low_threshold, ar_rec_threshold)
            dominance_recovery = compute_single_recovery_time(dominance, do_low_threshold, do_rec_threshold)

            y_vec = np.array([valence_recovery, arousal_recovery, dominance_recovery])
            window_recovery_time.append(y_vec)
            
            # within-window second order
            second_vec = _compute_within_window_second_order(window, y_vec)
            window_second_order_stats.append(second_vec)

            # cross-window summaries
            X_summaries.append(window.mean(axis=0))
            Y_targets.append(y_vec)
    
        # Aggregate
        window_recovery_time = np.array(window_recovery_time)
        window_second_order_stats = np.array(window_second_order_stats)
        X_summaries = np.array(X_summaries)
        Y_targets = np.array(Y_targets)
        
        avg_recovery = np.mean(window_recovery_time, axis=0)
        std_recovery = np.std(window_recovery_time, axis=0)
        
        cross_linear, cross_linear_std = _compute_cross_window_linear_stats_with_std(X_summaries, Y_targets)
        
        avg_second_order_stats = window_second_order_stats.mean(axis=0)
        std_second_order_stats = window_second_order_stats.std(axis=0)
        
        # Append to lists
        whole_recovery_time.append(window_recovery_time)
        mean_recovery_time.append(avg_recovery)
        std_recovery_time.append(std_recovery)
        
        linear_stats_list.append({
            'avg_linear_stats': cross_linear,
            'std_linear_stats': cross_linear_std,
            'raw_linear_stats': np.array([])
        })
        
        second_order_stats_list.append({
            'avg_second_order_stats': avg_second_order_stats,
            'std_second_order_stats': std_second_order_stats,
            'raw_second_order_stats': window_second_order_stats
        })

    return (whole_recovery_time, mean_recovery_time, std_recovery_time,
            linear_stats_list, second_order_stats_list)


def compute_single_emotion_zcr(series, threshold=0.5):
    """Compute zero crossing rate of an emotion sequence"""
    series = np.array(series)
    signs = np.sign(series - threshold)
    crossings = np.where(np.diff(signs))[0]
    zcr = len(crossings) / (len(series) - 1) if len(series) > 1 else 0
    return zcr


def compute_multi_emotion_zcr_enhanced(session_sequences, window_size=10, stride=2):
    """
    Compute enhanced ZCR with linear and second-order statistics
    
    Args:
        session_sequences: List of session emotion sequences
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (whole_emotion_zcr, mean_emotion_zcr, std_emotion_zcr,
                linear_stats_list, second_order_stats_list)
    """
    whole_emotion_zcr = []
    mean_emotion_zcr = []
    std_emotion_zcr = []
    linear_stats_list = []
    second_order_stats_list = []

    for session in session_sequences:
        window_emotion_zcr = []
        window_second_order_stats = []
        X_summaries = []
        Y_targets = []
        
        # Normalize emotion series
        emo_series_np = []
        for i in range(3):
            emotion_series = session[:, i]
            if np.std(emotion_series) > 0:
                emotion_series = (emotion_series - np.mean(emotion_series)) / np.std(emotion_series)
            emo_series_np.append(emotion_series)
        
        emo_series_np = np.array(emo_series_np).T

        window_size = min(window_size, emo_series_np.shape[0])
        for j in range(0, emo_series_np.shape[0] - window_size + 1, stride):
            end = j + window_size
            window = emo_series_np[j:end]
            valence = window[:, 1]
            arousal = window[:, 0]
            dominance = window[:, 2]

            # Compute ZCR (Dy=3)
            valence_zcr = compute_single_emotion_zcr(valence)
            arousal_zcr = compute_single_emotion_zcr(arousal)
            dominance_zcr = compute_single_emotion_zcr(dominance)

            y_vec = np.array([valence_zcr, arousal_zcr, dominance_zcr])
            window_emotion_zcr.append(y_vec)
            
            # within-window second order
            second_vec = _compute_within_window_second_order(window, y_vec)
            window_second_order_stats.append(second_vec)

            # cross-window summaries
            X_summaries.append(window.mean(axis=0))
            Y_targets.append(y_vec)

        # Convert to numpy arrays and compute statistics
        window_emotion_zcr = np.array(window_emotion_zcr)
        window_second_order_stats = np.array(window_second_order_stats)
        X_summaries = np.array(X_summaries)
        Y_targets = np.array(Y_targets)
        
        avg_zcr = np.mean(window_emotion_zcr, axis=0)
        std_zcr = np.std(window_emotion_zcr, axis=0)
        
        cross_linear, cross_linear_std = _compute_cross_window_linear_stats_with_std(X_summaries, Y_targets)
        
        avg_second_order_stats = window_second_order_stats.mean(axis=0)
        std_second_order_stats = window_second_order_stats.std(axis=0)
        
        # Append to lists
        whole_emotion_zcr.append(window_emotion_zcr)
        mean_emotion_zcr.append(avg_zcr)
        std_emotion_zcr.append(std_zcr)
        
        linear_stats_list.append({
            'avg_linear_stats': cross_linear,
            'std_linear_stats': cross_linear_std,
            'raw_linear_stats': np.array([])
        })
        
        second_order_stats_list.append({
            'avg_second_order_stats': avg_second_order_stats,
            'std_second_order_stats': std_second_order_stats,
            'raw_second_order_stats': window_second_order_stats
        })

    return (whole_emotion_zcr, mean_emotion_zcr, std_emotion_zcr,
            linear_stats_list, second_order_stats_list)


def pad_sequences_embedding(sequences, maxlen=None):
    """Pad sequences for consistent input to models"""
    if not sequences or len(sequences) == 0:
        return np.array([]), np.array([])
        
    lengths = [seq.shape[0] for seq in sequences]
    feat_dim = sequences[0].shape[1]
    
    if max(lengths) == 1:
        maxlen = 1
    else:
        maxlen = maxlen or max(lengths)
        
    padded = np.zeros((len(sequences), maxlen, feat_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded[i, :seq.shape[0], :] = seq
    return padded, np.array(lengths)


def get_enhanced_features(sessions, ws, st, feature_list, feature_type='avg+std+concatenation+linear_stats+second_order_stats', maxlen=204):
    """
    Enhanced feature extraction with linear and second-order statistics for all feature types
    
    Args:
        sessions: List of session sequences
        ws: Window size
        st: Stride
        feature_list: List of features to extract ['inertia', 'r_f_rates', 'recovery_times', 'zcr']
        feature_type: Type of features to extract ('avg', 'std', 'concatenation', or combinations)
        maxlen: Maximum sequence length for padding
        
    Returns:
        torch.Tensor: Concatenated feature tensor
    """
    avg_features = []
    std_features = []
    concatenation_features = []
    linear_stats_features = []
    second_order_stats_features = []

    # Filter out advanced_stats from feature_list if it exists
    filtered_feature_list = [f for f in feature_list if 'advanced_stats' not in str(f)]
    if len(filtered_feature_list) != len(feature_list):
        print(f"Warning: Filtered out 'advanced_stats' from feature list. Original: {feature_list}, Filtered: {filtered_feature_list}")
    feature_list = filtered_feature_list
    
    # ================ Enhanced Inertia Features ================
    if 'inertia' in feature_list:
        (raw_inertia, avg_inertia, std_inertia, 
         linear_stats, second_order_stats) = compute_session_emotional_inertia_enhanced(sessions, window_size=ws, stride=st)
        
        if 'avg' in feature_type:
            # Original inertia features
            avg_features.append(torch.tensor(avg_inertia, dtype=torch.float32))
    
        if 'std' in feature_type:
            # Original inertia features
            std_features.append(torch.tensor(std_inertia, dtype=torch.float32))
            
        if 'concatenation' in feature_type:
            concatenation_inertia, _ = pad_sequences_embedding(raw_inertia, maxlen=maxlen)
            concatenation_features.append(torch.tensor(concatenation_inertia, dtype=torch.float32).flatten(start_dim=1))
        
        if 'linear_stats' in feature_type:
            avg_linear_features = [stats['avg_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(avg_linear_features, dtype=torch.float32))
            std_linear_features = [stats['std_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(std_linear_features, dtype=torch.float32))
            
        if 'second_order_stats' in feature_type:
            avg_second_order_features = [stats['avg_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(avg_second_order_features, dtype=torch.float32))
            std_second_order_features = [stats['std_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(std_second_order_features, dtype=torch.float32))
    
    # ================ Enhanced Rising/Falling Rates Features ================
    if 'r_f_rates' in feature_list:
        (raw_rf, avg_rf, std_rf, 
         linear_stats, second_order_stats) = compute_emotion_rising_falling_rate_enhanced(sessions, window_size=ws, stride=st)
        
        if 'avg' in feature_type:
            # Original r_f_rates features
            avg_features.append(torch.tensor(avg_rf, dtype=torch.float32))
             
        if 'std' in feature_type:
            # Original r_f_rates features
            std_features.append(torch.tensor(std_rf, dtype=torch.float32))
                    
        if 'concatenation' in feature_type:
            concatenation_rf, _ = pad_sequences_embedding(raw_rf, maxlen=maxlen)
            concatenation_features.append(torch.tensor(concatenation_rf, dtype=torch.float32).flatten(start_dim=1))
            
        if 'linear_stats' in feature_type:
            avg_linear_features = [stats['avg_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(avg_linear_features, dtype=torch.float32))
            std_linear_features = [stats['std_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(std_linear_features, dtype=torch.float32))
            
        if 'second_order_stats' in feature_type:
            avg_second_order_features = [stats['avg_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(avg_second_order_features, dtype=torch.float32))
            std_second_order_features = [stats['std_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(std_second_order_features, dtype=torch.float32))
    
    # ================ Enhanced Recovery Times Features ================
    if 'recovery_times' in feature_list:
        (raw_rt, avg_rt, std_rt, 
         linear_stats, second_order_stats) = compute_multi_recovery_times_enhanced(sessions, window_size=ws, stride=st)
        
        if 'avg' in feature_type:
            # Original recovery_times features
            avg_features.append(torch.tensor(avg_rt, dtype=torch.float32))
      
        if 'std' in feature_type:
            # Original recovery_times features
            std_features.append(torch.tensor(std_rt, dtype=torch.float32))
                    
        if 'concatenation' in feature_type:
            concatenation_rt, _ = pad_sequences_embedding(raw_rt, maxlen=maxlen)
            concatenation_features.append(torch.tensor(concatenation_rt, dtype=torch.float32).flatten(start_dim=1))

        if 'linear_stats' in feature_type:
            avg_linear_features = [stats['avg_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(avg_linear_features, dtype=torch.float32))
            std_linear_features = [stats['std_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(std_linear_features, dtype=torch.float32))
            
        if 'second_order_stats' in feature_type:
            avg_second_order_features = [stats['avg_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(avg_second_order_features, dtype=torch.float32))
            std_second_order_features = [stats['std_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(std_second_order_features, dtype=torch.float32))
    
    # ================ Enhanced ZCR Features ================
    if 'zcr' in feature_list:
        (raw_zcr, avg_zcr, std_zcr, 
         linear_stats, second_order_stats) = compute_multi_emotion_zcr_enhanced(sessions, window_size=ws, stride=st)
        
        if 'avg' in feature_type:
            # Original zcr features
            avg_features.append(torch.tensor(avg_zcr, dtype=torch.float32))
                    
        if 'std' in feature_type:
            # Original zcr features
            std_features.append(torch.tensor(std_zcr, dtype=torch.float32))
        
        if 'concatenation' in feature_type:
            concatenation_zcr, _ = pad_sequences_embedding(raw_zcr, maxlen=maxlen)
            concatenation_features.append(torch.tensor(concatenation_zcr, dtype=torch.float32).flatten(start_dim=1))

        if 'linear_stats' in feature_type:
            avg_linear_features = [stats['avg_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(avg_linear_features, dtype=torch.float32))
            std_linear_features = [stats['std_linear_stats'] for stats in linear_stats]
            linear_stats_features.append(torch.tensor(std_linear_features, dtype=torch.float32))
            
        if 'second_order_stats' in feature_type:
            avg_second_order_features = [stats['avg_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(avg_second_order_features, dtype=torch.float32))
            std_second_order_features = [stats['std_second_order_stats'] for stats in second_order_stats]
            second_order_stats_features.append(torch.tensor(std_second_order_features, dtype=torch.float32))


    # Combine all feature types
    all_features = []
    if avg_features:
        all_features.extend(avg_features)
    if std_features:
        all_features.extend(std_features)
    if concatenation_features:
        all_features.extend(concatenation_features)
    if linear_stats_features:
        all_features.extend(linear_stats_features)
    if second_order_stats_features:
        all_features.extend(second_order_stats_features)

    return torch.cat(all_features, dim=1) if all_features else torch.empty(0, 0)


def form_session_sequences(csv_mapping_file, embedding_data):
    # Read CSV file
    # df_csv = pd.read_csv(csv_mapping_file)
    df_csv = csv_mapping_file
    df_combined = pd.DataFrame(df_csv, columns=['Wav-path','AVECParticipant_ID','Duration'])
    
    df_combined['embedding_feature'] = [embedding for embedding in embedding_data]
    
    df_combined_gt_1s = df_combined[df_combined['Duration'] >= 0.5]
    # Concatenate emotion vectors of the same session to form a sequence, eventually output a 3D numpy array
    # If emotion vectors belong to the same session, concatenate them together
    session_sequences = []
    
    # Sort df_combined_gt_1s by AVECParticipant_ID and Wav-path to ensure consistent order
    # This is very important to ensure the order of embedding to align with labels!!
    df_combined_gt_1s = df_combined_gt_1s.sort_values(by=['AVECParticipant_ID', 'Wav-path'])

    for session_id in df_combined_gt_1s['AVECParticipant_ID'].unique():
        session_data = df_combined_gt_1s[df_combined_gt_1s['AVECParticipant_ID'] == session_id]
        if not session_data.empty:
            session_embedding = session_data['embedding_feature'].values
            # Turn the list of embeddings into a numpy array
            session_embedding = np.array(session_embedding.tolist())
            session_sequences.append(session_embedding)
    return session_sequences
    
    
def get_window_based_selected_features(csv_mapping_file, pdem_embedding_data, vad_embedding_data, 
                                      percentage, feature_name='inertia', avd_feature='valence', window_size=10, stride=2, normalize=True):
    """
    Select PDEM embeddings based on ranking of trajectory features computed on VAD windows.
    
    Args:
        csv_mapping_file: DataFrame with mapping info
        pdem_embedding_data: PDEM embeddings aligned with csv rows
        vad_embedding_data: VAD embeddings aligned with csv rows  
        percentage: Top percentage of windows to select
        feature_name: Feature to compute for ranking ('inertia', 'r_f_rates', 'recovery_times', 'zcr')
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (selected_pdem_embeddings_list, selected_vad_embeddings_list) - lists per session
    """
    # Use relative imports since we're in the same module
    # from emotion_trajectory_features_select import (
    #     compute_emotional_inertia,
    #     compute_single_emotion_zcr
    # )
    
    # Form VAD session sequences
    vad_session_sequences = form_session_sequences(csv_mapping_file, vad_embedding_data)
    
    # Also form PDEM sequences for alignment
    pdem_session_sequences = form_session_sequences(csv_mapping_file, pdem_embedding_data)
    
    selected_pdem_list = []
    selected_vad_list = []

    avd_feature = avd_feature.split('_')
    
    for session_idx, vad_session in enumerate(vad_session_sequences):
        pdem_session = pdem_session_sequences[session_idx]
        
        if len(vad_session) < window_size:
            # If session too short, use all segments
            selected_pdem_list.append(pdem_session)
            selected_vad_list.append(vad_session)
            continue
            
        # Compute feature values for each window
        window_features = []
        window_indices = []

        valence_session = vad_session[:, 1]
        arousal_session = vad_session[:, 0]
        dominance_session = vad_session[:, 2]

        if normalize:
            # Normalize valence for ZCR computation
            if np.std(valence_session) > 0:
                valence_session = (valence_session - np.mean(valence_session)) / np.std(valence_session)
            if np.std(arousal_session) > 0:
                arousal_session = (arousal_session - np.mean(arousal_session)) / np.std(arousal_session)
            if np.std(dominance_session) > 0:
                dominance_session = (dominance_session - np.mean(dominance_session)) / np.std(dominance_session)

        va_low_threshold = np.percentile(valence_session, 25)
        va_rec_threshold = np.percentile(valence_session, 50)
        ar_low_threshold = np.percentile(arousal_session, 25)
        ar_rec_threshold = np.percentile(arousal_session, 50)
        do_low_threshold = np.percentile(dominance_session, 25)
        do_rec_threshold = np.percentile(dominance_session, 50)
        
        for i in range(0, len(vad_session) - window_size + 1, stride):
            arousal = arousal_session[i:i + window_size]
            valence = valence_session[i:i + window_size]
            dominance = dominance_session[i:i + window_size]
            window = vad_session[i:i + window_size]
            window_indices.append(i)

            # valence, arousal, dominance = window[:, 1], window[:, 0], window[:, 2]

            # Compute feature based on feature_name
            if feature_name == 'inertia':
                feature_values = []
                # Compute inertia for arousal, valence, dominance
                # arousal_inertia = compute_emotional_inertia(window[:, 0])
                # valence_inertia = compute_emotional_inertia(window[:, 1]) 
                # dominance_inertia = compute_emotional_inertia(window[:, 2])
                if 'valence' in avd_feature:
                    feature_values.append(compute_emotional_inertia(valence))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_emotional_inertia(arousal))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_emotional_inertia(dominance))
                
                if len(avd_feature) > 1:
                    feature_value = np.mean(feature_values)
                elif len(avd_feature) == 1:
                    feature_value = feature_values[0]
                # Use mean inertia as ranking criterion
                # feature_value = np.mean([arousal_inertia, valence_inertia, dominance_inertia])
                # feature_value = np.mean([arousal_inertia, valence_inertia])

            elif feature_name == 'r_f_rates':
                feature_values = []
                # Compute rising/falling rates for valence
                # valence_rising_rate = np.sum(np.diff(window[:, 1]) > 0) / len(window)
                # valence_falling_rate = np.sum(np.diff(window[:, 1]) < 0) / len(window)
                if 'valence' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(valence) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(valence) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                elif 'arousal' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(arousal) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(arousal) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                elif 'dominance' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(dominance) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(dominance) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                
                if len(avd_feature) > 1:
                    feature_value = np.mean(feature_values)
                elif len(avd_feature) == 1:
                    feature_value = feature_values[0]

                #feature_value = abs(feature_rising_rate - feature_falling_rate)

            elif feature_name == 'recovery_times':
                # Compute recovery times for valence
                feature_values = []
                #valence_recovery_times = compute_multi_recovery_times_enhanced(window[:, 1].reshape(-1, 1))
                # Use mean recovery time as ranking criterion
                if 'valence' in avd_feature:
                    feature_values.append(compute_single_recovery_time(valence, va_low_threshold, va_rec_threshold))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_single_recovery_time(arousal, ar_low_threshold, ar_rec_threshold))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_single_recovery_time(dominance, do_low_threshold, do_rec_threshold))
                
                feature_value = np.mean(feature_values)

            elif feature_name == 'zcr':
                feature_values = []
                # Compute ZCR for valence (normalized)
                if 'valence' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(valence))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(arousal))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(dominance))
                
                feature_value = np.mean(feature_values)

            else:
                # Default to valence variance
                feature_value = np.var(valence)

            window_features.append(feature_value)
        
        # Select top percentage windows based on feature values
        window_features = np.array(window_features)
        window_indices = np.array(window_indices)
        
        # Calculate deviations from mean
        feature_mean = np.mean(window_features)
        feature_deviations = np.abs(window_features - feature_mean)
        
        # Sort windows by absolute deviation from mean (descending)
        # This will select windows that are either much higher or much lower than average
        sorted_indices = np.argsort(feature_deviations)[::-1]
        
        # print(f"Session {session_idx}: Feature values: {window_features}")
        # print(f"Session {session_idx}: Feature mean: {feature_mean:.4f}")
        # print(f"Session {session_idx}: Feature deviations: {feature_deviations}")
        # print(f"Session {session_idx}: Selected indices (by deviation): {sorted_indices[:max(1, int(len(sorted_indices) * percentage / 100))]}")
        
        # print(f"Session {session_idx}: Feature values: {window_features}")
        # Select top percentage
        n_select = max(1, int(len(sorted_indices) * percentage / 100))
        selected_window_indices = sorted_indices[:n_select]
        
        # Get corresponding segment indices for selected windows
        selected_segment_indices = []
        for win_idx in selected_window_indices:
            start_idx = window_indices[win_idx]
            end_idx = start_idx + window_size
            selected_segment_indices.extend(range(start_idx, end_idx))
        
        # Remove duplicates and sort
        selected_segment_indices = sorted(list(set(selected_segment_indices)))
        
        # Select corresponding PDEM and VAD embeddings
        selected_pdem = pdem_session[selected_segment_indices]
        selected_vad = vad_session[selected_segment_indices]
        
        # Mean the selected segments to form a single embedding per session
        selected_pdem_mean = np.mean(selected_pdem, axis=0, keepdims=True)
        selected_vad_mean = np.mean(selected_vad, axis=0, keepdims=True)

        selected_pdem_std = np.std(selected_pdem, axis=0, keepdims=True)
        selected_vad_std = np.std(selected_vad, axis=0, keepdims=True)

        # Concatenate mean and std
        selected_pdem_mean_std = np.concatenate([selected_pdem_mean, selected_pdem_std], axis=1)
        selected_vad_mean_std = np.concatenate([selected_vad_mean, selected_vad_std], axis=1)

        selected_pdem_list.append(selected_pdem_mean_std.squeeze(0))
        selected_vad_list.append(selected_vad_mean_std.squeeze(0))

    # return as numpy arrays
    return np.array(selected_pdem_list), np.array(selected_vad_list)


def get_window_based_selected_features_test(pdem_session_sequences, vad_session_sequences, 
                                      percentage, feature_name='inertia', avd_feature='valence', window_size=10, stride=2, normalize=True):
    """
    Select PDEM embeddings based on ranking of trajectory features computed on VAD windows.
    
    Args:
        csv_mapping_file: DataFrame with mapping info
        pdem_embedding_data: PDEM embeddings aligned with csv rows
        vad_embedding_data: VAD embeddings aligned with csv rows  
        percentage: Top percentage of windows to select
        feature_name: Feature to compute for ranking ('inertia', 'r_f_rates', 'recovery_times', 'zcr')
        window_size: Size of sliding window
        stride: Stride for sliding window
        
    Returns:
        tuple: (selected_pdem_embeddings_list, selected_vad_embeddings_list) - lists per session
    """
    # Use functions from the same module - no import needed
    # from emotion_trajectory_features_select import (
    #     compute_emotional_inertia,
    #     compute_single_emotion_zcr
    # )
    
    # Form VAD session sequences
    # vad_session_sequences = form_session_sequences(csv_mapping_file, vad_embedding_data)
    
    # # Also form PDEM sequences for alignment
    # pdem_session_sequences = form_session_sequences(csv_mapping_file, pdem_embedding_data)
    
    selected_pdem_list = []
    selected_vad_list = []

    avd_feature = avd_feature.split('_')
    
    for session_idx, vad_session in enumerate(vad_session_sequences):
        pdem_session = pdem_session_sequences[session_idx]
        
        if len(vad_session) < window_size:
            # If session too short, use all segments
            mean_pdem = np.mean(pdem_session, axis=0, keepdims=True)
            mean_vad = np.mean(vad_session, axis=0, keepdims=True)

            std_pdem = np.std(pdem_session, axis=0, keepdims=True)
            std_vad = np.std(vad_session, axis=0, keepdims=True)

            selected_pdem_list.append(np.concatenate([mean_pdem, std_pdem], axis=1).squeeze(0))
            selected_vad_list.append(np.concatenate([mean_vad, std_vad], axis=1).squeeze(0))
            continue
            
        # Compute feature values for each window
        window_features = []
        window_indices = []

        valence_session = vad_session[:, 1]
        arousal_session = vad_session[:, 0]
        dominance_session = vad_session[:, 2]

        # if normalize:
        #     # Normalize valence for ZCR computation
        #     if np.std(valence_session) > 0:
        #         valence_session = (valence_session - np.mean(valence_session)) / np.std(valence_session)
        #     if np.std(arousal_session) > 0:
        #         arousal_session = (arousal_session - np.mean(arousal_session)) / np.std(arousal_session)
        #     if np.std(dominance_session) > 0:
        #         dominance_session = (dominance_session - np.mean(dominance_session)) / np.std(dominance_session)

        va_low_threshold = np.percentile(valence_session, 25)
        va_rec_threshold = np.percentile(valence_session, 50)
        ar_low_threshold = np.percentile(arousal_session, 25)
        ar_rec_threshold = np.percentile(arousal_session, 50)
        do_low_threshold = np.percentile(dominance_session, 25)
        do_rec_threshold = np.percentile(dominance_session, 50)

        # window_size = min(window_size, len(vad_session))

        for i in range(0, len(vad_session) - window_size + 1, stride):
            arousal = arousal_session[i:i + window_size]
            valence = valence_session[i:i + window_size]
            dominance = dominance_session[i:i + window_size]
            window = vad_session[i:i + window_size]
            window_indices.append(i)

            # valence, arousal, dominance = window[:, 1], window[:, 0], window[:, 2]
            
            # Normalize window if needed
            if normalize:
                if np.std(valence) > 0:
                    valence = (valence - np.mean(valence)) / np.std(valence)
                if np.std(arousal) > 0:
                    arousal = (arousal - np.mean(arousal)) / np.std(arousal)
                if np.std(dominance) > 0:
                    dominance = (dominance - np.mean(dominance)) / np.std(dominance)

            # Compute feature based on feature_name
            if feature_name == 'inertia':
                feature_values = []
                # Compute inertia for arousal, valence, dominance
                # arousal_inertia = compute_emotional_inertia(window[:, 0])
                # valence_inertia = compute_emotional_inertia(window[:, 1]) 
                # dominance_inertia = compute_emotional_inertia(window[:, 2])
                if 'valence' in avd_feature:
                    feature_values.append(compute_emotional_inertia(valence))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_emotional_inertia(arousal))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_emotional_inertia(dominance))
                
                if len(avd_feature) > 1:
                    feature_value = np.mean(feature_values)
                elif len(avd_feature) == 1:
                    feature_value = feature_values[0]
                # Use mean inertia as ranking criterion
                # feature_value = np.mean([arousal_inertia, valence_inertia, dominance_inertia])
                # feature_value = np.mean([arousal_inertia, valence_inertia])

            elif feature_name == 'r_f_rates':
                feature_values = []
                # Compute rising/falling rates for valence
                # valence_rising_rate = np.sum(np.diff(window[:, 1]) > 0) / len(window)
                # valence_falling_rate = np.sum(np.diff(window[:, 1]) < 0) / len(window)
                if 'valence' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(valence) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(valence) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                elif 'arousal' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(arousal) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(arousal) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                elif 'dominance' in avd_feature:
                    feature_rising_rate = np.sum(np.diff(dominance) > 0) / len(window)
                    feature_falling_rate = np.sum(np.diff(dominance) < 0) / len(window)
                    feature_values.append(abs(feature_rising_rate - feature_falling_rate))
                
                if len(avd_feature) > 1:
                    feature_value = np.mean(feature_values)
                elif len(avd_feature) == 1:
                    feature_value = feature_values[0]

                #feature_value = abs(feature_rising_rate - feature_falling_rate)

            elif feature_name == 'recovery_times':
                # Compute recovery times for valence
                feature_values = []
                #valence_recovery_times = compute_multi_recovery_times_enhanced(window[:, 1].reshape(-1, 1))
                # Use mean recovery time as ranking criterion
                if 'valence' in avd_feature:
                    feature_values.append(compute_single_recovery_time(valence, va_low_threshold, va_rec_threshold))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_single_recovery_time(arousal, ar_low_threshold, ar_rec_threshold))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_single_recovery_time(dominance, do_low_threshold, do_rec_threshold))
                
                feature_value = np.mean(feature_values)

            elif feature_name == 'zcr':
                feature_values = []
                # Compute ZCR for valence (normalized)
                if 'valence' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(valence))
                elif 'arousal' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(arousal))
                elif 'dominance' in avd_feature:
                    feature_values.append(compute_single_emotion_zcr(dominance))
                
                feature_value = np.mean(feature_values)

            else:
                # Default to valence variance
                feature_value = np.var(valence)

            window_features.append(feature_value)
        
        # Select top percentage windows based on feature values
        window_features = np.array(window_features)
        window_indices = np.array(window_indices)
        
        # Calculate deviations from mean
        feature_mean = np.mean(window_features)
        feature_deviations = np.abs(window_features - feature_mean)
        
        # Sort windows by absolute deviation from mean (descending)
        # This will select windows that are either much higher or much lower than average
        sorted_indices = np.argsort(feature_deviations)[::-1]
        
        # print(f"Session {session_idx}: Feature values: {window_features}")
        # print(f"Session {session_idx}: Feature mean: {feature_mean:.4f}")
        # print(f"Session {session_idx}: Feature deviations: {feature_deviations}")
        # print(f"Session {session_idx}: Selected indices (by deviation): {sorted_indices[:max(1, int(len(sorted_indices) * percentage / 100))]}")
        
        # print(f"Session {session_idx}: Feature values: {window_features}")
        # Select top percentage
        n_select = max(1, int(len(sorted_indices) * percentage / 100))
        selected_window_indices = sorted_indices[:n_select]
        
        # Get corresponding segment indices for selected windows
        selected_segment_indices = []
        for win_idx in selected_window_indices:
            start_idx = window_indices[win_idx]
            end_idx = start_idx + window_size
            selected_segment_indices.extend(range(start_idx, end_idx))
        
        # Remove duplicates and sort
        selected_segment_indices = sorted(list(set(selected_segment_indices)))
        
        # Select corresponding PDEM and VAD embeddings
        selected_pdem = pdem_session[selected_segment_indices]
        selected_vad = vad_session[selected_segment_indices]
        
        # Mean the selected segments to form a single embedding per session
        selected_pdem_mean = np.mean(selected_pdem, axis=0, keepdims=True)
        selected_vad_mean = np.mean(selected_vad, axis=0, keepdims=True)

        selected_pdem_std = np.std(selected_pdem, axis=0, keepdims=True)
        selected_vad_std = np.std(selected_vad, axis=0, keepdims=True)

        # Concatenate mean and std
        selected_pdem_mean_std = np.concatenate([selected_pdem_mean, selected_pdem_std], axis=1)
        selected_vad_mean_std = np.concatenate([selected_vad_mean, selected_vad_std], axis=1)

        if selected_vad_mean_std.shape[0] != 1:
            print(f"Warning: selected_vad_mean_std has unexpected shape {selected_vad_mean_std.shape}, expected (1,6)")

        selected_pdem_list.append(selected_pdem_mean_std.squeeze(0))
        selected_vad_list.append(selected_vad_mean_std.squeeze(0))

    # return as numpy arrays
    return np.array(selected_pdem_list), np.array(selected_vad_list)


def analyze_enhanced_feature_dimensions():
    """
    Analyze and report the dimensions of enhanced features.
    Computes dimensions based on current formulas to avoid drift.
    """

    def linear_dims(Dx, Dy):
        # [corr, slopes, cov] each Dx*Dy
        return 3 * Dx * Dy

    def second_order_dims(Dx, Dy):
        # [mean_x(Dx), var_x(Dx), ex2(Dx), y2(Dy), exy(Dx*Dy), exx_pairs(C(Dx,2)), yy_pairs(C(Dy,2))]
        from math import comb
        return (Dx + Dx + Dx + Dy + Dx * Dy + comb(Dx, 2) + comb(Dy, 2))

    Dx = 3
    feature_info = {
        'inertia': {
            'original': 3,
            'linear_stats': linear_dims(Dx, 3),
            'second_order_stats': second_order_dims(Dx, 3),
        },
        'r_f_rates': {
            'original': 6,
            'linear_stats': linear_dims(Dx, 6),
            'second_order_stats': second_order_dims(Dx, 6),
        },
        'recovery_times': {
            'original': 3,
            'linear_stats': linear_dims(Dx, 3),
            'second_order_stats': second_order_dims(Dx, 3),
        },
        'zcr': {
            'original': 3,
            'linear_stats': linear_dims(Dx, 3),
            'second_order_stats': second_order_dims(Dx, 3),
        }
    }

    # Add totals
    for k, v in feature_info.items():
        v['total_per_session'] = v['original'] + v['linear_stats'] + v['second_order_stats']
    
    print("Enhanced Feature Dimensions Analysis:")
    print("=" * 60)
    
    total_enhanced_dims = 0
    for feature_name, info in feature_info.items():
        print(f"\n{feature_name.upper()} Features:")
        print(f"  Original dimensions: {info['original']}")
        print(f"  Linear statistics: {info['linear_stats']}")
        print(f"  Second-order statistics: {info['second_order_stats']}")
        print(f"  Total per session: {info['total_per_session']}")
        
        total_enhanced_dims += info['total_per_session']
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total enhanced dimensions (all features): {total_enhanced_dims}")
    print(f"  Enhancement factor: ~{total_enhanced_dims/15:.1f}x original")
    print(f"  Original total dimensions: 15 (3+6+3+3)")
    
    return feature_info


if __name__ == "__main__":
    # Run analysis when script is executed directly
    analyze_enhanced_feature_dimensions()
    
    # I have some test code here
    # Build one session as (10,3) directly (columns correspond to 3 emotion dims)
    arousal_seq = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    arousal_seq_2 = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
    arousal_seq_3 = [0.1, 0.09, 0.1, 0.09, 0.1, 0.09, 0.1, 0.09, 0.1, 0.09]

    # Concatenate three (10,) lists into a float32 (10,3) numpy array
    session_sequences_np = np.column_stack([arousal_seq, arousal_seq_2, arousal_seq_3]).astype(np.float32)
    # Wrap as a list of sessions
    series_list = [session_sequences_np]

    test_features = get_enhanced_features(series_list, 5, 2, ['inertia'])

    (raw_inertia, avg_inertia, std_inertia, 
         linear_stats, second_order_stats) = compute_session_emotional_inertia_enhanced(series_list, window_size=5, stride=2)
    (raw_r_f_rate, avg_r_f_rate, std_r_f_rate, linear_stats_r_f, second_order_stats_r_f) = compute_emotion_rising_falling_rate_enhanced(series_list, window_size=5, stride=2)
    (raw_recovery_time, avg_recovery_time, std_recovery_time, linear_stats_rt, second_order_stats_rt) = compute_multi_recovery_times_enhanced(series_list, window_size=5, stride=2)
    (raw_zcr, avg_zcr, std_zcr, linear_stats_zcr, second_order_stats_zcr) = compute_multi_emotion_zcr_enhanced(series_list, window_size=5, stride=2)