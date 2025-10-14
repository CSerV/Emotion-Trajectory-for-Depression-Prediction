from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, normalize, PowerTransformer
from sklearn.svm import SVC, SVR
from sklearn.metrics import balanced_accuracy_score, make_scorer, mean_absolute_error, root_mean_squared_error, recall_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, Lasso
from scipy.stats import zscore, skew, kurtosis
from scipy import stats
from sklearn.utils.estimator_checks import check_estimator
from elm_kernel import ELM
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations
from datetime import datetime
import warnings
import pandas as pd
import numpy as np
import pickle
import torch
import sys

sys.path.append('/home/legalalien/Documents/Jiawei/EmoTracjectory')

# Import enhanced feature extraction functions
from utils.emotion_trajectory_features_select import (
    compute_emotional_inertia,
    compute_single_recovery_time
)

# Ignore all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# For PDEM

# Load map file
train_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_train.csv'
dev_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_dev.csv'
test_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_test.csv'


# Load feature file
PDEM_w2v2_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_train/w2v2_train.pkl'
PDEM_w2v2_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_dev/w2v2_dev.pkl'
PDEM_w2v2_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_test/w2v2_test.pkl'
VAD_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_train/vad_train.pkl'
VAD_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_dev/vad_dev.pkl'
VAD_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_test/vad_test.pkl'


# Load the data
train_data = pd.read_csv(train_mapped_label_file, sep = ',')
test_data = pd.read_csv(test_mapped_label_file, sep = ',')
dev_data = pd.read_csv(dev_mapped_label_file, sep = ',')
# Load the pickle files
train_PDEM = pd.read_pickle(PDEM_w2v2_train_file).to_numpy()
test_PDEM = pd.read_pickle(PDEM_w2v2_test_file).to_numpy()
dev_PDEM = pd.read_pickle(PDEM_w2v2_dev_file).to_numpy()

train_VAD = pd.read_pickle(VAD_train_file).to_numpy()
test_VAD = pd.read_pickle(VAD_test_file).to_numpy()
dev_VAD = pd.read_pickle(VAD_dev_file).to_numpy()

# Labels for train and dev
symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep = ';')
# PHQ_8Total is the total score of PHQ-8
PHQ_sev_train = symptoms_data.loc[56:, 'PHQ_8Total']
PHQ_sev_dev = symptoms_data.loc[:55, 'PHQ_8Total']

binary_labels = pd.read_csv('metadata.csv', sep = ',')
# 0: not depressed, 1: depressed
binary_train_label = binary_labels.loc[0:162, 'PHQ_Binary']
binary_dev_label = binary_labels.loc[163:, 'PHQ_Binary']

# labels for test
test_label = pd.read_csv('test_split.csv', sep = ',')
binary_test_label = test_label.loc[:, 'PHQ_Binary']

# Form Session sequences. Emotion vectors of the same session need to be concatenated together
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
    
# train_session_sequences = form_session_sequences(train_data, train_VAD)
# test_session_sequences = form_session_sequences(test_data, test_VAD)
# dev_session_sequences = form_session_sequences(dev_data, dev_VAD)


def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

def test_feature_combination(feature_type, vad_feature, ws, st, 
                           train_features, dev_features, 
                           train_severity, dev_severity, train_symptoms, dev_symptoms,
                           symptom_means_list, C_range, kernels, percentage=100):
    """Test a specific feature combination"""
    print(f"\n=== Testing: Type={feature_type}, VAD feature={vad_feature}, Percentage={percentage}% ===")
    
    combo_results = []

    print(f"Window size: {ws}, Stride: {st}")

    if len(train_features) == 0 or len(dev_features) == 0:
        print(f"Empty features for ws={ws}, st={st}, skipping...")
        return []
    
    # Convert to numpy
    train_features_np = train_features
    dev_features_np = dev_features

    # Scale features using StandardScaler
    scaler = StandardScaler().fit(train_features_np)
    X_train_scale = normalize(scaler.transform(train_features_np))
    X_dev_scale = normalize(scaler.transform(dev_features_np))
    
    # Add bias term for ELM
    X_train_scale = np.insert(X_train_scale, 0, 1, axis=1)
    X_dev_scale = np.insert(X_dev_scale, 0, 1, axis=1)
    
    # Test ELM with different C and kernel values
    best_ccc = -1
    best_params = None
    best_predictions = None
    best_metrics = None
    
    # print all hyperparameter combinations
    print("Testing hyperparameter combinations:")
    
    
    for C in C_range:
        for kernel in kernels:
            print(f"Testing C={C}, kernel={kernel}")
            try:
                # Train ELM
                elm = ELM(c=C, kernel=kernel, is_classification=False, weighted=True)
                elm.fit(X_train_scale, train_symptoms)

                # Predict symptoms, using original dev_features_np for prediction
                symps_pred_dev = elm.predict(X_dev_scale)

                # Denormalize symptoms
                for symp in range(8):
                    symps_pred_dev[:, symp] = (symps_pred_dev[:, symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp, 0]
                    # ccc_symp = concordance_correlation_coefficient(dev_symptoms[:, symp], symps_pred_dev[:, symp])
                    # rmse_symp = root_mean_squared_error(dev_symptoms[:, symp], symps_pred_dev[:, symp], squared=False)
                    # print(f"CCC for symptom {symp}: {ccc_symp:.3f}")
                    # print(f"RMSE for symptom {symp}: {rmse_symp:.3f}")
                    # # write to csv file
                    # with open("ccc_rmse_symptoms_valence_ezcr_dev_1.csv", "a") as f:
                    #     f.write(f"{symp},{rmse_symp:.3f},{ccc_symp:.3f}\n")
                    
                # Save CCC for each symptom if needed
                # e.g., save to a csv file
                # with open("ccc_symptoms_valence_ezcr_dev.csv", "a") as f:
                #     for symp, ccc in enumerate(ccc_symptoms):
                #         f.write(f"{symp},{ccc:.3f}\n")

                # Sanitize and round symptoms
                
                # symps_pred_dev = np.clip(symps_pred_dev, 0, 3)
                # symps_pred_dev = np.rint(symps_pred_dev)
                
                # Sanitizing symptoms:
                symps_pred_dev[symps_pred_dev < 0] = 0
                symps_pred_dev[symps_pred_dev > 3] = 3
                
                # Rounding symptoms:
                symps_pred_dev = np.rint(symps_pred_dev)

                w_symptoms_performance = []
                for symp in range(8):
                    ccc_symp = concordance_correlation_coefficient(dev_symptoms[:, symp], symps_pred_dev[:, symp])
                    rmse_symp = root_mean_squared_error(dev_symptoms[:, symp], symps_pred_dev[:, symp], squared=False)
                    # print(f"CCC for symptom {symp}: {ccc_symp:.3f}")
                    # print(f"RMSE for symptom {symp}: {rmse_symp:.3f}")
                    # # write to csv file
                    # with open("ccc_rmse_symptoms_valence_inertia_dev.csv", "a") as f:
                    #     f.write(f"{symp},{rmse_symp:.3f},{ccc_symp:.3f}\n")

                # Sum to get total severity
                pred_dev_summation = np.sum(symps_pred_dev, axis=1)
                pred_dev_summation = np.clip(pred_dev_summation, 0, 24)
                
                # Calculate metrics
                mae = mean_absolute_error(dev_severity, pred_dev_summation)
                rmse = root_mean_squared_error(dev_severity, pred_dev_summation, squared=False)
                ccc = concordance_correlation_coefficient(dev_severity, pred_dev_summation)
                r2 = r2_score(dev_severity, pred_dev_summation)
                
                # Update best if this is better
                if ccc > best_ccc:
                    best_ccc = ccc
                    best_params = {'C': C, 'kernel': kernel}
                    best_predictions = pred_dev_summation.copy()
                    best_metrics = {'mae': mae, 'rmse': rmse, 'ccc': ccc, 'r2': r2}
                    
            except Exception as e:
                print(f"Error with C={C}, kernel={kernel}: {str(e)}")
                continue
    
    if best_params is not None:
        combo_results.append({
            'feature_type': feature_type,
            'vad_feature': vad_feature,
            'window_size': ws,
            'stride': st,
            'percentage': percentage,  # Add percentage parameter
            'best_C': best_params['C'],
            'best_kernel': best_params['kernel'],
            'mae': best_metrics['mae'],
            'rmse': best_metrics['rmse'],
            'ccc': best_metrics['ccc'],
            'r2': best_metrics['r2'],
            'feature_dim': train_features_np.shape[1],
            'predictions': best_predictions
        })
        
        print(f"Best params: C={best_params['C']}, kernel={best_params['kernel']}")
        print(f"Results: MAE={best_metrics['mae']:.3f}, RMSE={best_metrics['rmse']:.3f}, CCC={best_metrics['ccc']:.3f}, R²={best_metrics['r2']:.3f}")
    
    return combo_results


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
    from utils.emotion_trajectory_features_select import (
        compute_emotional_inertia,
        compute_single_emotion_zcr
    )
    
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
            
            # Normalize window if needed
            # if normalize:
            #     if np.std(valence) > 0:
            #         valence = (valence - np.mean(valence)) / np.std(valence)
            #     if np.std(arousal) > 0:
            #         arousal = (arousal - np.mean(arousal)) / np.std(arousal)
            #     if np.std(dominance) > 0:
            #         dominance = (dominance - np.mean(dominance)) / np.std(dominance)

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
        
        # # Get the selected feature deviations for debugging
        # selected_feature_deviations = feature_deviations[sorted_indices[:max(1, int(len(sorted_indices) * percentage / 100))]]

        # # Save the mean of selected feature deviations for analysis
        # selected_feature_deviation_mean = np.mean(selected_feature_deviations)
        # selected_feature_deviation_std = np.std(selected_feature_deviations)
        # # Save to a file if needed
        # with open("selected_feature_deviation_means.csv", "a") as f:
        #     f.write(f"{session_idx},{selected_feature_deviation_mean:.4f},{selected_feature_deviation_std:.4f}  \n")

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


# ============================================================================
# MAIN EXPERIMENT EXECUTION
# ============================================================================


result_dict = {'Data_use_top_percentage': [], 'RMSE': [], 'CCC': []}

test_config_dict = {1: {'available_feature_types': ['inertia'], 'available_vad_features': ['arousal'], 'window_sizes': [15], 'strides': [14], 'Data_use_top_percentage_range': [70], 'C_range': [9], 'kernels': ['linear']},
                    2: {'available_feature_types': ['inertia'], 'available_vad_features': ['valence'], 'window_sizes': [20], 'strides': [5], 'Data_use_top_percentage_range': [60], 'C_range': [10], 'kernels': ['linear']},
                    3: {'available_feature_types': ['r_f_rates'], 'available_vad_features': ['arousal'], 'window_sizes': [25], 'strides': [10], 'Data_use_top_percentage_range': [80], 'C_range': [8], 'kernels': ['linear']},
                    4: {'available_feature_types': ['r_f_rates'], 'available_vad_features': ['valence'], 'window_sizes': [25], 'strides': [14], 'Data_use_top_percentage_range': [70], 'C_range': [9], 'kernels': ['linear']},
                    5: {'available_feature_types': ['recovery_times'], 'available_vad_features': ['arousal'], 'window_sizes': [20], 'strides': [3], 'Data_use_top_percentage_range': [80], 'C_range': [8], 'kernels': ['linear']},
                    6: {'available_feature_types': ['recovery_times'], 'available_vad_features': ['valence'], 'window_sizes': [30], 'strides': [16], 'Data_use_top_percentage_range': [60], 'C_range': [8], 'kernels': ['linear']},
                    7: {'available_feature_types': ['zcr'], 'available_vad_features': ['arousal'], 'window_sizes': [20], 'strides': [4], 'Data_use_top_percentage_range': [80], 'C_range': [10], 'kernels': ['linear']},
                    8: {'available_feature_types': ['zcr'], 'available_vad_features': ['valence'], 'window_sizes': [20], 'strides': [4], 'Data_use_top_percentage_range': [70], 'C_range': [10], 'kernels': ['linear']},
                   }


# Feature types to test
# available_feature_types = ['std', 'linear_stats', 'std+linear_stats', 'second_order_stats'] # 'avg', 'std', 'concatenation', 'avg+std', 'avg+concatenation', 'std+concatenation', 'avg+std+concatenation'
# available_features = ['inertia', 'r_f_rates', 'recovery_times', 'zcr']
available_feature_types = ['inertia', 'r_f_rates', 'recovery_times', 'zcr']


# available_vad_features = ['valence', 'arousal', 'dominance', 'valence_arousal', 'valence_dominance', 'arousal_dominance', 'valence_arousal_dominance']
available_vad_features = ['valence', 'arousal']

# Window size and stride parameters for feature extraction
# window_sizes = [10, 15, 20, 25, 30]
# strides = [3, 4, 5, 10, 14, 16, 20]

window_sizes = [25]
strides = [14]

# ELM parameters
# C_range = [1,2,3,4, 5, 6, 7, 8, 9, 10]
C_range = [10]

kernels = ['linear']

# Load severity and symptom data
symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep=';')
PHQ_sev_train = symptoms_data.loc[56:, 'PHQ_8Total']
PHQ_sev_dev = symptoms_data.loc[:55, 'PHQ_8Total']

PHQ_symptoms = symptoms_data.drop(columns=['Participant_ID', 'AVECParticipant_ID', 'PHQ_8Total']).to_numpy()
PHQ_symptoms_train = PHQ_symptoms[56:]
PHQ_symptoms_dev = PHQ_symptoms[:56]

y_train_sev = np.array(PHQ_sev_train)
y_dev_sev = np.array(PHQ_sev_dev)

# Normalize symptoms for training
y_train_symp_unscaled = PHQ_symptoms_train
symptom_means_list = []
y_train_symp = np.zeros((len(y_train_symp_unscaled), 8))
for i in range(8):
    mean_i = np.mean(y_train_symp_unscaled[:, i])
    sd_i = np.std(y_train_symp_unscaled[:, i])
    symptom_means_list.append([mean_i, sd_i])
    y_train_symp[:, i] = (y_train_symp_unscaled[:, i] - mean_i) / sd_i

symptom_means_list = np.array(symptom_means_list)

# Generate all possible feature combinations
print("=== COMPREHENSIVE FEATURE COMBINATION TESTING (ENHANCED VERSION) ===")
print(f"Available feature types: {available_feature_types}")
print(f"Available VAD features: {available_vad_features}")
print(f"Window sizes: {window_sizes}")
print(f"Strides: {strides}")
print(f"C range: {C_range}")
print(f"Kernels: {kernels}")


# Store all results
all_experiment_results = []
start_time = datetime.now()

Data_use_top_percentage_range = [20, 30, 40, 50, 60, 70, 80, 90]

for Data_use_top_percentage in Data_use_top_percentage_range:
    # Data_use_top_percentage = 60

    # Test all combinations
    for j, feature_type in enumerate(available_feature_types):
        for vad_feature in available_vad_features:
            print(f"\n{'='*60}")
            print(f"Testing combination {i*len(available_feature_types) + j + 1}/{len(available_feature_types)}")
            print(f"Feature type: {feature_type}, VAD feature: {vad_feature}, Top percentage: {Data_use_top_percentage}%")

            for ws in window_sizes:
                for st in strides:
                    if ws >= st:
                        try:
                            feature_index = feature_type
                            # Use window-based feature selection instead of get_chosen_features
                            print(f"Using window-based selection with feature: {feature_index}")
                            print(f"Selection window size: {ws}, stride: {st}")

                            dev_selected_pdem, dev_selected_vad = get_window_based_selected_features(
                                dev_data, dev_PDEM, dev_VAD, Data_use_top_percentage, 
                                feature_index, vad_feature, ws, st)
                            
                            train_selected_pdem, train_selected_vad = get_window_based_selected_features(
                                train_data, train_PDEM, train_VAD, Data_use_top_percentage, 
                                feature_index, vad_feature, ws, st)

                            # Use the selected VAD sequences for trajectory feature computation
                            print(f"Selected {len(train_selected_vad)} train sessions, {len(dev_selected_vad)} dev sessions")
                            
                            # Replace the original session sequences with selected ones
                            # train_feature_selected = np.concatenate((train_selected_vad, train_selected_pdem), axis=1)
                            # dev_feature_selected = np.concatenate((dev_selected_vad, dev_selected_pdem), axis=1)

                            train_feature_selected = train_selected_pdem
                            dev_feature_selected = dev_selected_pdem

                            combo_results = test_feature_combination(
                                feature_type, vad_feature, ws, st,
                                train_feature_selected, dev_feature_selected,
                                y_train_sev, y_dev_sev, y_train_symp, PHQ_symptoms_dev,
                                symptom_means_list, C_range, kernels, Data_use_top_percentage  # Pass percentage parameter
                            )
                            
                            all_experiment_results.extend(combo_results)
                        
                        except Exception as e:
                            print(f"Error with ws={ws}, st={st}: {str(e)}")
                            continue

# Save results to CSV
if all_experiment_results:
    results_df = pd.DataFrame(all_experiment_results)
    
    # Sort by CCC (best first)
    results_df = results_df.sort_values('ccc', ascending=False)
    
    # Save to CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'D_selecting_results_{timestamp}.csv'
    results_df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to: {csv_filename}")
    
    # Print top 10 results
    print(f"\n{'='*60}")
    print("TOP 10 BEST RESULTS (by CCC) - ENHANCED FEATURES:")
    print("="*60)
    
    for idx, row in results_df.head(10).iterrows():
        print(f"Rank {results_df.index.get_loc(idx) + 1}:")
        # print(f"  Features: {row['features']}")
        print(f"  Feature Type: {row['feature_type']}")
        print(f"  VAD Feature: {row['vad_feature']}")
        print(f"  Window Size: {row['window_size']}, Stride: {row['stride']}")
        print(f"  Percentage: {row['percentage']}%")  # Add percentage display
        print(f"  Best C: {row['best_C']}, Best Kernel: {row['best_kernel']}")
        print(f"  MAE: {row['mae']:.4f}, RMSE: {row['rmse']:.4f}")
        print(f"  CCC: {row['ccc']:.4f}, R²: {row['r2']:.4f}")
        print(f"  Feature Dim: {row['feature_dim']}")
        print()
    
    # Best overall result
    best_result = results_df.iloc[0]
    print(f"{'='*60}")
    print("BEST OVERALL RESULT (ENHANCED FEATURES):")
    print("="*60)
    # print(f"Features: {best_result['features']}")
    print(f"Feature Type: {best_result['feature_type']}")
    print(f"VAD Feature: {best_result['vad_feature']}")
    print(f"Window Size: {best_result['window_size']}, Stride: {best_result['stride']}")
    print(f"Percentage: {best_result['percentage']}%")  # Add percentage display
    print(f"Best C: {best_result['best_C']}, Best Kernel: {best_result['best_kernel']}")
    print(f"MAE: {best_result['mae']:.4f}")
    print(f"RMSE: {best_result['rmse']:.4f}")
    print(f"CCC: {best_result['ccc']:.4f}")
    print(f"R²: {best_result['r2']:.4f}")
    print(f"Feature Dimension: {best_result['feature_dim']}")
    
    # Analysis by feature type
    print(f"\n{'='*60}")
    print("ANALYSIS BY FEATURE TYPE:")
    print("="*60)
    for ft in available_feature_types:
        ft_results = results_df[results_df['feature_type'] == ft]
        if not ft_results.empty:
            best_ccc = ft_results['ccc'].max()
            avg_ccc = ft_results['ccc'].mean()
            print(f"{ft}: Best CCC = {best_ccc:.4f}, Average CCC = {avg_ccc:.4f}, Count = {len(ft_results)}")
    
    # Analysis by VAD feature
    print(f"\n{'='*60}")
    print("ANALYSIS BY VAD FEATURE:")
    print("="*60)
    for vad_feat in available_vad_features:
        vad_results = results_df[results_df['vad_feature'] == vad_feat]
        if not vad_results.empty:
            best_ccc = vad_results['ccc'].max()
            avg_ccc = vad_results['ccc'].mean()
            print(f"{vad_feat}: Best CCC = {best_ccc:.4f}, Average CCC = {avg_ccc:.4f}, Count = {len(vad_results)}")
    
    # Add analysis by percentage
    print(f"\n{'='*60}")
    print("ANALYSIS BY PERCENTAGE:")
    print("="*60)
    for pct in Data_use_top_percentage_range:
        pct_results = results_df[results_df['percentage'] == pct]
        if not pct_results.empty:
            best_ccc = pct_results['ccc'].max()
            avg_ccc = pct_results['ccc'].mean()
            print(f"{pct}%: Best CCC = {best_ccc:.4f}, Average CCC = {avg_ccc:.4f}, Count = {len(pct_results)}")

else:
    print("No results obtained!")

print(f"\nTotal time elapsed: {datetime.now() - start_time}")
print("Experiment completed with enhanced features!")
