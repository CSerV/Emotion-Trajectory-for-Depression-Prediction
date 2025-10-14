#!/usr/bin/env python3
"""
Title: Test Best Parameters on Test Set

This script loads the best parameters from validation results and evaluates them on the test set.
It selects the best performing parameters for each feature combination from the validation results CSV.
"""

import numpy as np
import pickle
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from elm_kernel import ELM
import warnings
from datetime import datetime
import ast
import os
import sys

sys.path.append('/home/legalalien/Documents/Jiawei/EmoTracjectory')

# Import enhanced feature extraction functions
from utils.emotion_trajectory_features_select import (
    get_window_based_selected_features_test,
)

# Ignore all RuntimeWarnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# DATA LOADING AND PREPROCESSING (copied from DEV script)
# ============================================================================

# Load map files
train_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_train.csv'
dev_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_dev.csv'
test_mapped_label_file = '/media/sac_research/Data/AVEC2019_EDAIC/split_audio_labels/detailed_test.csv'

# Load feature files
VAD_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_train/vad_train.pkl'
VAD_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_dev/vad_dev.pkl'
VAD_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_VAD_embedding/vad_test/vad_test.pkl'

# Load PDEM features
PDEM_w2v2_train_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_train/w2v2_train.pkl'
PDEM_w2v2_dev_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_dev/w2v2_dev.pkl'
PDEM_w2v2_test_file = '/home/legalalien/Documents/Jiawei/DAIC_WOZ_pdem_embedding/w2v2_test/w2v2_test.pkl'

# Load the pickle files
train_PDEM = pd.read_pickle(PDEM_w2v2_train_file).to_numpy()
test_PDEM = pd.read_pickle(PDEM_w2v2_test_file).to_numpy()
dev_PDEM = pd.read_pickle(PDEM_w2v2_dev_file).to_numpy()

# Load the data
train_data = pd.read_csv(train_mapped_label_file, sep=',')
dev_data = pd.read_csv(dev_mapped_label_file, sep=',')
test_data = pd.read_csv(test_mapped_label_file, sep=',')

# Load VAD features
train_VAD = pd.read_pickle(VAD_train_file).to_numpy()
dev_VAD = pd.read_pickle(VAD_dev_file).to_numpy()
test_VAD = pd.read_pickle(VAD_test_file).to_numpy()

# Load labels and symptoms data
symptoms_data = pd.read_csv('Detailed_PHQ8_Labels_mapped.csv', sep=';')
# PHQ_8Total is the total score of PHQ-8 (0-24)
PHQ_sev_train = symptoms_data.loc[56:, 'PHQ_8Total']
PHQ_sev_dev = symptoms_data.loc[:55, 'PHQ_8Total']

# Extract individual symptoms (8 columns, each 0-3)
symptom_cols = [c for c in symptoms_data.columns if c not in ['Participant_ID', 'AVECParticipant_ID', 'PHQ_8Total']]
symptoms_all = symptoms_data[symptom_cols].to_numpy()  # shape [N,8]

# Align counts (dev then train ordering) consistent with earlier code
dev_symptoms = symptoms_all[:56]
train_symptoms = symptoms_all[56:]

# Test labels from test_split.csv
test_split_data = pd.read_csv('test_split.csv', sep=',')
test_labels = test_split_data['PHQ_Score']  # Use PHQ_Score as the test labels

print(f"Data loaded successfully!")
print(f"Train samples: {len(PHQ_sev_train)}, Dev samples: {len(PHQ_sev_dev)}, Test samples: {len(test_labels)}")

# ============================================================================
# HELPER FUNCTIONS (copied from DEV script)
# ============================================================================

def form_session_sequences(csv_mapping_file, embedding_data):
    """Form Session sequences"""
    df_csv = csv_mapping_file
    df_combined = pd.DataFrame(df_csv, columns=['Wav-path', 'AVECParticipant_ID', 'Duration'])
    
    df_combined['embedding_feature'] = [embedding for embedding in embedding_data]
    
    df_combined_gt_1s = df_combined[df_combined['Duration'] >= 1.0]
    
    session_sequences = []
    avg_session_embedding = []
    std_session_embedding = []
    
    # Sort df_combined_gt_1s by AVECParticipant_ID and Wav-path to ensure consistent order
    # This is very important to ensure the order of embedding to align with labels!!
    df_combined_gt_1s = df_combined_gt_1s.sort_values(by=['AVECParticipant_ID', 'Wav-path'])

    
    for session_id in df_combined_gt_1s['AVECParticipant_ID'].unique():
        session_data = df_combined_gt_1s[df_combined_gt_1s['AVECParticipant_ID'] == session_id]
        if not session_data.empty:
            session_embedding = session_data['embedding_feature'].values
            session_embedding = np.array(session_embedding.tolist())
            avg_session_embedding.append(np.mean(session_embedding, axis=0))
            std_session_embedding.append(np.std(session_embedding, axis=0))
            session_sequences.append(session_embedding)
    return session_sequences, avg_session_embedding, std_session_embedding

# Feature computation functions have been moved to emotion_trajectory_features.py module

def concordance_correlation_coefficient(y_true, y_pred):
    """Compute concordance correlation coefficient."""
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

# ============================================================================
# MAIN TESTING FUNCTIONS
# ============================================================================

def load_best_parameters_from_csv(csv_file_path):
    """Load best parameters for each feature combination from validation results"""
    print(f"Loading best parameters from: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    print(f"Total validation experiments: {len(df)}")
    print(f"Unique feature combinations: {df['features'].nunique()}")

    df['percentage'] = pd.to_numeric(df['percentage'], errors='coerce')
    df = df[df['percentage'] != 90]
    print(f"Filtered out rows with percentage == 90. Remaining: {len(df)}")
    
    best_configs = []
    
    # Group by feature combination and find the best one (highest CCC)
    #for feature_combo in df['features'].unique():
    for (ftype, vad_feat), group_df in df.groupby(['feature_type', 'vad_feature']):
        # Skip feature combinations that contain 'advanced_stats'
        # if 'advanced_stats' in feature_combo:
        #     print(f"Skipping feature combination containing 'advanced_stats': {feature_combo}")
        #     continue
        # Drop rows with NaN CCC
        group_df = group_df.dropna(subset=['ccc'])
        if group_df.empty:
            print(f"Skipping group (feature_type={ftype}, vad_feature={vad_feat}) due to empty/NaN ccc or advanced_stats only.")
            continue
            
        # feature_data = df[df['features'] == feature_combo]
        # best_idx = feature_data['ccc'].idxmax()
        # best_row = feature_data.loc[best_idx]

        # Pick the row with the highest CCC within this group
        best_idx = group_df['ccc'].astype(float).idxmax()
        best_row = group_df.loc[best_idx]
        
        # Parse features list
        try:
            features_list = ast.literal_eval(best_row['features'])
        except:
            features_list = [best_row['features']]
        
        # Double check: skip if any feature in the list contains 'advanced_stats'
        if any('advanced_stats' in str(feature) for feature in features_list):
            print(f"Skipping feature list containing 'advanced_stats': {features_list}")
            continue
        
        config = {
            'features': features_list,
            'feature_type': best_row['feature_type'],
            'vad_feature': best_row['vad_feature'],
            'window_size': int(best_row['window_size']),
            'stride': int(best_row['stride']),
            'percentage': int(best_row['percentage']),
            'best_C': float(best_row['best_C']),
            'best_kernel': best_row['best_kernel'],
            'validation_ccc': best_row['ccc'],
            'validation_mae': best_row['mae'],
            'validation_rmse': best_row['rmse'],
            'validation_r2': best_row['r2'],
            'feature_dim': best_row['feature_dim']
        }
        
        best_configs.append(config)
    
    # Sort by validation CCC (highest first)
    best_configs.sort(key=lambda x: x['validation_ccc'], reverse=True)
    
    print(f"Found {len(best_configs)} unique feature combinations (after filtering out 'advanced_stats')")
    print("Top 10 configurations by validation CCC:")
    for i, config in enumerate(best_configs[:10]):
        features_str = ', '.join(map(str, config['features']))
        print(f"  {i+1:2d}. ({config['feature_type']}, {config['vad_feature']}) "
              f"[{features_str}] - CCC: {config['validation_ccc']:.4f}, MAE: {config['validation_mae']:.3f}")
    # for i, config in enumerate(best_configs[:10]):
    #     features_str = ', '.join(config['features'])
    #     print(f"  {i+1:2d}. [{features_str:35s}] - CCC: {config['validation_ccc']:.4f}, MAE: {config['validation_mae']:.3f}")
    
    return best_configs

def test_configuration_on_testset(config, train_dev_pdem_session_sequences, test_pdem_session_sequences, 
                                  train_dev_vad_session_sequences, test_vad_session_sequences,
                                train_symptoms, test_labels,
                                symptom_means_list):
    """Test a single configuration on the test set"""
    
    # features = config['features']
    feature_type = config['feature_type']
    window_size = config['window_size']
    stride = config['stride']
    vad_feature = config['vad_feature']
    percentage = config['percentage']
    # percentage = 100  # Limit percentage upper bound
    C = config['best_C']
    # C = min(config['best_C'], 10.0)  # Limit C value upper bound
    # C = 16
    kernel = config['best_kernel']
    # kernel = 'linear'  # Force using linear kernel
    feature_dim = config['feature_dim']
    
    # feature_type = 'zcr'
    # vad_feature = 'valence'
    # percentage = 100
    # window_size = 20
    # stride = 4
    # kernel = 'linear'
    # C = 10
    
    #print(f"Testing {features} with WS={window_size}, ST={stride}, C={C}, kernel={kernel}")
    print(f"Testing {feature_type}, VAD={vad_feature}, with WS={window_size}, ST={stride}, PCT={percentage}, C={C}, kernel={kernel}")
    
    # Extract features for train and test
    train_sel_pdem, train_sel_vad = get_window_based_selected_features_test(
            train_dev_pdem_session_sequences, train_dev_vad_session_sequences, percentage,
            feature_type, vad_feature, window_size, stride
        )
    test_sel_pdem, test_sel_vad = get_window_based_selected_features_test(
            test_pdem_session_sequences, test_vad_session_sequences, percentage,
            feature_type, vad_feature, window_size, stride
        )
    
    # Check feature distribution differences
    train_mean = train_sel_pdem.mean().item()
    test_mean = test_sel_pdem.mean().item()
    distribution_shift = abs(train_mean - test_mean) / (abs(train_mean) + 1e-8)
    
    if distribution_shift > 0.5:
        print(f"Warning: Large distribution shift detected: {distribution_shift:.3f}")
    
    # Convert to numpy
    train_features_np = train_sel_pdem
    test_features_np = test_sel_pdem

    # Normalize features
    scaler = StandardScaler()
    train_features_np = scaler.fit_transform(train_features_np)
    test_features_np = scaler.transform(test_features_np)
    
    # Normalize to unit norm
    train_features_np = normalize(train_features_np)
    test_features_np = normalize(test_features_np)
    
    # Add bias term for ELM
    train_features_np = np.insert(train_features_np, 0, 1, axis=1)
    test_features_np = np.insert(test_features_np, 0, 1, axis=1)
    
    # Train ELM with normalized symptoms
    elm = ELM(c=C, kernel=kernel, is_classification=False, weighted=True)
    elm.fit(train_features_np, train_symptoms)

    
    # Predict symptoms on test set
    symps_pred_test = elm.predict(test_features_np)
    
    # Denormalize symptoms
    for symp in range(8):
        symps_pred_test[:, symp] = (symps_pred_test[:, symp] * symptom_means_list[symp, 1]) + symptom_means_list[symp, 0]
    
    # Sanitize and round symptoms
    # symps_pred_test = np.rint(symps_pred_test)
    # symps_pred_test = np.clip(symps_pred_test, 0, 3)
    
    # Sum to get total severity
    pred_test_summation = np.sum(symps_pred_test, axis=1)
    pred_test_summation = np.clip(pred_test_summation, 0, 24)
    
    nan_instances = [36,40]
    # Excluding NaNs (the two test instances which the AVEC'19 competetion has left out), 
    # then compute scores
    test_labels_array = test_labels.values if hasattr(test_labels, 'values') else np.array(test_labels)
    
    # Remove nan_instances from both arrays
    # Method 1: Using np.delete (current approach)
    test_labels_clean = np.delete(test_labels_array, nan_instances, axis=0)
    pred_test_summation = np.delete(pred_test_summation, nan_instances, axis=0)
    
    print(f"Removed {len(nan_instances)} nan instances. Remaining samples: {len(test_labels_clean)}")
    
    
    # Calculate metrics
    mae = mean_absolute_error(test_labels_clean, pred_test_summation)
    rmse = root_mean_squared_error(test_labels_clean, pred_test_summation, squared=False)
    r2 = r2_score(test_labels_clean, pred_test_summation)
    ccc = concordance_correlation_coefficient(test_labels_clean, pred_test_summation)

    results = {
        # 'features': str(features),
        'feature_type': feature_type,
        'vad_feature': vad_feature,
        'window_size': window_size,
        'stride': stride,
        'percentage': percentage,
        'C': C,
        'kernel': kernel,
        'test_mae': mae,
        'test_rmse': rmse,
        'test_r2': r2,
        'test_ccc': ccc,
        'validation_ccc': config['validation_ccc'],
        'validation_mae': config['validation_mae'],
        'validation_rmse': config['validation_rmse'],
        'feature_dim': feature_dim,
        'predictions': pred_test_summation,
        'true_labels': test_labels.values,
        'predicted_symptoms': symps_pred_test
    }
    
    print(f"Results: MAE={mae:.3f}, RMSE={rmse:.3f}, CCC={ccc:.3f}, R²={r2:.3f}")
    
    return results
    
    # except Exception as e:
    #     print(f"Error testing configuration: {str(e)}")
    #     return None

def main():
    """Main function to test best parameters on test set"""
    print("="*60)
    print("TESTING BEST VALIDATION PARAMETERS ON TEST SET")
    print("NOTE: Excluding 'advanced_stats' features from evaluation")
    print("="*60)
    
    # Load validation results
    # validation_csv_path = 'enhanced_feature_test_results_20250914_203741_select_win_norm.csv'
    validation_csv_path = 'D_selecting_4fold_CV_results_20251008_150455.csv'
    
    
    if not os.path.exists(validation_csv_path):
        print(f"Error: {validation_csv_path} not found!")
        return
    
    best_configs = load_best_parameters_from_csv(validation_csv_path)
    
    
    test_config_dict = {
    1: {'available_feature_types': ['inertia'], 'available_vad_features': ['arousal'], 'window_sizes': [15], 'strides': [14], 'Data_use_top_percentage_range': [70], 'C_range': [9], 'kernels': ['linear']},
    2: {'available_feature_types': ['inertia'], 'available_vad_features': ['valence'], 'window_sizes': [20], 'strides': [5], 'Data_use_top_percentage_range': [60], 'C_range': [10], 'kernels': ['linear']},
    3: {'available_feature_types': ['r_f_rates'], 'available_vad_features': ['arousal'], 'window_sizes': [25], 'strides': [10], 'Data_use_top_percentage_range': [80], 'C_range': [8], 'kernels': ['linear']},
    4: {'available_feature_types': ['r_f_rates'], 'available_vad_features': ['valence'], 'window_sizes': [25], 'strides': [14], 'Data_use_top_percentage_range': [70], 'C_range': [9], 'kernels': ['linear']},
    5: {'available_feature_types': ['recovery_times'], 'available_vad_features': ['arousal'], 'window_sizes': [20], 'strides': [3], 'Data_use_top_percentage_range': [80], 'C_range': [8], 'kernels': ['linear']},
    6: {'available_feature_types': ['recovery_times'], 'available_vad_features': ['valence'], 'window_sizes': [30], 'strides': [16], 'Data_use_top_percentage_range': [60], 'C_range': [8], 'kernels': ['linear']},
    7: {'available_feature_types': ['zcr'], 'available_vad_features': ['arousal'], 'window_sizes': [20], 'strides': [4], 'Data_use_top_percentage_range': [80], 'C_range': [10], 'kernels': ['linear']},
    8: {'available_feature_types': ['zcr'], 'available_vad_features': ['valence'], 'window_sizes': [20], 'strides': [4], 'Data_use_top_percentage_range': [70], 'C_range': [10], 'kernels': ['linear']},
}
    
    
    # Form session sequences
    print("\nForming session sequences...")
    train_session_sequences_vad, _, _ = form_session_sequences(train_data, train_VAD)
    dev_session_sequences_vad, _, _ = form_session_sequences(dev_data, dev_VAD)
    test_session_sequences_vad, _, _ = form_session_sequences(test_data, test_VAD)
    
    train_session_sequences_pdem, _, _ = form_session_sequences(train_data, train_PDEM)
    dev_session_sequences_pdem, _, _ = form_session_sequences(dev_data, dev_PDEM)
    test_session_sequences_pdem, _, _ = form_session_sequences(test_data, test_PDEM)

    train_dev_session_sequences_vad = dev_session_sequences_vad + train_session_sequences_vad  # dev first, train second
    train_dev_session_sequences_pdem = dev_session_sequences_pdem + train_session_sequences_pdem  # Keep consistent with label order

    # train_dev_session_sequences_vad = train_session_sequences_vad
    # train_dev_session_sequences_pdem = train_session_sequences_pdem

    print(f"Train sessions: {len(train_dev_session_sequences_vad)} (Train+Dev)")
    print(f"Dev sessions: {len(dev_session_sequences_vad)}")
    print(f"Test sessions: {len(test_session_sequences_vad)}")

    # Prepare symptom normalization (using same normalization as training)
    print("\nPreparing symptom normalization...")
    # Fix label merge order: keep consistent with feature merge order (dev first, train second)
    train_dev_symptoms = np.concatenate((dev_symptoms, train_symptoms), axis=0)  # dev first, train second
    # train_dev_symptoms = train_symptoms  # Previous incorrect code
    # Normalize symptoms (mean=0, std=1)
    symptom_means_list = []
    train_symptoms_normalized = np.zeros((len(train_dev_symptoms), 8))
    for i in range(8):
        mean_symp = np.mean(train_dev_symptoms[:, i])
        std_symp = np.std(train_dev_symptoms[:, i]) + 1e-8
        train_symptoms_normalized[:, i] = (train_dev_symptoms[:, i] - mean_symp) / std_symp
        symptom_means_list.append([mean_symp, std_symp])
    
    symptom_means_list = np.array(symptom_means_list)
    
    # Test each best configuration on test set
    print(f"\nTesting {len(best_configs)} configurations on test set...")
    
    all_test_results = []
    
    for i, config in enumerate(best_configs):
        feature_combo = ', '.join(config['feature_type'])
        print(f"\n--- Testing configuration {i+1}/{len(best_configs)}: {config['feature_type']} ---")
        
        result = test_configuration_on_testset(
            config, train_dev_session_sequences_pdem, test_session_sequences_pdem,
            train_dev_session_sequences_vad, test_session_sequences_vad,
            train_symptoms_normalized, test_labels,
            symptom_means_list
        )
        
        if result:
            all_test_results.append(result)
    
    # Save results
    print(f"\n--- SUMMARY OF RESULTS ---")
    print(f"Successfully tested {len(all_test_results)} configurations")
    
    if all_test_results:
        # Sort by test CCC
        all_test_results.sort(key=lambda x: x['test_ccc'], reverse=True)
        
        print(f"\nTop 5 results on test set:")
        for i, result in enumerate(all_test_results[:5]):
            print(f"{i+1}. {result['feature_type']} - ")
            print(f"   Test  - CCC: {result['test_ccc']:.4f}, MAE: {result['test_mae']:.3f}, RMSE: {result['test_rmse']:.3f}")
            print(f"   Valid - CCC: {result['validation_ccc']:.4f}, MAE: {result['validation_mae']:.3f}")
            print(f"   Params: WS={result['window_size']}, ST={result['stride']}, C={result['C']}, kernel={result['kernel']}")
            print()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'test_set_results_{timestamp}.csv'
        
        # Prepare data for CSV (exclude predictions for file size)
        results_for_csv = []
        for result in all_test_results:
            result_copy = result.copy()
            del result_copy['predictions']
            del result_copy['true_labels'] 
            del result_copy['predicted_symptoms']
            results_for_csv.append(result_copy)
        
        df_results = pd.DataFrame(results_for_csv)
        df_results.to_csv(output_file, index=False)
        print(f"Detailed results saved to: {output_file}")
        
        # Save predictions separately
        predictions_file = f'test_set_predictions_{timestamp}.pkl'
        predictions_data = {
            'results': all_test_results,
            'test_labels': test_labels.values,
            'timestamp': timestamp
        }
        
        with open(predictions_file, 'wb') as f:
            pickle.dump(predictions_data, f)
        print(f"Predictions saved to: {predictions_file}")
        
        # Summary statistics
        test_cccs = [r['test_ccc'] for r in all_test_results]
        test_maes = [r['test_mae'] for r in all_test_results]
        
        print(f"\n--- OVERALL STATISTICS ---")
        print(f"Mean Test CCC: {np.mean(test_cccs):.4f} ± {np.std(test_cccs):.4f}")
        print(f"Best Test CCC: {np.max(test_cccs):.4f}")
        print(f"Mean Test MAE: {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f}")
        print(f"Best Test MAE: {np.min(test_maes):.3f}")
        
    print(f"\n✅ Testing completed successfully!")

if __name__ == "__main__":
    main()
