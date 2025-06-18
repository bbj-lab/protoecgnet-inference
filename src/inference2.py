import torch
import numpy as np
import json
import random
import os
import argparse
import pandas as pd
from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D
from training_functions import seed_everything
from sklearn.metrics import roc_auc_score, f1_score

# 1D Rhythm 
PRETRAINED_WEIGHTS = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/1D_rhythm_classifier_01/last.ckpt"
METADATA_JSON = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/1D_rhythm_projection_01/1D_rhythm_projection_01_prototype_metadata.json"
MODEL_TYPE = "1D"  # Set to "1D" or "2D"
LABEL_SET = "1"    # Set to "1" for 1D, "3" for 2D partial/morph, "4" for 2D global
BACKBONE = "resnet1d18"  # Set to "resnet18" for 2D, and "resnet1d18" for 1D
PROTO_DIM = 512
PROTO_TIME_LEN = 32  # Use 3 for 2D partial/morph, 32 for 1D and 2D global
SINGLE_PPC = 5      # single_class_prototype_per_class, Use 5 for 1D, 18 for 2D partial/morph, 7 for 2D global
JOINT_PPB = 0       # joint_prototypes_per_border

# 2D Partial/Morph 
# PRETRAINED_WEIGHTS = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morph_projection_01/2D_morph_projection_01_projection.pth"
# METADATA_JSON = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morph_projection_01/2D_morph_projection_01_prototype_metadata.json"
# MODEL_TYPE = "2D"  # Set to "1D" or "2D"
# LABEL_SET = "3"    # Set to "1" for 1D, "3" for 2D partial/morph, "4" for 2D global
# BACKBONE = "resnet18"  # Set to "resnet18" for 2D, and "resnet1d18" for 1D
# PROTO_DIM = 512
# PROTO_TIME_LEN = 3  # Use 3 for 2D partial/morph, 32 for 1D and 2D global
# SINGLE_PPC = 5      # single_class_prototype_per_class, Use 5 for 1D, 18 for 2D partial/morph, 7 for 2D global
# JOINT_PPB = 0       # joint_prototypes_per_border

# 2D Global
# PRETRAINED_WEIGHTS = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_global_projection_01/2D_global_projection_01_projection.pth"
# METADATA_JSON = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_global_projection_01/2D_global_projection_01_prototype_metadata.json"
# MODEL_TYPE = "2D"  # Set to "1D" or "2D"
# LABEL_SET = "4"    # Set to "1" for 1D, "3" for 2D partial/morph, "4" for 2D global
# BACKBONE = "resnet18"  # Set to "resnet18" for 2D, and "resnet1d18" for 1D
# PROTO_DIM = 512
# PROTO_TIME_LEN = 32  # Use 3 for 2D partial/morph, 32 for 1D and 2D global
# SINGLE_PPC = 5      # single_class_prototype_per_class, Use 5 for 1D, 18 for 2D partial/morph, 7 for 2D global
# JOINT_PPB = 0       # joint_prototypes_per_border

# --- Utility Functions ---
def list_test_sample_ids(test_loader, max_samples=20):
    """
    Print the available test sample IDs (up to max_samples for brevity).
    """
    print("\n--- Available Test Sample IDs (showing up to {}): ---".format(max_samples))
    ids = []
    count = 0
    for _, _, id_batch in test_loader:
        for ecg_id in id_batch:
            ids.append(int(ecg_id.item()))
            count += 1
            if count >= max_samples:
                break
        if count >= max_samples:
            break
    print(ids)
    if count == max_samples:
        print("... (truncated, more samples available)")
    return ids

def get_test_sample_by_id(test_loader, target_id):
    for X_batch, y_batch, id_batch in test_loader:
        for x, y, ecg_id in zip(X_batch, y_batch, id_batch):
            if int(ecg_id.item()) == target_id:
                return x, y, ecg_id.item()
    raise ValueError(f"ECG ID {target_id} not found in the test set.")

def get_random_test_sample(test_loader):
    test_iter = iter(test_loader)
    batch = next(test_iter)
    X_batch, y_batch, sample_ids = batch
    idx = random.randint(0, len(X_batch) - 1)
    return X_batch[idx], y_batch[idx], sample_ids[idx]

def load_model_weights(model, weights_path):
    if weights_path.endswith('.pth'):
        state_dict = torch.load(weights_path, map_location='cpu')
        # If saved with DataParallel, keys may have 'module.' prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace('module.', '')] = v
            state_dict = new_state_dict
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path} (pth format)")
    elif weights_path.endswith('.ckpt'):
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        # PyTorch Lightning saves weights under 'state_dict'
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        # Remove 'model.' or 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[len('model.'):]] = v
            elif k.startswith('net.'):
                new_state_dict[k[len('net.'):]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded weights from {weights_path} (ckpt format)")
    else:
        raise ValueError(f"Unsupported weights file format: {weights_path}")
    return model

# --- Main Granular Prototype Display ---
def display_granular_prototypes(probabilities, similarity_scores, label_names, prototype_metadata, threshold=0.5, top_n=5):
    print("\n--- Granular Prototype Analysis ---")
    predicted_classes_indices = np.where(probabilities >= threshold)[0]
    if len(predicted_classes_indices) == 0:
        print("No classes predicted above the threshold.")
        return
    for class_idx in predicted_classes_indices: 
        class_name = label_names[class_idx]
        print(f"\n--- Top Prototypes for Predicted Class: {class_name} (Prob: {probabilities[class_idx]:.4f}) ---")
        relevant_prototypes = []
        for proto_id_str, meta in prototype_metadata.items():
            if meta.get('prototype_class') == class_name:
                proto_idx = int(proto_id_str)
                # Get true label names from true_labels vector
                true_labels = meta.get('true_labels', [])
                true_label_names = [label_names[i] for i, v in enumerate(true_labels) if v == 1.0]
                relevant_prototypes.append({
                    'proto_id': proto_idx,
                    'score': similarity_scores[proto_idx],
                    'prototype_class': meta.get('prototype_class', 'N/A'),
                    'true_label_names': true_label_names
                })
        if not relevant_prototypes:
            print(f"  No prototypes found directly associated with '{class_name}'.")
            continue
        relevant_prototypes.sort(key=lambda x: x['score'], reverse=True)
        for i, p_info in enumerate(relevant_prototypes[:top_n]):
            print(f"  {i+1}. Prototype {p_info['proto_id']} (Prototype Class: {p_info['prototype_class']}, True Labels: {', '.join(p_info['true_label_names'])}) – Score: {p_info['score']:.4f}")

def print_metadata_head(metadata_json_path, num_lines=50):
    print(f"\n--- First {num_lines} lines of {metadata_json_path} ---")
    with open(metadata_json_path, 'r') as f:
        for i, line in enumerate(f):
            print(line.rstrip())
            if i + 1 >= num_lines:
                break

def sanity_check_model_performance(model, test_loader, label_names, device):
    print("\n--- Running Sanity Check: Model Performance on Test Set ---")
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            logits, _, _ = model(X_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.append(y_batch.cpu().numpy())
            y_pred.append(probs)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    try:
        macro_auc = roc_auc_score(y_true, y_pred, average='macro')
    except Exception as e:
        macro_auc = f"Error: {e}"
    f1 = f1_score(y_true, (y_pred > 0.5).astype(int), average='micro')
    print(f"Macro AUC: {macro_auc}")
    print(f"F1 (micro): {f1}")

def check_label_mismatch(label_names, label_set, csv_path="scp_statementsRegrouped2.csv"):
    print("\n--- Checking for Label Mismatch ---")
    category = int(label_set)
    df = pd.read_csv(csv_path)
    csv_labels = df[df['prototype_category'] == category].iloc[:, 0].tolist()
    print(f"\n--- Labels from CSV for category {category} ---")
    for i, name in enumerate(csv_labels):
        print(f"{i}: {name}")
    print("\n--- label_names from inference ---")
    for i, name in enumerate(label_names):
        print(f"{i}: {name}")
    if csv_labels == label_names:
        print("\nLabel order and content match exactly!")
    else:
        print("\nWARNING: Label order or content does NOT match!")
        for i, (csv_label, infer_label) in enumerate(zip(csv_labels, label_names)):
            if csv_label != infer_label:
                print(f"Mismatch at index {i}: CSV='{csv_label}' vs Inference='{infer_label}'")
        if len(csv_labels) != len(label_names):
            print(f"CSV label count: {len(csv_labels)}, Inference label count: {len(label_names)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and prototype analysis.")
    parser.add_argument('--show-prototypes', action='store_true', help='Show Top Prototypes and Similarity Scores for Each Class')
    parser.add_argument('--sanity-check', action='store_true', help='Run model performance sanity check on test set')
    parser.add_argument('--show-metadata', action='store_true', help='Print first 50 lines of metadata JSON')
    parser.add_argument('--check-labels', action='store_true', help='Check for label mismatch between CSV and inference label_names')
    args = parser.parse_args()

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Label Mappings ---
    label_mappings = load_label_mappings(custom_groups=True, prototype_category=int(LABEL_SET))
    label_names = label_mappings["custom"]
    num_classes = len(label_names)

    # --- Load Prototype Metadata ---
    with open(METADATA_JSON, 'r') as f:
        prototype_metadata = json.load(f)

    # --- Load Model ---
    if MODEL_TYPE == "1D":
        model = ProtoECGNet1D(
            num_classes=num_classes,
            single_class_prototype_per_class=SINGLE_PPC,
            joint_prototypes_per_border=JOINT_PPB,
            proto_dim=PROTO_DIM,
            backbone=BACKBONE,
            prototype_activation_function='arc',
            latent_space_type='arc',
            add_on_layers_type='linear',
            class_specific=True,
            last_layer_connection_weight=1.0,
            m=0.05,
            dropout=0.0,
            custom_groups=True,
            label_set=LABEL_SET,
            pretrained_weights=None  # Don't load here
        ).to(device)
    elif MODEL_TYPE == "2D":
        model = ProtoECGNet2D(
            num_classes=num_classes,
            single_class_prototype_per_class=SINGLE_PPC,
            joint_prototypes_per_border=JOINT_PPB,
            proto_dim=PROTO_DIM,
            proto_time_len=PROTO_TIME_LEN,
            backbone=BACKBONE,
            prototype_activation_function='arc',
            latent_space_type='arc',
            add_on_layers_type='linear',
            class_specific=True,
            last_layer_connection_weight=1.0,
            m=0.05,
            dropout=0.0,
            custom_groups=True,
            label_set=LABEL_SET,
            pretrained_weights=None  # Don't load here
        ).to(device)
    else:
        raise ValueError("MODEL_TYPE must be '1D' or '2D'")
    model = load_model_weights(model, PRETRAINED_WEIGHTS)
    model.eval()

    # --- Get Test Data Loader ---
    test_loader_params = {
        "batch_size": 1,
        "mode": MODEL_TYPE,
        "sampling_rate": 100,
        "label_set": LABEL_SET,
        "work_num": 0,
        "return_sample_ids": True,
        "custom_groups": True,
        "standardize": False,
        "remove_baseline": True,
    }
    _, _, test_loader, _ = get_dataloaders(**test_loader_params)

    if args.show_metadata:
        print_metadata_head(METADATA_JSON, num_lines=50)

    if args.sanity_check:
        sanity_check_model_performance(model, test_loader, label_names, device)

    # --- Select a Sample for Inference ---
    target_ecg_ids = list_test_sample_ids(test_loader, max_samples=20)

    # --- Run Inference ---
    for target_ecg_id in target_ecg_ids:
        if target_ecg_id is not None:
            try:
                X_sample, y_sample, ecg_id = get_test_sample_by_id(test_loader, target_ecg_id)
            except ValueError as e:
                print(e)
                continue
        else:
            X_sample, y_sample, ecg_id = get_random_test_sample(test_loader)
        print(f"\n--- Running Inference for ECG ID: {ecg_id} ---")
        X_sample = X_sample.unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, similarity_scores = model(X_sample)
            probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
            similarity_scores = similarity_scores.cpu().numpy().flatten()

        # --- Display Overall Predictions ---
        print("\n--- Model Overall Predictions (Top 10) ---")
        sorted_predictions = sorted(zip(label_names, probabilities), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_predictions[:10]:
            print(f"  {label}: {prob:.4f}")

        # --- Display True Labels (if available) ---
        if y_sample is not None:
            true_labels_indices = np.where(y_sample.cpu().numpy() == 1)[0]
            true_label_names = [label_names[i] for i in true_labels_indices]
            print(f"\n--- True Labels for ECG ID {ecg_id} ---")
            if true_label_names:
                print(f"  {', '.join(true_label_names)}")
            else:
                print("  No true labels (all zeros) for this sample in the loaded test set.")

        # --- Display Granular Prototypes for Predicted Classes ---
        display_granular_prototypes(
            probabilities,
            similarity_scores,
            label_names,
            prototype_metadata,
            threshold=0.5
        )

        # --- Display Prototypes and Similarity Scores for Each Class ---
        if args.show_prototypes:
            top_n = 5  # Number of top prototypes to show per class
            print("\n--- Top Prototypes and Similarity Scores for Each Class (regardless of probability) ---")
            for class_idx, class_name in enumerate(label_names):
                relevant_prototypes = []
                for proto_id_str, meta in prototype_metadata.items():
                    # Match using prototype_class
                    if meta.get('prototype_class') == class_name:
                        proto_idx = int(proto_id_str)
                        # Get true label names from true_labels vector
                        true_labels = meta.get('true_labels', [])
                        true_label_names = [label_names[i] for i, v in enumerate(true_labels) if v == 1.0]
                        relevant_prototypes.append({
                            'proto_id': proto_idx,
                            'score': similarity_scores[proto_idx],
                            'prototype_class': meta.get('prototype_class', 'N/A'),
                            'true_label_names': true_label_names
                        })
                if not relevant_prototypes:
                    print(f"  No prototypes found for class '{class_name}'.")
                    continue
                relevant_prototypes.sort(key=lambda x: x['score'], reverse=True)
                print(f"\nClass: {class_name}")
                for i, p_info in enumerate(relevant_prototypes[:top_n]):
                    print(f"  {i+1}. Prototype {p_info['proto_id']} (Prototype Class: {p_info['prototype_class']}, True Labels: {', '.join(p_info['true_label_names'])}) – Score: {p_info['score']:.4f}")

    if args.check_labels:
        check_label_mismatch(label_names, LABEL_SET, csv_path="scp_statementsRegrouped2.csv") 