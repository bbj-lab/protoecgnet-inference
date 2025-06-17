import torch
import numpy as np
import pandas as pd
import json
import os
import argparse
import random # Needed for get_random_test_sample

from ecg_utils import get_dataloaders, load_label_mappings
from proto_models1D import ProtoECGNet1D
from proto_models2D import ProtoECGNet2D
from fusion import FusionProtoClassifier
from training_functions import seed_everything # Assuming this is available

# --- Model Loading Function ---
def load_fusion_model(
    num_classes_fusion: int,
    # Parameters for 1D model (Category 1)
    backbone_1d: str, proto_dim_1d: int, single_ppc_1d: int, joint_ppb_1d: int,
    # Parameters for 2D partial model (Category 3)
    backbone_2d_partial: str, proto_dim_2d_partial: int, proto_time_len_2d_partial: int, single_ppc_2d_partial: int, joint_ppb_2d_partial: int,
    # Parameters for 2D global model (Category 4)
    backbone_2d_global: str, proto_dim_2d_global: int, proto_time_len_2d_global: int, single_ppc_2d_global: int, joint_ppb_2d_global: int,
    # Common parameters (should typically be 'arc' for paper results)
    prototype_activation_function: str = 'arc', latent_space_type: str = 'arc',
    add_on_layers_type: str = 'linear', class_specific: bool = True, last_layer_connection_weight: float = 1.0,
    m: float = 0.05, dropout: float = 0.0, custom_groups: bool = True, device: torch.device = torch.device('cpu')
):
    """
    Loads and initializes the three branch models and the fusion classifier.
    """
    print("Loading pretrained 1D and 2D models for fusion...")

    # Determine num_classes for individual branches from label mappings
    num_classes_1d = len(load_label_mappings(custom_groups=custom_groups, prototype_category=1)["custom"])
    num_classes_2d_partial = len(load_label_mappings(custom_groups=custom_groups, prototype_category=3)["custom"])
    num_classes_2d_global = len(load_label_mappings(custom_groups=custom_groups, prototype_category=4)["custom"])

    # Load 1D model
    print(f"Creating category 1 model with backbone {backbone_1d}...")
    model_1d = ProtoECGNet1D(
        num_classes=num_classes_1d,
        single_class_prototype_per_class=single_ppc_1d,
        joint_prototypes_per_border=joint_ppb_1d,
        proto_dim=proto_dim_1d,
        backbone=backbone_1d,
        prototype_activation_function=prototype_activation_function,
        latent_space_type=latent_space_type,
        add_on_layers_type=add_on_layers_type,
        class_specific=class_specific,
        last_layer_connection_weight=last_layer_connection_weight,
        m=m,
        dropout=dropout,
        custom_groups=custom_groups,
        label_set="1", # Important for loading correct co-occurrence matrix
        pretrained_weights=PRETRAINED_WEIGHTS_1D
    ).to(device)

    # Load 2D partial model (Category 3)
    print(f"Creating category 3 model with backbone {backbone_2d_partial}...")
    model_2d_partial = ProtoECGNet2D(
        num_classes=num_classes_2d_partial,
        single_class_prototype_per_class=single_ppc_2d_partial,
        joint_prototypes_per_border=joint_ppb_2d_partial,
        proto_dim=proto_dim_2d_partial,
        proto_time_len=proto_time_len_2d_partial,
        backbone=backbone_2d_partial,
        prototype_activation_function=prototype_activation_function,
        latent_space_type=latent_space_type,
        add_on_layers_type=add_on_layers_type,
        class_specific=class_specific,
        last_layer_connection_weight=last_layer_connection_weight,
        m=m,
        dropout=dropout,
        custom_groups=custom_groups,
        label_set="3", # Important for loading correct co-occurrence matrix
        pretrained_weights=PRETRAINED_WEIGHTS_2D_PARTIAL
    ).to(device)

    # Load 2D global model (Category 4)
    print(f"Creating category 4 model with backbone {backbone_2d_global}...")
    model_2d_global = ProtoECGNet2D(
        num_classes=num_classes_2d_global,
        single_class_prototype_per_class=single_ppc_2d_global,
        joint_prototypes_per_border=joint_ppb_2d_global,
        proto_dim=proto_dim_2d_global,
        proto_time_len=proto_time_len_2d_global,
        backbone=backbone_2d_global,
        prototype_activation_function=prototype_activation_function,
        latent_space_type=latent_space_type,
        add_on_layers_type=add_on_layers_type,
        class_specific=class_specific,
        last_layer_connection_weight=last_layer_connection_weight,
        m=m,
        dropout=dropout,
        custom_groups=custom_groups,
        label_set="4", # Important for loading correct co-occurrence matrix
        pretrained_weights=PRETRAINED_WEIGHTS_2D_GLOBAL
    ).to(device)

    # Set all submodels to evaluation mode and freeze parameters
    for m in [model_1d, model_2d_partial, model_2d_global]:
        m.eval()
        for param in m.parameters():
            param.requires_grad = False

    # Initialize and load the FusionProtoClassifier
    fusion_model = FusionProtoClassifier(model_1d, model_2d_partial, model_2d_global, num_classes=num_classes_fusion).to(device)
    fusion_model.eval()
    for param in fusion_model.parameters(): # Freeze fusion classifier head for inference
        param.requires_grad = False

    print("Fusion classifier initialized and loaded.")
    return fusion_model, model_1d, model_2d_partial, model_2d_global

# --- Fused Inference Function ---
def run_fused_inference(fusion_model: FusionProtoClassifier, x_sample: torch.Tensor):
    """
    Runs inference on a single ECG sample using the fused model and extracts
    combined prototype similarity scores.
    Returns: probabilities (np.ndarray), combined_similarity_scores (np.ndarray)
    """
    x_sample = x_sample.unsqueeze(0).to(fusion_model.device) # Add batch dimension and move to device
    
    with torch.no_grad():
        # Ensure input shape is correct for 1D branch (N, 12, 1000)
        # Assuming x_sample is initially [1, 1, 12, 1000]
        x1d = x_sample.squeeze(1) 
        
        _, _, sim1d = fusion_model.model1d(x1d)
        _, _, sim2d_partial = fusion_model.model2d_partial(x_sample)
        _, _, sim2d_global = fusion_model.model2d_global(x_sample)

        sims = torch.cat([sim1d, sim2d_partial, sim2d_global], dim=1) # Concatenate all similarity scores
        logits = fusion_model.classifier(sims) # Get final fused logits

    probabilities = torch.sigmoid(logits).cpu().numpy().flatten()
    combined_similarity_scores = sims.cpu().numpy().flatten()

    return probabilities, combined_similarity_scores

# --- Sample Selection Functions (Adapted from inference.ipynb) ---
def get_test_sample_by_id(test_loader, target_id):
    """
    Retrieve a specific ECG sample from the test_loader by ecg_id.
    Returns: (sample_x, sample_y, sample_id) or raises ValueError if not found.
    """
    for X_batch, y_batch, id_batch in test_loader:
        for x, y, ecg_id in zip(X_batch, y_batch, id_batch):
            if int(ecg_id.item()) == target_id:
                return x, y, ecg_id.item()
    raise ValueError(f"ECG ID {target_id} not found in the test set.")

def get_random_test_sample(test_loader):
    """Returns a random test ECG from the DataLoader."""
    test_iter = iter(test_loader)
    batch = next(test_iter)  # Get the first batch
    X_batch, y_batch, sample_ids = batch
    idx = random.randint(0, len(X_batch) - 1)  # Select a random sample
    return X_batch[idx], y_batch[idx], sample_ids[idx]

# --- Granular Prototype Display Function ---
def display_granular_prototypes(
    probabilities: np.ndarray,
    combined_similarity_scores: np.ndarray,
    fusion_label_names: list,
    model_1d_prototypes_metadata: dict,
    model_2d_partial_prototypes_metadata: dict,
    model_2d_global_prototypes_metadata: dict,
    threshold: float = 0.5,
    top_n: int = 5
):
    """
    Displays the most activated prototypes for each predicted class.
    
    NOTE: This implementation assumes that the 'labels' in your prototype_metadata.json
    (which are category-specific SCP codes like 'LMI', 'TAB_') can be directly
    matched as strings against the 'fusion_label_names' (the 71 global labels).
    If your global labels are different, a more complex mapping might be needed here.
    """
    print("\n--- Granular Prototype Analysis ---")

    # Get predicted classes based on threshold
    predicted_classes_indices = np.where(probabilities >= threshold)[0]
    
    if len(predicted_classes_indices) == 0:
        print("No classes predicted above the threshold.")
        return

    # Combine all prototype metadata for easy lookup, adjusting indices for concatenation
    all_prototypes_info = {}
    current_proto_idx_offset = 0

    # 1D Prototypes
    for proto_id_str, metadata in model_1d_prototypes_metadata.items():
        original_idx = int(proto_id_str) # JSON keys are strings
        all_prototypes_info[current_proto_idx_offset + original_idx] = {
            'labels': metadata['labels'], # These are the category 1 SCP codes
            'category': metadata['category'],
            'original_branch_proto_id': original_idx,
            'branch': '1D'
        }
    current_proto_idx_offset += len(model_1d_prototypes_metadata)

    # 2D Partial Prototypes
    for proto_id_str, metadata in model_2d_partial_prototypes_metadata.items():
        original_idx = int(proto_id_str)
        all_prototypes_info[current_proto_idx_offset + original_idx] = {
            'labels': metadata['labels'], # These are the category 3 SCP codes
            'category': metadata['category'],
            'original_branch_proto_id': original_idx,
            'branch': '2D_Partial'
        }
    current_proto_idx_offset += len(model_2d_partial_prototypes_metadata)

    # 2D Global Prototypes
    for proto_id_str, metadata in model_2d_global_prototypes_metadata.items():
        original_idx = int(proto_id_str)
        all_prototypes_info[current_proto_idx_offset + original_idx] = {
            'labels': metadata['labels'], # These are the category 4 SCP codes
            'category': metadata['category'],
            'original_branch_proto_id': original_idx,
            'branch': '2D_Global'
        }
    
    # Iterate through each predicted class (from the 71 global labels)
    for class_idx in predicted_classes_indices:
        class_name = fusion_label_names[class_idx]
        print(f"\n--- Top Prototypes for Predicted Class: {class_name} (Prob: {probabilities[class_idx]:.4f}) ---")
        
        relevant_prototypes = []
        for proto_combined_idx, proto_info in all_prototypes_info.items():
            # Check if any of the prototype's _category-specific_ labels match the current _global predicted class name_
            if class_name in proto_info['labels']: 
                relevant_prototypes.append({
                    'proto_id': proto_combined_idx,
                    'score': combined_similarity_scores[proto_combined_idx],
                    'branch': proto_info['branch'],
                    'original_labels': proto_info['labels']
                })
        
        if not relevant_prototypes:
            print(f"  No specific prototypes found directly associated with '{class_name}' via simple label matching.")
            continue

        # Sort by score and take top_n
        relevant_prototypes.sort(key=lambda x: x['score'], reverse=True)
        for i, p_info in enumerate(relevant_prototypes[:top_n]):
            print(f"  {i+1}. Prototype {p_info['proto_id']} (Branch: {p_info['branch']}, Original Labels: {', '.join(p_info['original_labels'])}) – Score: {p_info['score']:.4f}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- IMPORTANT: UPDATE THESE PATHS ---
    # These should point to the .pth and .json files generated by the 'projection' stage
    # for each of your 1D, 2D_Partial (Category 3), and 2D_Global (Category 4) models.
    #
    # Example for `2D_morphology_projection1` job from previous discussion:
    # PRETRAINED_WEIGHTS_2D_PARTIAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morphology_projection1/2D_morphology_projection1_projection.pth"
    # METADATA_JSON_2D_PARTIAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morphology_projection1/2D_morphology_projection1_prototype_metadata.json"
    #
    # You will need to have trained and projected each branch individually to get these files.
    

    PRETRAINED_WEIGHTS_1D = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/1D_rhythm_projection2/1D_rhythm_projection2_projection.pth"
    METADATA_JSON_1D = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/1D_rhythm_projection2/1D_rhythm_projection2_prototype_metadata.json"

    PRETRAINED_WEIGHTS_2D_PARTIAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morphology_projection1/2D_morphology_projection1_projection.pth"
    METADATA_JSON_2D_PARTIAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_morphology_projection1/2D_morphology_projection1_prototype_metadata.json"

    PRETRAINED_WEIGHTS_2D_GLOBAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_global_projection1/2D_global_projection1_projection.pth"
    METADATA_JSON_2D_GLOBAL = "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints/2D_global_projection1/2D_global_projection1_prototype_metadata.json"

    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Label Mappings ---
    # Need the full 71 labels for fusion model output interpretation
    fusion_label_mappings = load_label_mappings(custom_groups=False, prototype_category=None)
    fusion_label_names = fusion_label_mappings["all"]
    num_fusion_classes = len(fusion_label_names)

    # --- Load Prototype Metadata (for granular display) ---
    try:
        with open(METADATA_JSON_1D, 'r') as f:
            model_1d_prototypes_metadata = json.load(f)
        with open(METADATA_JSON_2D_PARTIAL, 'r') as f:
            model_2d_partial_prototypes_metadata = json.load(f)
        with open(METADATA_JSON_2D_GLOBAL, 'r') as f:
            model_2d_global_prototypes_metadata = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading prototype metadata JSON: {e}.")
        print("Please ensure the METADATA_JSON_1D, METADATA_JSON_2D_PARTIAL, and METADATA_JSON_2D_GLOBAL paths are correct and the files exist.")
        exit()

    # --- Model Parameters (Adjust these based on your trained models' configurations) ---
    # These parameters are crucial and MUST match the configuration used when you trained
    # and projected each individual branch model.
    #
    # Example values, replace with your actual values from your training scripts:
    common_params = {
        "prototype_activation_function": 'arc', # Based on your main.py default
        "latent_space_type": 'arc', # Based on your main.py default
        "add_on_layers_type": 'linear', # Common default
        "class_specific": True,
        "last_layer_connection_weight": 1.0,
        "m": 0.05,
        "dropout": 0.0,
        "custom_groups": True, # Individual branches typically trained with custom groups
        "device": device
    }

    # 1D Model specific parameters (Category 1)
    model_1d_params = {
        "backbone": "resnet1d18",
        "proto_dim": 512,
        "proto_time_len": 32, 
        "single_ppc": 5, # Example: Single-class prototypes per class for 1D
        "joint_ppb": 0 # Example: Joint prototypes per border for 1D
    }

    # 2D Partial Model specific parameters (Category 3)
    model_2d_partial_params = {
        "backbone": "resnet18",
        "proto_dim": 512,
        "proto_time_len": 3, # Specific for partial prototypes
        "single_ppc": 18, # Example: Single-class prototypes per class for 2D partial
        "joint_ppb": 0 # Example: Joint prototypes per border for 2D partial
    }

    # 2D Global Model specific parameters (Category 4)
    model_2d_global_params = {
        "backbone": "resnet18",
        "proto_dim": 512,
        "proto_time_len": 32, 
        "single_ppc": 3, # Example: Single-class prototypes per class for 2D global
        "joint_ppb": 0 # Example: Joint prototypes per border for 2D global
    }

    # --- Load Fusion Model ---
    # This will load each branch model and then the fusion classifier
    fusion_model, model_1d, model_2d_partial, model_2d_global = load_fusion_model(
        num_classes_fusion=num_fusion_classes,
        backbone_1d=model_1d_params["backbone"], proto_dim_1d=model_1d_params["proto_dim"],
        single_ppc_1d=model_1d_params["single_ppc"], joint_ppb_1d=model_1d_params["joint_ppb"],
        backbone_2d_partial=model_2d_partial_params["backbone"], proto_dim_2d_partial=model_2d_partial_params["proto_dim"],
        proto_time_len_2d_partial=model_2d_partial_params["proto_time_len"], single_ppc_2d_partial=model_2d_partial_params["single_ppc"],
        joint_ppb_2d_partial=model_2d_partial_params["joint_ppb"],
        backbone_2d_global=model_2d_global_params["backbone"], proto_dim_2d_global=model_2d_global_params["proto_dim"],
        proto_time_len_2d_global=model_2d_global_params["proto_time_len"], single_ppc_2d_global=model_2d_global_params["single_ppc"],
        joint_ppb_2d_global=model_2d_global_params["joint_ppb"],
        **common_params
    )

    # --- Get Test Data Loader ---
    # Parameters for get_dataloaders (adjust as needed for your specific dataset setup)
    test_loader_params = {
        "batch_size": 1, # Use batch size 1 for single sample inference
        "mode": "2D", # Fusion model expects 2D input [N, 1, 12, 1000]
        "sampling_rate": 100, # Match your dataset's sampling rate
        "label_set": "all", # Load with all labels for test set (even if not used for prediction directly)
        "work_num": 0, # Use 0 workers for single sample loading for simplicity
        "return_sample_ids": True, # Essential for getting ECG IDs
        "custom_groups": False, # Test loader typically uses all labels if fusion is on 71 classes
        "standardize": False, # Match your preprocessing
        "remove_baseline": True, # Match your preprocessing
    }
    _, _, test_loader, _ = get_dataloaders(**test_loader_params)

    # --- Select a Sample for Inference ---
    # You can specify an ECG ID or set to None to get a random sample
    target_ecg_id = None # <-- REPLACE with an actual ECG ID from your test set, or set to None for random
    
    if target_ecg_id is not None:
        try:
            X_sample, y_sample, ecg_id = get_test_sample_by_id(test_loader, target_ecg_id)
        except ValueError as e:
            print(e)
            exit()
    else:
        X_sample, y_sample, ecg_id = get_random_test_sample(test_loader)
    
    print(f"\n--- Running Inference for ECG ID: {ecg_id} ---")

    # --- Run Fused Inference ---
    probabilities, combined_similarity_scores = run_fused_inference(fusion_model, X_sample)

    # --- Display Overall Fused Predictions ---
    print("\n--- Fused Model Overall Predictions (Top 10) ---")
    sorted_predictions = sorted(zip(fusion_label_names, probabilities), key=lambda x: x[1], reverse=True)
    for label, prob in sorted_predictions[:10]: # Display top 10 predicted classes
        print(f"  {label}: {prob:.4f}")

    # --- Display True Labels (if available in test_loader) ---
    if y_sample is not None:
        true_labels_indices = np.where(y_sample.cpu().numpy() == 1)[0]
        true_label_names = [fusion_label_names[i] for i in true_labels_indices]
        print(f"\n--- True Labels for ECG ID {ecg_id} ---")
        if true_label_names:
            print(f"  {', '.join(true_label_names)}")
        else:
            print("  No true labels (all zeros) for this sample in the loaded test set.")

    # --- Display Granular Prototypes for Predicted Classes ---
    # This is the new functionality for detailed prototype analysis
    display_granular_prototypes(
        probabilities,
        combined_similarity_scores,
        fusion_label_names, # List of all 71 global label names
        model_1d_prototypes_metadata,
        model_2d_partial_prototypes_metadata,
        model_2d_global_prototypes_metadata,
        threshold=0.5 # Confidence threshold for a class to be considered "predicted"
    )
