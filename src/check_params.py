import json
import argparse
import os

def load_metadata(metadata_path):
    """Load and return metadata from a JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find metadata file at {metadata_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {metadata_path}")
        return None

def extract_hyperparameters(metadata):
    """Extract relevant hyperparameters from metadata."""
    if not metadata:
        return None
    
    # Initialize with default values in case some parameters are missing
    params = {
        'single_class_prototype_per_class': None,
        'proto_dim': None,
        'proto_time_len': None,
        'backbone': None,
        'label_set': None,
        'category': None
    }
    
    # Print the structure of the metadata to help debug
    print("\nMetadata structure:")
    print(json.dumps(metadata, indent=2)[:1000])  # Print first 1000 chars to avoid overwhelming output
    
    # Try to get parameters from metadata
    # First check if there's a model_config section
    if 'model_config' in metadata:
        config = metadata['model_config']
        params.update({
            'single_class_prototype_per_class': config.get('single_class_prototype_per_class'),
            'proto_dim': config.get('proto_dim'),
            'proto_time_len': config.get('proto_time_len'),
            'backbone': config.get('backbone'),
            'label_set': config.get('label_set'),
            'category': config.get('category')
        })
    
    # Check if parameters are at the root level
    for key in params:
        if params[key] is None and key in metadata:
            params[key] = metadata[key]
    
    # Check if parameters are in a 'args' or 'config' section
    for section in ['args', 'config', 'parameters']:
        if section in metadata:
            section_data = metadata[section]
            for key in params:
                if params[key] is None and key in section_data:
                    params[key] = section_data[key]
    
    # Check for parameters in prototype metadata
    if 'prototypes' in metadata:
        # Try to get category from first prototype
        first_proto = next(iter(metadata['prototypes'].values()), None)
        if first_proto and 'category' in first_proto:
            params['category'] = first_proto['category']
    
    return params

def print_hyperparameters(params, model_name):
    """Print hyperparameters in a formatted way."""
    print(f"\n=== Hyperparameters for {model_name} ===")
    print("Relevant parameters for fusion:")
    print(f"  single_class_prototype_per_class: {params['single_class_prototype_per_class']}")
    print(f"  proto_dim: {params['proto_dim']}")
    print(f"  proto_time_len: {params['proto_time_len']}")
    print(f"  backbone: {params['backbone']}")
    print(f"  label_set: {params['label_set']}")
    print(f"  category: {params['category']}")
    print("=" * 40)

def main():
    parser = argparse.ArgumentParser(description='Check metadata files for model hyperparameters')
    parser.add_argument('--metadata_paths', nargs='+', required=True,
                      help='Paths to metadata JSON files (space-separated)')
    parser.add_argument('--model_names', nargs='+', required=True,
                      help='Names of the models (space-separated, in same order as metadata_paths)')
    
    args = parser.parse_args()
    
    if len(args.metadata_paths) != len(args.model_names):
        print("Error: Number of metadata paths must match number of model names")
        return
    
    print("\nChecking metadata files for fusion hyperparameters...")
    print("=" * 40)
    
    for metadata_path, model_name in zip(args.metadata_paths, args.model_names):
        print(f"\nChecking {model_name}...")
        metadata = load_metadata(metadata_path)
        if metadata:
            params = extract_hyperparameters(metadata)
            if params:
                print_hyperparameters(params, model_name)
            else:
                print(f"Could not extract hyperparameters from {metadata_path}")
        print("-" * 40)

if __name__ == "__main__":
    main() 