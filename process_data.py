import argparse
import os
from utils.correlations_computation import compute_corrs
from utils.video_manipulation import compress, scale
from utils.feature_extraction import extract_features
from yaml import safe_load

def parse_arguments():
    # Create a parser and the required arguments
    parser = argparse.ArgumentParser(description="Process video datasets")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to a configuration file")

    return parser.parse_args()


def validate_dataset(path):
    # Directory check 
    if not os.path.isdir(path):
        raise ValueError(f"Dataset {path} is not a directory")
    
    # Structure and files check
    ls_dir = os.listdir(path)
    if ls_dir != ["test", "train"]:
        raise ValueError(f"Dataset {path} has wrong structure")

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp')
    for obj in ls_dir:
        obj_path = os.path.join(path, obj)
        if not os.path.isdir(obj_path):
            raise ValueError(f"Dataset {path} has wrong structure")

        dir_path = obj_path
        for file in os.listdir(dir_path):
            if file.is_file() and not file.suffix.lower() in valid_extensions:
                raise ValueError(f"Dataset {path} has wrong structure")

if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Load a config from arguments
    config = None
    with open(args.config, 'r') as file:
        config = safe_load(file)

    # Preprocess dataset
    dataset_path = config.get("dataset_path")
    validate_dataset(path)

    compression = config.get("compression")
    if compression["required"]:    
        compress(path, compression["values"])

    scaling = config.get("scaling")
    if scaling["required"]:    
        scale(path, scaling["values"])

    # Extract features
    extract_features(config)

    # Postprocess features
    compute_corrs(config)

    ...