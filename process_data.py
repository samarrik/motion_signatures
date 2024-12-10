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
    if ["test", "train"] in ls_dir:
        raise ValueError(f"Dataset {path} has wrong structure")

    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.3gp')
    for dir in ["test", "train"]:
        dir_path = os.path.join(path, dir)
        if not os.path.isdir(dir_path):
            raise ValueError(f"Dataset {path} has wrong structure")

        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and not os.path.splitext(file)[-1].lower() in valid_extensions:
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
    validate_dataset(dataset_path)

    compression = config.get("compression")
    if compression["required"]:    
        compress(dataset_path, compression["values"])

    scaling = config.get("scaling")
    if scaling["required"]:    
        scale(dataset_path, scaling["values"])

    # Extract features
    # ! As feature extraction is a highly expensive task computationally, it is performed
    # ! on subdatasets, NOT the whole dataset at once
    extraction = config.get("extraction")
    if extraction["required"]:
        to_extract_dir = extraction["to_extract_dir"]
        extracted_path = extraction["extracted_path"]
        extractors = extraction["extractors"]

        subdataset_path = os.path.join(dataset_path, to_extract_dir)
        extracted_subdataset_path = os.path.join(extracted_path, to_extract_dir)
        extract_features(subdataset_path, extracted_subdataset_path, extractors)

    # Postprocess features
    # correlations_path = config.get("correlations_path")
    # compute_corrs(extracted_path, correlations_path, config)

    ...