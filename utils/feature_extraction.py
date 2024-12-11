from feat import Detector
from feat.utils.io import video_to_tensor
import os
from tqdm import tqdm

def extract_features(input_path, output_path, extractors):
    # Setup extractors
    extractors_objects = {}
    for extractor, setup in extractors.items():
        if extractor == "py-feat":
            extractors_objects["py-feat"] = Detector(device="cuda")
        ... # Other extractors

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    files = [os.path.join(input_path, f) for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]

    for file in tqdm(files, desc="Processing Videos"):
        video_name = os.path.splitext(os.path.basename(file))[0]
        features_path = os.path.join(output_path, f"{video_name}.csv")

        detector = extractors_objects.get("py-feat")

        # # Perform feature extraction (server)
        video_tensor = video_to_tensor(file)
        detector.detect(
            video_tensor,
            data_type="tensor",
            face_detection_threshold=0.85,
            num_workers=10,
            batch_size=500,
            save=features_path,
            progress_bar=True,
        )