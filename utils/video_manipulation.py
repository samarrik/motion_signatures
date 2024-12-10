import os
import subprocess
from tqdm import tqdm

def compress_video(origin, destination, value):
    ffmpeg_command = [
        "ffmpeg",
        "-i", origin,
        "-vcodec", "libx264",
        "-crf", str(value),  # Compression level
        "-preset", "medium",
        destination
    ]
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFprobe error:\n{result.stderr}")
        raise RuntimeError(f"Compression failed for {origin}.")


def compress(path: str, values: list):
    subdatasets = ["train", "test"]
    for subdataset in  tqdm(subdatasets, desc="Compressing subdatasets"):
        for value in values:
            # Subdataset
            subdataset_path = os.path.join(path, subdataset)
            if not os.path.isdir(subdataset_path):
                continue

            # Compressed version of a subdataset
            compressed_subdataset = f"{subdataset}_c{value}"
            compressed_path = os.path.join(path, compressed_subdataset)
            os.makedirs(compressed_path, exist_ok=True)

            for video in os.listdir(subdataset_path):
                origin = os.path.join(subdataset_path, video)
                destination = os.path.join(compressed_path, video)
                compress_video(origin, destination, value)


def scale_video(origin, destination, value):    
    # Get original dimensions
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        origin
    ]
    result = subprocess.run(ffprobe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"FFprobe error:\n{result.stderr}")
        raise RuntimeError(f"Extraction of dimensions failed for {origin}.")
    width, height = map(int, result.stdout.strip().split(','))

    # Calculate new dimensions
    new_width = int(width * value) // 2 * 2 # Round down to the nearest even number
    new_height = int(height * value) // 2 * 2

    # Resize video
    ffmpeg_command = [
        "ffmpeg",
        "-i", origin,
        "-vf", f"scale={new_width}:{new_height}",
        "-c:v", "libx264",
        "-preset", "medium",
        destination
    ]
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"FFprobe error:\n{result.stderr}")
        raise RuntimeError(f"Scaling failed for {origin}.")

def scale(path: str, values: list):
    subdatasets = ["train", "test"]
    for subdataset in  tqdm(subdatasets, desc="Scaling subdatasets"):
        for value in values:
            # Subdataset
            subdataset_path = os.path.join(path, subdataset)
            if not os.path.isdir(subdataset_path):
                continue

            # Scaled version of a subdataset
            scaled_subdataset = f"{subdataset}_s{value}"
            scaled_path = os.path.join(path, scaled_subdataset)
            os.makedirs(scaled_path, exist_ok=True)

            for video in os.listdir(subdataset_path):
                origin = os.path.join(subdataset_path, video)

                destination = os.path.join(scaled_path, video)
                scale_video(origin, destination, value)
