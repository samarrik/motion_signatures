import os
import pandas as pd
from tqdm import tqdm

# Constants
CONSTANT_VIDEO_SIZE_S = 30

def compute_corrs(input_path, output_path, extractors, clip_configs):
    # Determine the selected features from the configured extractors
    selected_features = []
    for extractor, features in extractors.items():
        selected_features.extend(features)

    subdatasets = os.listdir(input_path)
    subdatasets = [d for d in subdatasets if os.path.isdir(os.path.join(path, d))]
    for subdataset in tqdm(subdatasets, desc="Computing Correlations for subdatasets"):
        # List all CSV files
        extracted_features_files = []
        subdataset_path = os.path.join(input_path, subdataset)
        for file in os.listdir(subdataset_path):
            file_path = os.path.join(subdataset_path, file)
            extracted_features_files.append(file_path)

        # Process each clip configuration
        for _, clips_config in clip_configs.items():
            length_clip = clips_config["length"]
            overlap_clip = clips_config["overlap"]

            # Prepare a DataFrame to store all correlations for this configuration
            correlations = pd.DataFrame()

            for extracted_file in extracted_features_files:
                video_basename = os.path.splitext(os.path.basename(extracted_file))[0]
                df_video = pd.read_csv(extracted_file)

                # Select only the required features
                df_video = df_video[selected_features]

                frame_cnt_video = len(df_video)
                fps_video = frame_cnt_video / CONSTANT_VIDEO_SIZE_S
                frame_cnt_clip = int(round(fps_video * length_clip))
                frame_int_clip = int(round(fps_video * (length_clip - overlap_clip)))

                # Generate clip start frames
                clip_starts = []
                current_start_frame = 0
                while current_start_frame + frame_cnt_clip <= frame_cnt_video:
                    clip_starts.append(current_start_frame)
                    current_start_frame += frame_int_clip

                # Process clips
                for id_clip, clip_start in enumerate(clip_starts):
                    clip_end = clip_start + frame_cnt_clip
                    if clip_end > frame_cnt_video:
                        break

                    # Extract clip frames
                    df_clip = df_video.iloc[clip_start:clip_end].copy()

                    #! HIGHLY IMPORTANT

                    #! NANs in rows (whole frames)
                    #! Unfortunately these rows cannot be deleted because otherwise we moment when dedector would stop extract features properly would be hard to detect
                    # nan_row_theshold_percent = 0.1
                    # NAN_THRESHOLD_V = int(len(df_clip.columns) * NAN_THRESHOLD_P)
                    # # Remove the rows which go over this threshold
                    # df_clip = df_clip[df_clip.isna().sum(axis=1) < NAN_THRESHOLD_V]
                    # # Calculate the percentage of removed rows
                    # removed_row_count = frame_cnt_clip - len(df_clip)
                    # removed_percentage = (removed_row_count / frame_cnt_clip) * 100
                    # # If more than 30 percent of the frames were removed, this is not a high-quality video, remove it
                    # if removed_percentage > 30:
                    #     continue # skip this clip
                    
                    #! NANs in columns (whole features)
                    # Check the percentage of NaNs in each column
                    for col in df_clip.columns:
                        nan_ratio = df_clip[col].isna().mean()
                        # If more than 30% are NaNs, make all NaN values 0, otherwise fill with mean
                        if nan_ratio > 0.3:
                            df_clip[col] = df_clip[col].fillna(0)
                        else:
                            df_clip[col] = df_clip[col].fillna(df_clip[col].mean())
                    # No NANs after this
                    
                    # Compute the correlation matrix
                    corr_matrix = df_clip.corr()

                    # Flatten correlation matrix into unique pairs
                    corr_pairs = {
                        f"{min(col1, col2)}*{max(col1, col2)}": corr_matrix.loc[col1, col2] # Make sure col1*col2 and co2*cor1 are treated as same 
                        for col1 in corr_matrix.columns
                        for col2 in corr_matrix.columns
                        if col1 != col2 # Autocorrelations are excluded
                    }

                    # Add to correlations DataFrame
                    clip_label = f"{video_basename}_c{id_clip:05d}"
                    correlations = pd.concat([correlations, pd.DataFrame(corr_pairs, index=[clip_label])])

            # Save the computed correlations for this clip configuration
            output_filename = f"correlations_l{length_clip}_o{overlap_clip}.csv"
            output_path_subdataset = os.path.join(output_path, subdataset)    
            # Ensure output directory exists
            os.makedirs(output_path_subdataset, exist_ok=True)
            correlations_name = os.path.join(output_path_subdataset, output_filename)
            correlations.to_csv(correlations_name)