import glob
import os
import mne
from tqdm import tqdm
import pandas as pd

def summarize_edf_files(data_dir="../data/raw/files/", limit=None, verbose=True):
    """
    Recursively summarize the contents of .edf files in the given directory.
    """
    edf_files = glob.glob(os.path.join(data_dir, "**/*.edf"), recursive=True)
    
    if not edf_files:
        print("No EDF files found in the directory or subdirectories.")
        return None
    
    if limit and isinstance(limit, int):
        edf_files = edf_files[:limit]
    
    if verbose:
        print(f"Found {len(edf_files)} EDF files in {data_dir}**")
    
    summaries = []
    
    # Use tqdm for progress tracking
    for edf_file in tqdm(edf_files, desc="Processing EDF files", disable=not verbose):
        try:
            # Using read_raw_edf with minimal options for faster loading
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False, 
                                     stim_channel=None, exclude=())
            info = raw.info
            
            rel_path = os.path.relpath(edf_file, data_dir)
            
            # Store information in a dictionary
            summary = {
                'file': rel_path,
                'file_size_mb': os.path.getsize(edf_file) / (1024 * 1024),
                'n_channels': len(info['ch_names']),
                'n_samples': len(raw),
                'sample_freq': info['sfreq'],
                'duration_sec': len(raw) / info['sfreq']
            }
            
            summaries.append(summary)
            
            # Force garbage collection of the raw object
            del raw
            
        except Exception as e:
            if verbose:
                print(f"Error reading {edf_file}: {e}")
    
    # Convert results to DataFrame for better organization
    summary_df = pd.DataFrame(summaries)
    
    if verbose and not summary_df.empty:
        print("\nSummary Statistics:")
        print(f"Total files: {len(summary_df)}")
        print(f"Total duration: {summary_df['duration_sec'].sum() / 60:.2f} minutes")
        print(f"Average channels per file: {summary_df['n_channels'].mean():.1f}")
        print(f"Average file duration: {summary_df['duration_sec'].mean():.2f} seconds")
    
    return summary_df