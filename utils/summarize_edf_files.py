import glob
import os
import mne

def summarize_edf_files(data_dir="../data/files/"):
    """
    Recursively summarize the contents of all .edf files in the given directory.
    """
    edf_files = glob.glob(os.path.join(data_dir, "**/*.edf"), recursive=True)
    
    if not edf_files:
        print("No EDF files found in the directory or subdirectories.")
        return
    
    print(f"Found {len(edf_files)} EDF files in {data_dir}**:\n")
    
    for edf_file in edf_files:
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=False, verbose=False)
            info = raw.info
            
            rel_path = os.path.relpath(edf_file, data_dir)  # Get relative path for better readability
            
            print(f"File: {rel_path}")
            print(f"  - Channels: {len(info['ch_names'])}")
            print(f"  - Length: {len(raw)}")
            print(f"  - Sampling Frequency: {info['sfreq']} Hz")
            print("-" * 40)
        
        except Exception as e:
            print(f"Error reading {edf_file}: {e}")