import os
import mne

os.environ["MNE_SKIP_NETWORK_TESTS"] = "1"  # Disable requests to confirm download

def load_data(subjects, runs, data_dir="../data/raw/files/"):
    """
    Load EEG data for given subjects and runs.
    """

    os.makedirs(data_dir, exist_ok=True) 

    all_raws = []

    for subject in subjects:
        print(f"\n=== Loading data from_volunteer {subject} ===")
        try:
            # Download data to the specified directory
            raw_fnames = mne.datasets.eegbci.load_data(subject, runs, path=data_dir)
            raws = [mne.io.read_raw_edf(f, preload=True, verbose=True) for f in raw_fnames]  
            raw = mne.concatenate_raws(raws)
        except Exception as e:
            warnings.warn(f"Skipping subject {subject} due to an error: {e}")
            continue

        # Set standard montage
        try:
            raw.set_montage("standard_1005", on_missing="ignore")
        except Exception as e:
            warnings.warn(f"Could not set montage for subject {subject}: {e}")

        # # Extract events from annotations
        # try:
        #     events, _ = mne.events_from_annotations(raw)
        #     new_annot = mne.annotations_from_events(
        #         events=events, 
        #         event_desc=new_labels_events, 
        #         sfreq=raw.info['sfreq'], 
        #         orig_time=raw.info['meas_date']
        #         )
        #     raw.set_annotations(new_annot)
        # except Exception as e:
        #     warnings.warn(f"Could not extract event for subject {subject}: {e}")

        all_raws.append(raw)

    if not all_raws:
        raise ValueError("No valid EEG data loaded. Check subject and run IDs.")

    # Concatenate all subjects' data
    return mne.concatenate_raws(all_raws)
