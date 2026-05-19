import pandas as pd
import numpy as np
import librosa
import json
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


class SaragaProcessor:
    """
    Handles operations with the Saraga Carnatic dataset.
    Reads metadata JSON files and links them with audio + annotations.
    """

    def __init__(self, config):
        self.config = config
        self.recordings = []  # will hold all recordings

    def explore_dataset_structure(self):
        """
        Check if Saraga dataset exists and return root path.
        """
        print("EXPLORING SARAGA DATASET STRUCTURE")
        print("=" * 50)

        base_path = Path(self.config.BASE_PATH)

        if not base_path.exists():
            print("Saraga dataset not found at", base_path)
            return False

        dataset_root = base_path / "saraga_carnatic" / "carnatic"
        if not dataset_root.exists():
            dataset_root = base_path / "carnatic"

        if not dataset_root.exists():
            print("Could not find saraga_carnatic/carnatic")
            return False

        print(f"Dataset root: {dataset_root}")
        self.config.DATASET_ROOT = dataset_root
        return True

    def find_recordings_by_raga(self, target_raga: str) -> List[Dict]:
        """
        Reads JSON metadata files and finds recordings for a given raga.
        """
        dataset_root = getattr(self.config, 'DATASET_ROOT', Path(self.config.BASE_PATH))
        recordings = []
        target_raga = target_raga.lower()

        json_files = list(dataset_root.rglob("*.json"))
        print(f"Found {len(json_files)} JSON metadata files")

        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if "raaga" in data and isinstance(data["raaga"], list):
                    for r in data["raaga"]:
                        raga_name = r.get("name", "").lower()
                        if target_raga in raga_name:
                            # Collect audio files in the same folder
                            audio_files = list(json_file.parent.glob("*.mp3")) + list(json_file.parent.glob("*.wav"))
                            for audio_file in audio_files:
                                recordings.append({
                                    "audio_file": audio_file,
                                    "json_file": json_file,
                                    "raga": r.get("name", ""),
                                    "melody_file": self.find_melody_annotation(audio_file)
                                })
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        self.recordings = recordings
        print(f"Found {len(recordings)} recordings for raga '{target_raga}'")
        return recordings

    def find_melody_annotation(self, audio_file: Path) -> Optional[Path]:
        """
        Try to find melody annotation CSV for a given audio file.
        """
        base_name = audio_file.stem
        search_paths = [audio_file.parent, audio_file.parent / "annotations", audio_file.parent / "melody"]

        for search_path in search_paths:
            if search_path.exists():
                candidates = list(search_path.glob("*.csv"))
                for csv_file in candidates:
                    if base_name in csv_file.name:
                        return csv_file
        return None

    def load_melody_annotation(self, melody_file: Path) -> Optional[pd.DataFrame]:
        """
        Load melody CSV into a DataFrame with time + frequency.
        """
        try:
            df = pd.read_csv(melody_file)

            # detect columns
            time_col = None
            freq_col = None
            for col in df.columns:
                if 'time' in col.lower():
                    time_col = col
                if any(word in col.lower() for word in ['freq', 'pitch', 'f0', 'melody']):
                    freq_col = col

            if not time_col or not freq_col:
                print(f"Could not find proper columns in {melody_file}")
                return None

            df = df.rename(columns={time_col: "time", freq_col: "frequency"})
            df = df[df["frequency"] > 0]
            df = df.dropna(subset=["time", "frequency"])
            return df
        except Exception as e:
            print(f"Error loading {melody_file}: {e}")
            return None

def find_melody_annotation(self, audio_file: Path) -> Optional[Path]:
    """
    Find melody annotation (pitch track) for a given audio file.
    Works with Saraga Carnatic where annotations are in .pitch-vocal.txt files.
    """
    base_name = audio_file.stem.lower()

    # candidate folders (same dir, annotations/, pitch/ etc.)
    search_paths = [
        audio_file.parent,
        audio_file.parent / "annotations",
        audio_file.parent / "melody",
        audio_file.parent / "annotations-melody",
        audio_file.parent / "pitch",
        audio_file.parent.parent / "annotations",
        audio_file.parent.parent / "annotations-melody",
        audio_file.parent.parent / "pitch"
    ]

    for search_path in search_paths:
        if search_path.exists():
            for f in search_path.glob("*.txt"):
                if "pitch" in f.stem.lower() and base_name.split("-")[0] in f.stem.lower():
                    return f

    return None



def load_pitch_file(pitch_file: Path) -> pd.DataFrame:
    """
    Load Saraga .pitch-vocal.txt file into a DataFrame.
    Columns: time (s), frequency (Hz)
    """
    df = pd.read_csv(pitch_file, sep="\t", header=None, names=["time", "frequency"])
    return df

def plot_pitch(df: pd.DataFrame, title="Pitch Contour"):
    """
    Plot pitch contour, ignoring unvoiced (0 Hz).
    """
    voiced = df[df["frequency"] > 0]
    plt.figure(figsize=(12, 4))
    plt.plot(voiced["time"], voiced["frequency"], linewidth=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.show()

# Example usage
pitch_file = Path("saraga_carnatic/carnatic/188/V. Shankaranarayanan - Shloka.pitch-vocal.txt")
df = load_pitch_file(pitch_file)
plot_pitch(df, title="V. Shankaranarayanan - Shloka")


# Example usage
if __name__ == "__main__":
    from saraga_setup import SaragaConfig  # your config class

    processor = SaragaProcessor(SaragaConfig)

    if processor.explore_dataset_structure():
        # Search for Bhairavi (or any raga)
        recs = processor.find_recordings_by_raga("Bhairavi")

        if recs:
            processor.visualize_sample_recording(0)
