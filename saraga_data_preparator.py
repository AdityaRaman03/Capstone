import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import librosa
from typing import List, Tuple, Dict
import pickle
from pathlib import Path

class SaragaTrainingDataPreparator:
    """
    This class converts Saraga's melody annotations into training data.
    Think of it as a chef who takes raw ingredients (pitch data) 
    and prepares them into a meal (training data) that our AI can eat.
    """
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.pitch_scaler = StandardScaler()
        self.tonic_frequencies = {}  # Store tonic for each recording
        
    def extract_tonic_from_melody(self, melody_df: pd.DataFrame, method='histogram') -> float:
        """
        Finds the tonic (Sa) frequency from a melody.
        """
        frequencies = melody_df['frequency'][melody_df['frequency'] > 0]
        
        if method == 'histogram':
            hist, bins = np.histogram(frequencies, bins=50)
            peak_idx = np.argmax(hist)
            tonic_estimate = (bins[peak_idx] + bins[peak_idx + 1]) / 2
        elif method == 'minimum':
            tonic_estimate = np.percentile(frequencies, 10)  # 10th percentile
        else:
            tonic_estimate = np.median(frequencies)
        
        return tonic_estimate
    
    def frequency_to_swara(self, frequency: float, tonic_freq: float) -> str:
        """
        Converts a frequency (Hz) to a swara name (S, r, g, etc.).
        """
        if frequency <= 0:
            return 'silence'
        
        semitones = 12 * np.log2(frequency / tonic_freq)
        semitone_rounded = round(semitones) % 12
        
        semitone_to_swara = {
            0: 'S', 1: 'r', 2: 'R', 3: 'g',
            4: 'G', 5: 'm', 6: 'M', 7: 'P',
            8: 'd', 9: 'D', 10: 'n', 11: 'N'
        }
        return semitone_to_swara.get(semitone_rounded, 'unknown')
    
    def process_single_recording(self, recording_info: Dict) -> Dict:
        """
        Processes one recording and converts it to training data.
        """
        from saraga_processor import SaragaProcessor
        processor = SaragaProcessor(self.config)
        
        melody_df = processor.load_melody_annotation(recording_info['melody_file'])
        if melody_df is None:
            return None
        
        print(f"🎵 Processing: {recording_info['audio_file'].name}")
        
        tonic_freq = self.extract_tonic_from_melody(melody_df)
        print(f"   Tonic frequency: {tonic_freq:.1f} Hz")
        
        swaras, valid_frequencies = [], []
        for _, row in melody_df.iterrows():
            freq = row['frequency']
            if freq > 0:
                swara = self.frequency_to_swara(freq, tonic_freq)
                if swara in self.config.BHAIRAVI_SWARAS:
                    swaras.append(swara)
                    valid_frequencies.append(freq)
        
        if len(swaras) < 10:
            print(f"   Too few valid swaras ({len(swaras)}), skipping")
            return None
        
        semitones = [12 * np.log2(freq / tonic_freq) for freq in valid_frequencies]
        
        result = {
            'audio_file': recording_info['audio_file'].name,
            'tonic_freq': tonic_freq,
            'pitch_sequence': np.array(semitones),
            'swara_sequence': swaras,
            'num_frames': len(swaras)
        }
        
        print(f"   Generated {len(swaras)} swara labels")
        return result
    
    def create_training_sequences(self, processed_recordings: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates fixed-length sequences for training.
        """
        X_sequences, y_sequences = [], []
        sequence_length = self.config.SEQUENCE_LENGTH
        
        all_swaras = []
        for recording in processed_recordings:
            all_swaras.extend(recording['swara_sequence'])
        
        unique_swaras = list(set(all_swaras))
        unique_swaras.append('PAD')
        self.label_encoder.fit(unique_swaras)
        
        print(f"Swara classes: {self.label_encoder.classes_}")
        
        for recording in processed_recordings:
            pitch_seq = recording['pitch_sequence']
            swara_seq = recording['swara_sequence']
            swara_encoded = self.label_encoder.transform(swara_seq)
            
            step_size = sequence_length // 2
            for start_idx in range(0, len(pitch_seq) - sequence_length + 1, step_size):
                end_idx = start_idx + sequence_length
                pitch_window = pitch_seq[start_idx:end_idx]
                swara_window = swara_encoded[start_idx:end_idx]
                X_sequences.append(pitch_window)
                y_sequences.append(swara_window)
        
        for recording in processed_recordings:
            pitch_seq = recording['pitch_sequence']
            swara_seq = recording['swara_sequence']
            if len(pitch_seq) < sequence_length:
                swara_encoded = self.label_encoder.transform(swara_seq)
                pad_length = sequence_length - len(pitch_seq)
                pad_token_id = self.label_encoder.transform(['PAD'])[0]
                padded_pitch = np.pad(pitch_seq, (0, pad_length), 'constant', constant_values=0)
                padded_swaras = np.pad(swara_encoded, (0, pad_length), 'constant', constant_values=pad_token_id)
                X_sequences.append(padded_pitch)
                y_sequences.append(padded_swaras)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def prepare_training_data(self, bhairavi_recordings: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: Raw recordings → Training data
        """
        print("PREPARING TRAINING DATA")
        print("=" * 40)
        
        processed_recordings = []
        for i, recording in enumerate(bhairavi_recordings):
            print(f"\nProcessing {i+1}/{len(bhairavi_recordings)}...")
            if recording['melody_file'] is None:
                #print("   ❌ No melody annotation, skipping")
                continue
            result = self.process_single_recording(recording)
            if result is not None:
                processed_recordings.append(result)
        
        if len(processed_recordings) == 0:
            print("10 recordings could be processed!")
            return None, None
        
        print(f"\nSuccessfully processed {len(processed_recordings)} recordings")
        
        print("\nCreating training sequences...")
        X_data, y_data = self.create_training_sequences(processed_recordings)
        
        print("Normalizing pitch data...")
        X_data_reshaped = X_data.reshape(-1, 1)
        X_data_scaled = self.pitch_scaler.fit_transform(X_data_reshaped)
        X_data = X_data_scaled.reshape(X_data.shape)
        
        print(f"\nTRAINING DATA SUMMARY:")
        print(f"Number of sequences: {len(X_data)}")
        print(f"Sequence length: {X_data.shape[1]}")
        print(f"Number of swara classes: {len(self.label_encoder.classes_)}")
        print(f"Data shape - X: {X_data.shape}, y: {y_data.shape}")
        
        return X_data, y_data
    
    def split_train_test(self, X_data: np.ndarray, y_data: np.ndarray, test_size: float = 0.2):
        """
        Splits data into training and testing sets.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=42, stratify=None
        )
        print(f"DATA SPLIT:")
        print(f"Training set: {len(X_train)} sequences")
        print(f"Testing set: {len(X_test)} sequences")
        return X_train, X_test, y_train, y_test
    
    def save_preprocessing_info(self, filepath: str):
        """
        Saves the preprocessing settings.
        """
        preprocessing_info = {
            'label_encoder': self.label_encoder,
            'pitch_scaler': self.pitch_scaler,
            'config': self.config,
            'swara_classes': self.label_encoder.classes_.tolist()
        }
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessing_info, f)
        print(f"Preprocessing info saved: {filepath}")


class CarnaticDataset(Dataset):
    """ PyTorch Dataset for Carnatic music data. """
    def __init__(self, X_data: np.ndarray, y_data: np.ndarray):
        self.X = torch.FloatTensor(X_data).unsqueeze(-1)
        self.y = torch.LongTensor(y_data)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_data_loaders(X_train, X_test, y_train, y_test, config):
    """ Creates PyTorch data loaders. """
    train_dataset = CarnaticDataset(X_train, y_train)
    test_dataset = CarnaticDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"DATA LOADERS CREATED:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    return train_loader, test_loader


if __name__ == "__main__":
    print("TESTING DATA PREPARATION")
    print("=" * 40)
    
    from saraga_setup import SaragaConfig
    from saraga_processor import SaragaProcessor
    
    config = SaragaConfig()
    processor = SaragaProcessor(config)
    preparator = SaragaTrainingDataPreparator(config)
    
    print("Step 1: Loading Bhairavi recordings...")
    if hasattr(processor, 'bhairavi_recordings') and len(processor.bhairavi_recordings) > 0:
        recordings = processor.bhairavi_recordings
    else:
        print("No recordings loaded. Running processor first...")
        processor.explore_dataset_structure()
        #processor.load_metadata()
        recordings = processor.find_recordings_by_raga("Bhairavi")
    
    if len(recordings) == 0:
        print("No Bhairavi recordings found!")
        print("Please ensure the Saraga dataset is downloaded and extracted.")
    else:
        print(f"Found {len(recordings)} recordings")
        test_recordings = recordings[:3]
        
        print("\nStep 2: Preparing training data...")
        X_data, y_data = preparator.prepare_training_data(test_recordings)
        
        if X_data is not None:
            print("\nStep 3: Splitting data...")
            X_train, X_test, y_train, y_test = preparator.split_train_test(X_data, y_data)
            
            print("\nStep 4: Creating data loaders...")
            train_loader, test_loader = create_data_loaders(X_train, X_test, y_train, y_test, config)
            
            print("\nStep 5: Saving preprocessing info...")
            preparator.save_preprocessing_info("models/preprocessing_info.pkl")
            
            print("\nDATA PREPARATION COMPLETE!")
        else:
            print("Data preparation Successful!")
    
    print("\nNext: Run model training scripts")
