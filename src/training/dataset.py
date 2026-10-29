"""
src/training/dataset.py
PyTorch Dataset for Phase 1 CKD Sequences.

Research Prototype | Synthetic Data | Not for Clinical Use
"""
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple

from src.features.sequence_builder import load_sequences

class CKDSequenceDataset(Dataset):
    """
    Loads pre-built Phase 1 sequence arrays into PyTorch Tensors.
    """
    def __init__(self, sequences_dir: str | Path, split_name: str):
        self.split_name = split_name
        
        # Load from disk via Phase 1 function
        seqs = load_sequences(sequences_dir, split_name)
        
        self.patient_ids = seqs["patient_ids"]
        self.static_features = torch.tensor(seqs["static_features"], dtype=torch.float32)
        self.historical_features = torch.tensor(seqs["historical_features"], dtype=torch.float32)
        self.time_features = torch.tensor(seqs["time_features"], dtype=torch.float32)
        self.masks = torch.tensor(seqs["masks"], dtype=torch.int8)
        
        # Targets
        # Stage is stored as 1.0 to 6.0 floats. CrossEntropy needs 0 to 5 longs.
        # We handle NaN mapping carefully.
        stage_raw = torch.tensor(seqs["future_targets"]["stage"], dtype=torch.float32)
        # Shift 1-6 -> 0-5. Leave NaNs as NaNs, we'll convert NaN to -1
        stage_shifted = torch.where(torch.isnan(stage_raw), torch.tensor(-1.0), stage_raw - 1.0)
        self.targets_stage = stage_shifted.long()
        
        self.targets_egfr = torch.tensor(seqs["future_targets"]["egfr"], dtype=torch.float32)
        self.targets_prog = torch.tensor(seqs["future_targets"]["progression"], dtype=torch.float32)
        
        self.n_patients = len(self.patient_ids)

    def __len__(self) -> int:
        return self.n_patients

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns
        -------
        static : Tensor [F_s]
        hist : Tensor [SeqLen, F_tv]
        time : Tensor [SeqLen, F_t]
        mask : Tensor [SeqLen]
        targets : dict of Tensors (stage, egfr, progression)
        """
        static = self.static_features[idx]
        hist = self.historical_features[idx]
        time = self.time_features[idx]
        mask = self.masks[idx]
        
        targets = {
            "stage": self.targets_stage[idx],
            "egfr": self.targets_egfr[idx],
            "progression": self.targets_prog[idx],
        }
        
        return static, hist, time, mask, targets

def get_dataloader(sequences_dir: str | Path, split_name: str, batch_size: int, shuffle: bool, num_workers: int = 0, pin_memory: bool = False):
    """Factory function for DataLoaders."""
    from torch.utils.data import DataLoader
    
    dataset = CKDSequenceDataset(sequences_dir, split_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
