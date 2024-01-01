import torch
from typing import Union
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ModelConfig:
    seq_len: int = None
    # vocab_size: int = None
    d_model: int = None
    n_layers: int = None
    heads: int = None
    dropout: float = None

@dataclass
class TrainConfig:
    batch_size: int
    val_interval: int
    start_epoch: int
    end_epoch: int
    dataset_pattern: str
    val_dataset_pattern: str
    weight_decay: float
    checkpoint: Path = None
    learning_rate: float = None
    max_checkpoints: int = None
    lr_scheduler: str = None

    def __post_init__(self):
        if self.start_epoch < 0:
            raise Exception(f'Invalid start epoch {self.start_epoch}. Must be >= 0.')

        if self.end_epoch < 0:
            raise Exception(f'Invalid end epoch {self.end_epoch}. Must be >= 0.')

        if self.start_epoch > self.end_epoch:
            raise Exception(f'Invalid start epoch {self.start_epoch} and end epoch {self.end_epoch}. Start epoch must be <= end epoch.')

        # max_checkpoints at least 1
        if self.max_checkpoints < 1:
            raise Exception(f'Invalid max checkpoints {self.max_checkpoints}. Must be >= 1.')

        if self.checkpoint is not None and not self.checkpoint.exists():
            raise Exception(f'Checkpoint {self.checkpoint} does not exist.')

@dataclass
class RunConfig:
    base_dir: Path
    run_id: int

    parallel_mode: str      # single, dp, ddp
    dist_master_addr: str
    dist_master_port: str
    dist_backend: str

    datasets_dir: Path

    run_dir: Path = Path()
    logs_dir: Path = Path()
    checkpoints_dir: Path = Path()

    local_rank: int = None
    device: Union[str, torch.device] = None


    case: str = None        # movies, instacart

    def __post_init__(self):
        # verified by the caller, double checking here
        if not self.base_dir.exists():
            raise Exception(f'Base directory {self.base_dir} does not exist.')

        if self.parallel_mode not in ['single', 'dp', 'ddp']:
            raise Exception(f'Invalid parallel mode {self.parallel_mode}. Valid values are single, dp, ddp.')

        datasets_dir = self.base_dir / 'datasets'
        if not datasets_dir.exists():
            raise Exception(f'Datasets directory {datasets_dir} does not exist.')

        self.run_dir = self.base_dir / 'runs' / f'run{self.run_id}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.run_dir / 'logs'
        self.logs_dir.mkdir(parents=False, exist_ok=True)

        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=False, exist_ok=True)

        if self.case not in ['movies', 'instacart']:
            raise Exception(f'Invalid case {self.case}. Valid values are movies, instacart.')

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    run: RunConfig
