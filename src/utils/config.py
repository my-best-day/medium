import torch
from typing import Union, Optional
from pathlib import Path
from dataclasses import dataclass, fields


class BaseConfig:
    def to_dict(self):
        """
        A json serializable dict representation of the dataclass.
        Some types are converted to str for json serialization.
        """
        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, torch.device):
                value = str(value)
            elif isinstance(value, (ModelConfig, TrainConfig, RunConfig)):
                value = value.to_dict()
            result[f.name] = value
        return result


@dataclass
class ModelConfig(BaseConfig):
    seq_len: Optional[int] = None
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    heads: Optional[int] = None


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    val_interval: int
    start_epoch: int
    end_epoch: int
    dataset_pattern: str
    val_dataset_pattern: str
    weight_decay: float
    dataset_percentage: float
    val_dataset_percentage: float

    learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    lr_decay_iters: int
    max_iters: int

    dropout: Optional[float] = None
    checkpoint: Optional[Path] = None

    max_checkpoints: Optional[int] = None
    lr_scheduler: Optional[str] = None

    def __post_init__(self):
        if self.start_epoch < 0:
            raise ValueError(f'Invalid start epoch {self.start_epoch}. Must be >= 0.')

        if self.end_epoch < 0:
            raise ValueError(f'Invalid end epoch {self.end_epoch}. Must be >= 0.')

        if self.start_epoch > self.end_epoch:
            raise ValueError(f'Invalid start epoch {self.start_epoch} and end epoch '
                             f'{self.end_epoch}. Start epoch must be <= end epoch.')

        # max_checkpoints at least 1
        if self.max_checkpoints < 1:
            raise ValueError(f'Invalid max checkpoints {self.max_checkpoints}. Must be >= 1.')

        if self.checkpoint is not None and not self.checkpoint.exists():
            raise ValueError(f'Checkpoint {self.checkpoint} does not exist.')

        # verify percentage is between 0 and 1
        if self.dataset_percentage < 0 or self.dataset_percentage > 1:
            raise ValueError(f'Invalid dataset percentage {self.dataset_percentage}. '
                             'Must be between 0 and 1.')

        if self.val_dataset_percentage < 0 or self.val_dataset_percentage > 1:
            raise ValueError(f'Invalid val dataset percentage {self.val_dataset_percentage}. '
                             'Must be between 0 and 1.')


@dataclass
class RunConfig(BaseConfig):
    base_dir: Path
    run_id: int

    parallel_mode: str      # single, dp, ddp
    dist_master_addr: str
    dist_master_port: str
    dist_backend: str

    wandb: bool
    compile: bool
    async_to_device: bool
    fused_adamw: bool
    flash: bool

    datasets_dir: Path

    run_dir: Path = Path()
    logs_dir: Path = Path()
    checkpoints_dir: Path = Path()

    local_rank: Optional[int] = None
    device: Union[str, torch.device] = None
    is_primary: bool = True

    case: Optional[str] = None        # movies, instacart

    def __post_init__(self):
        # verified by the caller, double checking here
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if not self.base_dir.exists():
            raise ValueError(f'Base directory {self.base_dir} does not exist.')

        if self.parallel_mode not in ['single', 'dp', 'ddp']:
            raise ValueError(f'Invalid parallel mode {self.parallel_mode}. '
                             'Valid values are single, dp, ddp.')

        datasets_dir = self.base_dir / 'datasets'
        if not datasets_dir.exists():
            raise ValueError(f'Datasets directory {datasets_dir} does not exist.')

        self.run_dir = self.base_dir / 'runs' / f'run{self.run_id}'
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.run_dir / 'logs'
        self.logs_dir.mkdir(parents=False, exist_ok=True)

        self.checkpoints_dir = self.run_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=False, exist_ok=True)

        if self.case not in ['movies', 'instacart', 'dickens']:
            raise ValueError(f'Invalid case {self.case}. '
                             'Valid values are movies, instacart, dickens.')

        self.is_primary = self.local_rank in (None, 0)


@dataclass
class Config(BaseConfig):
    model: ModelConfig
    train: TrainConfig
    run: RunConfig
