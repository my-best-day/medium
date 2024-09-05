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
    task_type: Optional[str] = None  # gpt, mlm, cola, sst2
    seq_len: Optional[int] = None
    d_model: Optional[int] = None
    n_layers: Optional[int] = None
    heads: Optional[int] = None


@dataclass
class TrainConfig(BaseConfig):
    batch_size: int
    val_interval: int
    dataset_pattern: str
    val_dataset_pattern: str
    test_dataset_pattern: str
    weight_decay: float
    dataset_percentage: float
    val_dataset_percentage: float

    learning_rate: float
    min_learning_rate: float
    warmup_iters: int
    resume_warmup_iters: int
    lr_decay_iters: int
    max_iters: int

    val_iters: int
    test_iters: int

    dropout: Optional[float] = None
    checkpoint: Optional[Path] = None

    # When loading from a checkpoint, if the task has changed (e.g., MLM to classification),
    # we skip loading the language model head, optimizer, and trainer states. If the task
    # remains the same, set this flag to start a new training phase (e.g., switching from
    # pre-training to fine-tuning or adjusting the training setup like learning rate schedule).
    switch_training: bool = False
    dont_load_optimizer: bool = False
    test: bool = False

    max_checkpoints: Optional[int] = None
    lr_scheduler: Optional[str] = None

    def __post_init__(self):
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
    nproc: int
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

    # might be set to False if loading from a checkpoint
    init_base_model_weights: bool = True
    # might be set to False if loading from a checkpoint
    init_lm_head_weights: bool = True

    # might be set to False if loading from a checkpoint
    # If loading from a checkpoint, we need to sync the optimizer state
    # across all processes to ensure the optimizer state is consistent.
    # Otherwise, the optimizer initialization is deterministic and we can skip
    # the synchronization.
    skip_sync_optimizer_state: bool = True

    def __post_init__(self):
        # verified by the caller, double checking here
        if isinstance(self.base_dir, str):
            self.base_dir = Path(self.base_dir)
        if not self.base_dir.exists():
            raise ValueError(f'Base directory {self.base_dir} does not exist.')

        if self.parallel_mode not in ['single', 'dp', 'ddp']:
            raise ValueError(f'Invalid parallel mode {self.parallel_mode}. '
                             'Valid values are single, dp, ddp.')

        if self.datasets_dir is None:
            self.datasets_dir = self.base_dir / 'datasets'
        if not self.datasets_dir.exists():
            raise ValueError(f'Datasets directory {self.datasets_dir} does not exist.')

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
