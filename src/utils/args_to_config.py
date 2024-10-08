from .config import RunConfig, TrainConfig, ModelConfig, Config


def get_config(args):
    model_config = ModelConfig(
        task_type=args.task_type,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        heads=args.heads,
    )
    train_config = TrainConfig(
        dropout=args.dropout,
        batch_size=args.batch_size,
        val_interval=args.val_interval,
        checkpoint=args.checkpoint,
        switch_training=args.switch_training,
        dont_load_optimizer=args.dont_load_optimizer,
        test=args.test,

        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        resume_warmup_iters=args.resume_warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        max_iters=args.max_iters,
        val_iters=args.val_iters,
        test_iters=args.test_iters,

        dataset_pattern=args.dataset_pattern,
        val_dataset_pattern=args.val_dataset_pattern,
        test_dataset_pattern=args.test_dataset_pattern,
        max_checkpoints=args.max_checkpoints,
        lr_scheduler=args.lr_scheduler,
        weight_decay=args.weight_decay,
        dataset_percentage=args.dataset_percentage,
        val_dataset_percentage=args.val_dataset_percentage,
    )
    run_config = RunConfig(
        base_dir=args.base_dir,
        run_id=args.run_id,
        parallel_mode=args.parallel_mode,
        nproc=args.nproc,
        local_rank=args.local_rank,
        dist_master_addr=args.dist_master_addr,
        dist_master_port=args.dist_master_port,
        dist_backend=args.dist_backend,
        case=args.case,
        datasets_dir=args.datasets_dir,
        wandb=args.wandb,
        compile=args.compile,
        async_to_device=args.async_to_device,
        fused_adamw=args.fused_adamw,
        flash=args.flash,
    )
    config = Config(
        model=model_config,
        train=train_config,
        run=run_config,
    )
    return config
