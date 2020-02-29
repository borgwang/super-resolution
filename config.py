cfg = {
    # data
    "train_dir": {"hr": "./data/DIV2K_train_HR",
                  "lr": "./data/DIV2K_train_LR_mild"},
    "valid_dir": {"hr": "./data/DIV2K_valid_HR",
                  "lr": "./data/DIV2K_valid_LR_mild"},
    # model
    "n_feats": 64,
    "n_residual_blocks": 16,
    "scale": 4,
    "high_resolution_size": 256,
    # training
    "n_epoch": 10000,
    "init_lr": 1e-4,
    "batch_size": 16,
    # misc
    "sample_dir": "./samples",
    "checkpoint_dir": "./checkpoints"
}
