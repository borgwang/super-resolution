cfg = {
    "data": {
        "train": {
            "hr": "./data/DIV2K_train_HR",
            "lr": "./data/DIV2K_train_LR_x8"
            },
        "valid": {
            "hr": "./data/DIV2K_valid_HR",
            "lr": "./data/DIV2K_valid_LR_x8"
            }
        },
    "train": {
        "n_epoch": 300,
        "init_lr": 5e-5,
        "batch_size": 16
        },
    "model": {
        "n_feats": 64,
        "n_residual_blocks": 16,
        "scale": 4
        }
    }
