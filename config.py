scale4_feat64_block16 = {
    "train_dir": {"hr": "./data/DIV2K_train_HR",
                  "lr": "./data/DIV2K_train_LR_mild"},
    "valid_dir": {"hr": "./data/DIV2K_valid_HR",
                  "lr": "./data/DIV2K_valid_LR_mild"},
    "n_feats": 64,
    "n_residual_blocks": 16,
    "rescale": 1.0,
    "scale": 4,
    "n_epoch": 2000,
    "init_lr": 1e-4,
    "batch_size": 16,
    "lr_decay_every": 20000
}

scale4_feat256_block32 = {
    "train_dir": {"hr": "./data/DIV2K_train_HR",
                  "lr": "./data/DIV2K_train_LR_mild"},
    "valid_dir": {"hr": "./data/DIV2K_valid_HR",
                  "lr": "./data/DIV2K_valid_LR_mild"},
    "n_feats": 128,
    "n_residual_blocks": 32,
    "rescale": 0.1,
    "scale": 4,
    "n_epoch": 2000,
    "init_lr": 1e-4,
    "batch_size": 16,
    "lr_decay_every": 20000
}

cfg_dict = {
    "scale4-feat64-block16": scale4_feat64_block16,
    "scale4-feat256-block32": scale4_feat256_block32
}
