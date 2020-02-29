scale4_feat64_block16 = {
    # data
    "train_dir": {"hr": "./data/DIV2K_train_HR",
                  "lr": "./data/DIV2K_train_LR_mild"},
    "valid_dir": {"hr": "./data/DIV2K_valid_HR",
                  "lr": "./data/DIV2K_valid_LR_mild"},
    # model
    "n_feats": 64,
    "n_residual_blocks": 16,
    "scale": 4,
    # training
    "n_epoch": 2000,
    "init_lr": 1e-4,
    "batch_size": 16,
    "lr_decay_every": 20000
}

cfg_dict = {
    "scale4-feat64-block16": scale4_feat64_block16
}
