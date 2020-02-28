cfg = {
    # data
    "train_dir": {
        "hr": "./data/DIV2K_train_HR",
        "lr": "./data/DIV2K_train_LR_mild"
        },
    "valid_dir": {
        "hr": "./data/DIV2K_valid_HR",
        "lr": "./data/DIV2K_valid_LR_x8"
        },
    # training
    "n_epoch": 300,
    "init_lr": 5e-5,
    "batch_size": 16,
    # model
    "n_feats": 64,
    "n_residual_blocks": 16,
    "scale": 4,
    # misc
    "sample_dir": "./samples",
    "model_dir": "./models"
}
