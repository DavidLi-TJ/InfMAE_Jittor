from __future__ import annotations

SEED = 42
NUM_CLASSES = 9
IMAGE_SIZE = 224
NUM_WORKERS = 2

# ===== 预训练超参数 (与 PyTorch 对齐) =====
PRETRAIN_BATCH_SIZE = 48
PRETRAIN_EPOCHS = 50
PRETRAIN_LR = 1.5e-4
PRETRAIN_WARMUP_EPOCHS = 5
PRETRAIN_WEIGHT_DECAY = 0.05
PRETRAIN_MASK_RATIO = 0.75
INF30_MAX_SAMPLES = 2000

# ===== 下游超参数 (与 PyTorch 对齐) =====
DOWNSTREAM_BATCH_SIZE = 30
DOWNSTREAM_EPOCHS = 40
DOWNSTREAM_LR = 1e-4
DOWNSTREAM_BACKBONE_LR = 0       # 0 表示自动取 0.1 * DOWNSTREAM_LR
MSRS_TRAIN_MAX_SAMPLES = 0       # 0 = 不限，使用全量训练集
MSRS_VAL_MAX_SAMPLES = 0         # 0 = 不限，使用全量验证集
MSRS_TEST_MAX_SAMPLES = 100

# ===== 数据路径 =====
INF30_DATA_PATH = 'data/inf30_subset'
MSRS_DATA_ROOT = 'data/msrs_shared'
MSRS_MANIFEST = 'data/msrs_shared/manifest.json'

# ===== 工作目录 =====
PRETRAIN_WORK_DIR = 'work_dirs/pretrain'
DOWNSTREAM_WORK_DIR = 'work_dirs/downstream'
DOWNSTREAM_TEST_DIR = 'work_dirs/downstream_test'

# ===== 兼容旧脚本引用 =====
BATCH_SIZE = DOWNSTREAM_BATCH_SIZE
EPOCHS = PRETRAIN_EPOCHS
LEARNING_RATE = DOWNSTREAM_LR
MASK_RATIO = PRETRAIN_MASK_RATIO


def as_dict() -> dict:
    return {
        "seed": SEED,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "batch_size": DOWNSTREAM_BATCH_SIZE,
        "epochs": PRETRAIN_EPOCHS,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "downstream_epochs": DOWNSTREAM_EPOCHS,
        "learning_rate": DOWNSTREAM_LR,
        "num_workers": NUM_WORKERS,
        "inf30_max_samples": INF30_MAX_SAMPLES,
        "msrs_train_max_samples": MSRS_TRAIN_MAX_SAMPLES,
        "msrs_val_max_samples": MSRS_VAL_MAX_SAMPLES,
        "msrs_test_max_samples": MSRS_TEST_MAX_SAMPLES,
        "mask_ratio": PRETRAIN_MASK_RATIO,
    }
