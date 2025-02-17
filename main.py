import os
import shutil
import warnings
import multiprocessing
import numpy as np
import sklearn
import sklearn.metrics
import monai
from monai.config import print_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, StochasticWeightAveraging
import torch
from torch.utils.data import DataLoader
import dotenv

# Local imports
from config import get_args
from model import SEGNET, BasicUNet
from dataset import PCGDataset
from utils import set_seed

# Load environment variables from .env file
dotenv.load_dotenv()


# ----------------------------
# Helper Functions
# ----------------------------
def create_or_reset_folder(folder_path: str):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)


def load_data_2016(load_path: str):
    data2016 = np.load(load_path, allow_pickle=True)
    data_test2016 = data2016[:336]
    data_train2016 = data2016[336:]
    data_train2016, data_valid2016 = sklearn.model_selection.train_test_split(
        data_train2016, test_size=0.2, random_state=42
    )
    return data_train2016, data_valid2016, data_test2016


def build_trainer(model, max_epochs=100, learning_rate=1e-3):
    trainer = pl.Trainer(
        log_every_n_steps=1,
        gradient_clip_algorithm='norm',
        accumulate_grad_batches=4,
        sync_batchnorm=True,
        benchmark=True,
        accelerator='gpu',
        devices=-1,  # Use all available GPUs
        max_epochs=max_epochs,
        strategy='ddp_find_unused_parameters_true',
        check_val_every_n_epoch=1,
        callbacks=[
            model.checkpoint_callback,
            LearningRateMonitor(),
            EarlyStopping('val_loss', patience=20),
            StochasticWeightAveraging(
                swa_epoch_start=0.1,
                annealing_epochs=2,
                swa_lrs=learning_rate
            )
        ],
    )
    return trainer


# ----------------------------
# Main Function
# ----------------------------
def main():
    args = get_args()

    # ----------------------
    # Environment Setup
    # ----------------------
    os.environ["HTTP_PROXY"] = os.getenv("HTTP_PROXY", "")
    os.environ["HTTPS_PROXY"] = os.getenv("HTTPS_PROXY", "")
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Suppress warnings
    warnings.filterwarnings(action='ignore')

    # ----------------------
    # Device and Worker Setup
    # ----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = os.cpu_count()
    print("Number of workers:", NUM_WORKERS)
    print("multiprocessing.cpu_count()", multiprocessing.cpu_count())
    print("cuda.is_available:", torch.cuda.is_available())
    print("Using device:", device)
    print_config()

    # Fixed parameters and constants
    IN_CHANNELS = 2
    OUT_CHANNELS = 4
    MINSIZE = 50
    THR = 0.5
    VERSION = 2
    MAX_EPOCHS = 250
    LEARNING_RATE = 2e-4

    # Create a comment string to name your results more systematically
    comment = (
        f"ver{VERSION}_d{args.target_sr}_v{args.ver}_low{args.lowpass}"
        f"twice_{args.twice}_nofft_{args.fft}"
    )

    load_path = f"/workspace/data/PhysioNet{args.year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
    infer_pth = f"/workspace/data/lightning_logs/version_{args.ver}/checkpoints/"

    set_seed(args.seed)
    print(f"pytorch_lightning version: {pl.__version__}")

    # Model setup
    net = BasicUNet(
        spatial_dims=1,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        features=(64, 64, 128, 256, 512, 512, 64),
        norm='instance',
        upsample='pixelshuffle',
        act='gelu',
        fft=args.fft,          # FFT-based layer
        twice=args.twice       # double-Conv in Down / UpCat
    )

    model = SEGNET(
        net=net,
        featureLength=args.featureLength,
        learning_rate=LEARNING_RATE,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        minsize=MINSIZE,
        thr=THR,
        device=device,
        path=f'/workspace/data/pcg_2016_jupyters/result/{args.year}_toler{args.toler}_{comment}/',
        toler=args.toler
    )

    # ----------------------
    # Training Phase
    # ----------------------
    if not args.infer:
        path = f"/workspace/data/pcg_2016_jupyters/result/{args.year}_toler{args.toler}_{comment}/"
        create_or_reset_folder(path)
        data_train2016, data_valid2016, data_test2016 = load_data_2016(load_path)
        print(len(data_train2016), len(data_valid2016), len(data_test2016))

        # Create datasets & loaders
        train_ds2016 = PCGDataset(data_train2016, 'train')
        train_loader = DataLoader(
            train_ds2016,
            shuffle=True,
            batch_size=args.batch,
            drop_last=True
        )
        valid_ds2016 = PCGDataset(data_valid2016)
        valid_loader = DataLoader(
            valid_ds2016,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        test_ds2016 = PCGDataset(data_test2016)
        test_loader = DataLoader(
            test_ds2016,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )

        trainer = build_trainer(model, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
        trainer.fit(model, train_loader, valid_loader)
    else:
        print("Inference mode requested. Skipping training.")

    # ----------------------
    # Testing Phases
    # ----------------------
    if not args.not_2016:
        year = 2016
        test_path = f"/workspace/data/pcg_2016_jupyters/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)

        data2016 = np.load(load_path, allow_pickle=True)
        data_test2016 = data2016[:336]
        test_ds2016 = PCGDataset(data_test2016)
        test_loader_2016 = DataLoader(
            test_ds2016,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )

        print("\n############# Toler 40 Internal 2016 start #############\n")
        checkpoint_file = os.path.join(infer_pth, 'best.ckpt')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        print("Checkpoint state_dict keys:", len(checkpoint['state_dict'].keys()))
        print("Model state_dict keys:", len(model.state_dict().keys()))
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
        trainer.test(model, test_loader_2016, ckpt_path=checkpoint_file)

    if not args.not_2022:
        print("\nToler 40 External 2022 start\n")
        year = 2022
        test_path = f"/workspace/data/pcg_2016_jupyters/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)
        new_load_path = f"/workspace/data/PhysioNet{year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
        data2022 = np.load(new_load_path, allow_pickle=True)
        test_ds2022 = PCGDataset(data2022)
        test_loader_2022 = DataLoader(
            test_ds2022,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
        trainer.test(model, test_loader_2022, ckpt_path=os.path.join(infer_pth, 'best.ckpt'))

    if not args.not_amc:
        print("\nToler 40 External amc start\n")
        year = "amc"
        test_path = f"/workspace/data/pcg_2016_jupyters/result/{year}_toler{args.toler}_{comment}/"
        if not args.nofolder:
            create_or_reset_folder(test_path)
        amc_load_path = f"/workspace/data/{year}_{args.target_sr}Hz_{args.lowpass}_fe_{args.featureLength}.npy"
        data_amc = np.load(amc_load_path, allow_pickle=True)
        test_ds_amc = PCGDataset(data_amc)
        test_loader_amc = DataLoader(
            test_ds_amc,
            batch_size=1,
            collate_fn=monai.data.utils.default_collate
        )
        trainer = build_trainer(model, max_epochs=MAX_EPOCHS, learning_rate=LEARNING_RATE)
        trainer.test(model, test_loader_amc, ckpt_path=os.path.join(infer_pth, 'best.ckpt'))


if __name__ == "__main__":
    main()
