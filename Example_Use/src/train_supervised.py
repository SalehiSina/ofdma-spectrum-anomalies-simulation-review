# Import ---------------------------------------------------

import argparse
import os
import sys

import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm

import albumentations as trans
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

_repo_name = "Example_Use"
_module_path = __file__[: __file__.find(_repo_name) + len(_repo_name)]
sys.path.append(os.path.abspath(_module_path))

from utils.functions import set_random, load_config, fold_image_horizontal, flip_x_axis
from src.data_loading import SpectralImagesSupervised, load_labels
from src.models import ResNet18classification


set_random(42)

config_path = os.path.join(_module_path, "src", "config.yml")
config = load_config(config_path)


# ------------------------------------------------
# Batch size for train and inference 
# ------------------------------------------------
train_batch_size = config["hyperparameters"]["train_batch_size"]
inference_batch_size = config["hyperparameters"]["inference_batch_size"]

# ------------------------------------------------
# Image Transformations 
# ------------------------------------------------

base_transform = trans.Compose(
    [
        ToTensorV2(),
    ]
)

train_transform = trans.Compose(
    [
        trans.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        base_transform,
    ]
)

# ------------------------------------------------
# Data Loading 
# ------------------------------------------------

def get_dataset(root, df, load_dt_images=False):
    num_train_normal = int(config['supervised']["num_train_normal"])
    num_valid_normal = int(config["supervised"]["num_valid_normal"])
    num_train_abnormal = int(config["supervised"]["num_train_anomaly"])
    num_valid_abnormal = int(config["supervised"]["num_valid_anomaly"])

    jammers = ['barrage', 'deceptive', 'pilot', 'sweep']
    normal = df[df["jammer_type"] == "no jammer"]

    normal_train_frames = normal[:num_train_normal]
    normal_valid_frames = normal[
                          num_train_normal: num_train_normal + num_valid_normal
                          ]

    ab_train = []
    ab_valid = []

    per_jammer_train = num_train_abnormal // len(jammers)
    per_jammer_valid = num_valid_abnormal // len(jammers)

    for j in jammers:
        jammer_df = df[df['jammer_type'] == j]

        # deterministic split (keeps original order)
        train_part = jammer_df[:per_jammer_train]
        valid_part = jammer_df[
                     per_jammer_train: per_jammer_train + per_jammer_valid
                     ]

        ab_train.append(train_part)
        ab_valid.append(valid_part)

    abnormal_train_frames = pd.concat(ab_train, ignore_index=True)
    abnormal_valid_frames = pd.concat(ab_valid, ignore_index=True)

    train_frames = pd.concat(
        [normal_train_frames, abnormal_train_frames],
        ignore_index=True
    )

    valid_frames = pd.concat(
        [normal_valid_frames, abnormal_valid_frames],
        ignore_index=True
    )

    train_set = SpectralImagesSupervised(
        root,
        transform=base_transform,
        dataframe=train_frames,
        load_dt_images=load_dt_images,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )

    valid_set = SpectralImagesSupervised(
        root,
        transform=base_transform,
        dataframe=valid_frames,
        load_dt_images=load_dt_images,
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size=inference_batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, valid_loader


# ------------------------------------------------
# Training Loop
# ------------------------------------------------

def train(
    now,
    device,
    n_epochs,
    optimizer,
    scheduler,
    classifier,
    trainloader,
    validloader,
    criterion
):

    try:
        trial_name = f"Res_{now}"
        out_dir = os.path.join(
            _module_path, "Weights", "OUT", trial_name
        )
        os.makedirs(out_dir, exist_ok=True)

        log_dir = os.path.join(
            _module_path, "Weights", "LOG", trial_name
        )
        os.makedirs(log_dir, exist_ok=True)
        log = SummaryWriter(log_dir)

        best_val_loss = float("inf")

        for epoch in tqdm(range(n_epochs), desc="Training Epoch"):

            # ------------------------------------------------
            # Training
            # ------------------------------------------------
            classifier.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for _, img, label, _, _ in trainloader:

                img = img.to(device).float()

                label[label>0] = 1                
                label = label.to(device).float()
                
                r = torch.randint(10, 20, (1,)).item()
                rolled_img = fold_image_horizontal(img, r * 5)
                fliped_img = flip_x_axis(img)

                # Concatenate original and augmented batch
                combined_inputs = torch.cat([rolled_img, img, fliped_img], dim=0)
                combined_targets = torch.cat([label, label, label], dim=0)
                
                optimizer.zero_grad()

                outputs = classifier(combined_inputs)
                
                loss = criterion(outputs, combined_targets.unsqueeze(1))

                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).int()
                correct += (predicted == combined_targets.unsqueeze(1)).sum().item()
                total += label.size(0)

            train_loss /= len(trainloader)
            train_acc = correct / (3*total)

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------
            classifier.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for _, img, label, _, _  in validloader:

                    img = img.to(device).float()

                    label[label>0] = 1
                    label = label.to(device).float()

                    outputs = classifier(img)
                    loss = criterion(outputs, label.unsqueeze(1))

                    val_loss += loss.item()

                    #_, predicted = torch.max(outputs, 1)
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > 0.5).int()
                    correct += (predicted == label.unsqueeze(1)).sum().item()
                    total += label.size(0)

            val_loss /= len(validloader)
            val_acc = correct / total

            # ------------------------------------------------
            # Log Writing
            # ------------------------------------------------
            log.add_scalar("Loss/train", train_loss, epoch)
            log.add_scalar("Loss/valid", val_loss, epoch)
            log.add_scalar("Accuracy/train", train_acc, epoch)
            log.add_scalar("Accuracy/valid", val_acc, epoch)

            tqdm.write(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}"
            )

            # ------------------------------------------------
            # Save Best Model
            # ------------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                torch.save(
                    classifier.state_dict(),
                    os.path.join(out_dir, "best_model.pth"),
                )

        log.close()
        
        torch.save(
                    classifier.state_dict(),
                    os.path.join(out_dir, "final_model.pth"),
                )

    except Exception as e:
        print("Training interrupted. Error: ", e)



if __name__ == "__main__":

    # ------------------------------------------------
    # Parse arguments
    # ------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--Max_LR", type=float, required=True)
    parser.add_argument("--initial_weights", type=str, default=None)
    args = parser.parse_args()
    data_path = args.data_path
    max_lr = args.Max_LR
    initial_weights = args.initial_weights

    # ------------------------------------------------
    # GPU setup
    # ------------------------------------------------
    try:
        torch.distributed.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        print("Device:", device)
        Multiple_GPU = True

    except Exception as e:
        print(e)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device:", device)
        Multiple_GPU = False


    now = datetime.datetime.now().strftime("%m%d-%H%M%S")
    df = load_labels(data_path)
    train_loader, valid_loader = get_dataset(data_path, df, load_dt_images=False)

    # ------------------------------------------------
    # Model, Optimizer, and Scheduler setup
    # ------------------------------------------------
    
    classifier = ResNet18classification(initial_weights)
    classifier = classifier.to(device)

    n_epochs = config["hyperparameters"]["supervised"]["n_epochs"]
    reconstruction_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
    )

    # ------------------------------------------------
    # Training Loop
    # ------------------------------------------------
    train(
        now,
        device,
        n_epochs,
        optimizer,
        scheduler,
        classifier,
        train_loader,
        valid_loader,
        reconstruction_loss
    )

    if Multiple_GPU:
        torch.distributed.destroy_process_group()
    else:
        print("Finished!")