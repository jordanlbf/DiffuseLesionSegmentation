import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import model as model
import torch.nn.functional as F
import time

from SegFormer.test import predict_and_display
from loss_functions import DiceLoss, JaccardLoss, CombinedLoss, FocalLoss, dice_coefficient
from SegFormer.data_loader import JointTransform, SkinLesionDataset

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global variable to set the input size
INPUT_SIZE = (128, 128)  # Example input size

# Define joint transform with the global input size
joint_transform = JointTransform(size=INPUT_SIZE)

# Define test transform
test_transform = transforms.Compose([
    transforms.Resize(INPUT_SIZE),
    transforms.ToTensor()
])

model = model.model()

# Load the saved model state if available

# Ensure all model parameters require gradients
for param in model.parameters():
    assert param.requires_grad, "Some parameters do not require gradients"

# Define loss function options

def train(loss, load, save, lr, path):
    # Paths to your images and masks
    train_image_dir = path + "images"
    train_mask_dir = path + "masks"

    if load is not None:
        model.load_state_dict(torch.load(load))

    predict_and_display(model, device, "", test_transform)

    # Create dataset
    dataset = SkinLesionDataset(train_image_dir, train_mask_dir, joint_transform=joint_transform)
    print("Dataset Size: " + str(len(dataset)))

    # Choose the loss function
    loss_function = loss  # Options: "dice", "jaccard", "focal", "combined", "cross_entropy"

    if loss_function == "dice":
        criterion = DiceLoss()
    elif loss_function == "jaccard":
        criterion = JaccardLoss()
    elif loss_function == "focal":
        criterion = FocalLoss()
    elif loss_function == "combined":
        criterion = CombinedLoss()
    else:  # default to CrossEntropyLoss
        criterion = nn.CrossEntropyLoss()

    # Define optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)


    # Training loop with dynamic validation set creation and early stopping
    num_epochs = 250
    early_stopping_patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    val_split = 0.2  # 20% of the data used for validation

    for epoch in range(num_epochs):
        start_time = time.time()

        # Split dataset into training and validation sets
        dataset_size = len(dataset)
        val_size = int(val_split * dataset_size)
        train_size = dataset_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(epoch))

        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)



        # Training phase
        model.train()
        running_loss = 0.0
        running_dice = 0.0  # Initialize running Dice coefficient for training
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images).logits

            # Resize outputs to match the size of the target masks
            outputs = F.interpolate(outputs, size=INPUT_SIZE, mode='bilinear', align_corners=False)

            # Convert masks to the required shape
            masks = masks.squeeze(1).long()

            # Calculate loss
            loss = criterion(outputs, masks)

            # Calculate Dice coefficient
            dice = dice_coefficient(outputs, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice.item()

        train_loss = running_loss / len(train_dataloader)
        train_dice = running_dice / len(train_dataloader)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_dice = 0.0  # Initialize running Dice coefficient for validation
        with torch.no_grad():
            for val_images, val_masks in val_dataloader:
                val_images, val_masks = val_images.to(device), val_masks.to(device)

                # Forward pass
                val_outputs = model(val_images).logits

                # Resize outputs to match the size of the target masks
                val_outputs = F.interpolate(val_outputs, size=INPUT_SIZE, mode='bilinear', align_corners=False)

                # Convert masks to the required shape
                val_masks = val_masks.squeeze(1).long()

                # Calculate loss
                val_loss = criterion(val_outputs, val_masks)

                # Calculate Dice coefficient
                val_dice = dice_coefficient(val_outputs, val_masks)

                val_running_loss += val_loss.item()
                val_running_dice += val_dice.item()

        val_loss = val_running_loss / len(val_dataloader)
        val_dice = val_running_dice / len(val_dataloader)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Step the scheduler with the validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Time: {epoch_time:.2f} seconds")

        # Early stopping and saving the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), save)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    new_image_dir = "tests"
    predict_and_display(model, device, new_image_dir, test_transform)
    print("Training complete")

dataset1 = ''
dataset2 = ''
dataset3 = ''
dataset4 = ''

train("", None, "1.pth", 0.0005, dataset1)
train("", "1.pth", "2.pth", 0.0001, dataset2)
train("", "2.pth", "3.pth", 0.00005, dataset3)
train("", "3.pth", "final.pth", 0.00001, dataset4)