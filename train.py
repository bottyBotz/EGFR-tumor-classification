import random
import torch
import pytorch_lightning as pl
from src.conv3d import CNN3DModel
# Dataset was very tiny
# -> Didn't create class for dataset for time saving
from src.dataset import *
import src.config as config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_loss_weights():
    # Class Imbalance found through previous inspection:
    # 1    23
    # 0    94
    # Calculate weights
    total_samples = 94 + 23  # Total samples
    class_0_weight = 1 / 94  # Weight for class 0
    class_1_weight = 1 / 23  # Weight for class 1

    # Normalize weights
    normalized_class_0_weight = class_0_weight / (class_0_weight + class_1_weight)
    normalized_class_1_weight = class_1_weight / (class_0_weight + class_1_weight)

    # Create a tensor of weights
    loss_weights = torch.tensor(
        [normalized_class_0_weight, normalized_class_1_weight],
        dtype=torch.float32,
        device=config.DEVICE,
    )
    
    return loss_weights


if __name__ == "__main__":
    # Set the seed for reproducibility
    set_seed(42)
    
    loss_weights = calculate_loss_weights()
    print(f"Loss weights: {loss_weights}")
    
    # Initialize the model and trainer
    model = CNN3DModel(learning_rate=config.LEARNING_RATE, loss_weights=loss_weights)
    trainer = pl.Trainer(
        max_epochs=config.NUM_EPOCHS, log_every_n_steps=3, fast_dev_run=False,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=config.PATIENCE)]
    )

    # Train the model
    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader)
