import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torchvision.utils as vutils
import sys, os 
import argparse
from accelerate import Accelerator
from accelerate.utils import find_batch_size
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

# Add import for the model registry
from cmb_cond_models import MODEL_REGISTRY

# Add model selection to argument parser
parser = argparse.ArgumentParser(description='Conditional Diffusion Model for CMB')
parser.add_argument('--evaluate', action='store_true', help='Skip training and evaluate the model only')
parser.add_argument('--model', type=str, default='original', 
                   choices=MODEL_REGISTRY.keys(), 
                   help='Model architecture to use')
parser.add_argument('--sample_size', type=int, default=256,
                   help='Sample size for UNet (should be compatible with downsampling depth)')
args = parser.parse_args()

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

with open('load_data.py') as f:
    exec(f.read())

# Get the model class based on the selected architecture
ModelClass = MODEL_REGISTRY[args.model]

# Initialize the selected model
if args.model == 'original':
    net = ModelClass().to(device)
else:
    # For the new models, pass the sample_size argument
    net = ModelClass(sample_size=args.sample_size).to(device)

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Name:", torch.cuda.get_device_name(0))
    print("Supports Mixed Precision:", torch.cuda.get_device_capability(0)[0] >= 7)

# Set mixed_precision to 'fp16' for faster training if your GPU supports it
# Set cpu=True if you want to debug on CPU
accelerator = Accelerator(
    mixed_precision='fp16',  # Options: 'no', 'fp16', 'bf16'
    cpu=False
)
device = accelerator.device

print(f"Using device: {device}")
print(f"Process count: {accelerator.num_processes}")

# Initialize variables for early stopping
min_valid_loss = float('inf')
best_model_path = 'best_model_conditional_diffusion.pth'
checkpoint_dir  = 'cond_diff_checkpoints'

# Create model-specific checkpoint directory if not using the original model
if args.model != 'original':
    checkpoint_dir = f'cond_diff_checkpoints_{args.model}'
    best_model_path = f'best_model_conditional_diffusion_{args.model}.pth'

save_directory = checkpoint_dir
os.makedirs(checkpoint_dir, exist_ok=True)
path_to_checkpoint = os.path.join(save_directory, best_model_path)

patience = 10
patience_counter = 0

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

# Prepare loaders based on whether we're evaluating or training
if args.evaluate:
    # For evaluation, we only need to prepare the test_loader and the model
    net, test_loader = accelerator.prepare(net, test_loader)
    print("Evaluation mode: Training will be skipped")
else:
    # For training, prepare all loaders
    net, opt, train_loader, valid_loader, test_loader = accelerator.prepare(
        net, opt, train_loader, valid_loader, test_loader
)

# Print shapes of input and target images from the first batch of train_loader
# print("\n" + "="*50)
# print("Checking train data shapes:")
# for batch_idx, (small_fg, large_fg) in enumerate(train_loader):
#     print(f"Input small-scale CMB (condition) shape: {small_fg.shape}")
#     print(f"Target images shape: {large_fg.shape}")
#     print(f"Frequencies data type: {small_fg.dtype}, min: {small_fg.min().item()}, max: {small_fg.max().item()}")
#     print(f"Images data type: {large_fg.dtype}, min: {large_fg.min().item()}, max: {large_fg.max().item()}")
#     break  # Only process the first batch
# print("="*50 + "\n")

# print("Checking test data shapes:")
# for batch_idx, (small_fg, large_fg) in enumerate(test_loader):
#     print(f"Input small-scale CMB (condition) shape: {small_fg.shape}")
#     print(f"Target images shape: {large_fg.shape}")
#     print(f"Frequencies data type: {small_fg.dtype}, min: {small_fg.min().item()}, max: {small_fg.max().item()}")
#     print(f"Images data type: {large_fg.dtype}, min: {large_fg.min().item()}, max: {large_fg.max().item()}")
#     break  # Only process the first batch
# print("="*50 + "\n")

# Check if the saved model exists
if os.path.exists(path_to_checkpoint):
    net = accelerator.unwrap_model(net)
    checkpoint = torch.load(path_to_checkpoint, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Loaded saved model")
    net.load_state_dict(checkpoint['model_state_dict'])
    
    if not args.evaluate:
        # Only load optimizer state if training
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        #Restore other relevant states
        start_epoch = checkpoint['epoch']
        min_valid_loss = checkpoint['min_valid_loss']
        patience_counter = checkpoint['patience_counter']
        print(f"Resumed from epoch {start_epoch}")
    
    net.to(device)
else:
    if args.evaluate:
        print("Error: No saved model found for evaluation")
        sys.exit(1)
    net.to(device)

# Our loss finction
loss_fn = nn.MSELoss()

# For plotting
train_loss_history = []
valid_loss_history = []

# Noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000) # num_train_timesteps?


# Skip training loop if in evaluation mode
if not args.evaluate:
    # The training loop
    n_epochs = 100  # Adjust as needed

    # Storage for plotting losses
    train_losses_per_epoch = []
    valid_losses_per_epoch = []
    min_valid_losses = []
    epochs_completed = []

    for epoch in range(n_epochs):
        # Training phase
        net.train()
        train_losses = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
        for y, x in progress_bar:
            batch_size = find_batch_size(x)
            
            # Generate noisy input
            timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)        
            noise = torch.randn_like(x)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

            # Get model prediction
            pred = net(noisy_x, timesteps, y) # Note that we pass in the small-scale FG: y

            # Calculate loss
            loss = loss_fn(pred, noise)

            # Backprop
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()

            train_losses.append(accelerator.gather(loss).mean().item())
            progress_bar.set_postfix({"train_loss": train_losses[-1]})

        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        train_loss_history.append(avg_train_loss)

        # **SAVE PLOTS OF y, x, AND pred**
        if accelerator.is_main_process:
            with torch.no_grad():
                first_y = y[0].cpu()  # Move to CPU
                first_x = x[0].cpu()
                first_pred = pred[0].cpu()

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                
                axes[0].imshow(first_y.squeeze(), cmap='gray')
                axes[0].set_title("Ground Truth (y)")

                axes[1].imshow(first_x.squeeze(), cmap='gray')
                axes[1].set_title("Input Image (x)")

                axes[2].imshow(first_pred.squeeze(), cmap='gray')
                axes[2].set_title("Predicted Image (pred)")

                for ax in axes:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(f"epoch_{epoch+1:03d}_train_samples.png")
                plt.close()

        # Validation phase
        net.eval()
        valid_losses = []
        
        progress_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]")
        with torch.no_grad():
            for y, x in progress_bar:
                timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)        
                noise = torch.randn_like(x)
                noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

                # Get model prediction
                pred = net(noisy_x, timesteps, y)

                # Calculate loss
                loss = loss_fn(pred, noise)
                valid_losses.append(accelerator.gather(loss).mean().item())

                progress_bar.set_postfix({"valid_loss": valid_losses[-1]})

        # Calculate average validation loss
        avg_valid_loss = sum(valid_losses) / len(valid_losses)
        valid_loss_history.append(avg_valid_loss)

        # Print loss
        accelerator.print(f'Epoch {epoch+1}/{n_epochs}: Train Loss = {avg_train_loss:.5f}, Valid Loss = {avg_valid_loss:.5f}')

        # **Early Stopping & Checkpointing**
        if accelerator.is_main_process:
            if avg_valid_loss < min_valid_loss:
                min_valid_loss = avg_valid_loss
                patience_counter = 0
                
                accelerator.wait_for_everyone()
                accelerator.save_model(net, save_directory)
                accelerator.save_state(checkpoint_dir)

                unwrapped_model = accelerator.unwrap_model(net)
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': unwrapped_model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'min_valid_loss': min_valid_loss,
                    'patience_counter': patience_counter
                }
                accelerator.save(checkpoint, path_to_checkpoint)
                print(f'New best model saved with validation loss: {min_valid_loss:.5f}')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping triggered after {epoch + 1} epochs')
                    break

        # **Plot Loss Curves Every 5 Epochs**
        if accelerator.is_main_process and epoch % 5 == 0 and epoch != 0:
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss_history, label='Training Loss')
            plt.plot(valid_loss_history, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Store losses for plotting
            train_losses_per_epoch.append(avg_train_loss)
            valid_losses_per_epoch.append(avg_valid_loss)
            min_valid_losses.append(min(min_valid_loss, avg_valid_loss))  # Track minimum validation loss so far
            epochs_completed.append(epoch + 1)

    accelerator.print('Training completed!')

    # Plot final loss curves
    if accelerator.is_main_process:
        plt.figure(figsize=(12, 6))
        
        # Plot train and validation losses
        plt.plot(epochs_completed, train_losses_per_epoch, 'b-', label='Training Loss')
        plt.plot(epochs_completed, valid_losses_per_epoch, 'g-', label='Validation Loss')
        plt.plot(epochs_completed, min_valid_losses, 'r--', label='Min Validation Loss')
        
        # Add markers for the best model
        best_epoch_idx = min_valid_losses.index(min(min_valid_losses))
        best_epoch = epochs_completed[best_epoch_idx]
        best_valid_loss = min_valid_losses[best_epoch_idx]
        plt.plot(best_epoch, best_valid_loss, 'ro', markersize=8, label=f'Best Model (Epoch {best_epoch})')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training, Validation, and Minimum Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # Add text annotation for best model
        plt.annotate(f'Best: {best_valid_loss:.6f}',
                    xy=(best_epoch, best_valid_loss),
                    xytext=(best_epoch + 1, best_valid_loss * 1.1),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8))
        
        plt.savefig(os.path.join(save_directory, "training_history.png"), dpi=300)
        plt.close()
        
        # Save the loss data as NPY files for future reference
        losses_dict = {
            'epochs': np.array(epochs_completed),
            'train_loss': np.array(train_losses_per_epoch),
            'valid_loss': np.array(valid_losses_per_epoch),
            'min_valid_loss': np.array(min_valid_losses),
            'best_epoch': best_epoch,
            'best_valid_loss': best_valid_loss
        }
        
        np.save(os.path.join(save_directory, "training_losses.npy"), losses_dict)
        
        print(f"Saved training history plot and losses data to {save_directory}")

# Make sure the test_loader is loaded to the correct device
device = next(net.parameters()).device
print(f"Using device: {device}")

# Create directory for saving generated samples
save_directory_dataset = os.path.join(save_directory, "test_samples")
os.makedirs(save_directory_dataset, exist_ok=True)

# Evaluation section - this runs for both training and evaluate modes
print("\n" + "="*50)
print("Starting model evaluation on test dataset...")
print("="*50 + "\n")

# Ensure the model is in evaluation mode
net.eval()

# List to store correlation and MSE values
correlations = []
mse_values = []

# Loop through batches in the test loader
for batch_idx, (fg_small, fg_large) in enumerate(tqdm(test_loader, desc="Evaluating test data")):
    # Move data to the correct device
    fg_small = fg_small.to(device)
    fg_large = fg_large.to(device)
    
    # Check dimensions for first batch
    if batch_idx == 0:
        print(f"Conditioning input shape: {fg_small.shape}")
        print(f"Target image shape: {fg_large.shape}")
    
    # Initialize x with random noise
    batch_size = fg_small.shape[0]
    img_size = fg_large.shape[2:]  # Assuming fg_large has the target output size
    
    # Initialize random noise (match the shape of fg_large)
    x = torch.randn(batch_size, fg_large.shape[1], *img_size, device=device)
    
    # Sampling loop
    for t in tqdm(noise_scheduler.timesteps, desc=f"Generating samples for batch {batch_idx+1}"):
        # Get model prediction
        with torch.no_grad():
            residual = net(x, t, fg_small)
        
        # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample
    
    # Get numpy arrays for calculations
    gen_np = x.detach().cpu().numpy()
    true_np = fg_large.detach().cpu().numpy()
    
    # Calculate metrics for each image in the batch
    batch_correlations = []
    batch_mse = []
    
    for i in range(batch_size):
        # Flatten images for correlation calculation
        gen_flat = gen_np[i].flatten()
        true_flat = true_np[i].flatten()
        
        # Calculate correlation coefficient
        corr = np.corrcoef(gen_flat, true_flat)[0, 1]
        batch_correlations.append(corr)
        
        # Calculate MSE
        mse = ((gen_np[i] - true_np[i])**2).mean()
        batch_mse.append(mse)
    
    # Add to overall metrics
    correlations.extend(batch_correlations)
    mse_values.extend(batch_mse)
    
    # Calculate average metrics for this batch
    avg_corr = np.mean(batch_correlations)
    avg_mse = np.mean(batch_mse)
    
    print(f"Batch {batch_idx+1} - Avg Correlation: {avg_corr:.4f}, Avg MSE: {avg_mse:.6f}")
    
    # Plot comparison for each image in the batch
    for i in range(batch_size):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot true image
        im1 = axes[0].imshow(true_np[i, 0], cmap='RdYlBu')
        axes[0].set_title("True Target")
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Plot generated image
        im2 = axes[1].imshow(gen_np[i, 0], cmap='RdYlBu')
        axes[1].set_title(f"Generated (Corr: {batch_correlations[i]:.4f}, MSE: {batch_mse[i]:.6f})")
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory_dataset, f"batch_{batch_idx+1}_sample_{i+1}.png"), dpi=300)
        plt.close()

# Print overall statistics
avg_correlation = np.mean(correlations)
avg_mse = np.mean(mse_values)
print("\n" + "="*50)
print(f"Overall Test Results - Avg Correlation: {avg_correlation:.4f}, Avg MSE: {avg_mse:.6f}")
print("="*50 + "\n")

print("\nTest evaluation completed!")
