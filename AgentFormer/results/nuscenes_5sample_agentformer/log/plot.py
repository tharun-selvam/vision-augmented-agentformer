import re
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import numpy as np

def parse_log_by_epoch(log_file):
    """
    Parses the log file and calculates the average loss for each epoch.
    """
    
    # Regex to capture:
    # Group 1: Current Epoch (e.g., "00")
    # Group 2: Current Iteration (e.g., "0021")
    # Group 3: Total Iterations (e.g., "1034")
    # Group 4: kld avg loss (e.g., "386.909")
    # Group 5: diverse avg loss (e.g., "0.985")
    # Group 6: recon avg loss (e.g., "16.436")
    # Group 7: total_loss avg loss (e.g., "121.855")
    pattern = re.compile(
        r"Epo: (\d+)/\d+, It: (\d+)/(\d+).*"
        r"kld: [\d\.]+ \((\d+\.\d+)\).*"
        r"diverse: [\d\.]+ \((\d+\.\d+)\).*"
        r"recon: [\d\.]+ \((\d+\.\d+)\).*"
        r"total_loss: [\d\.]+ \((\d+\.\d+)\)"
    )

    # Use defaultdict to automatically create empty lists for new epochs
    epoch_data_accumulator = defaultdict(lambda: {
        'kld': [],
        'diverse': [],
        'recon': [],
        'total_loss': []
    })

    print(f"Parsing log file: {log_file}...")
    try:
        with open(log_file, 'r') as f:
            for line in f:
                # Lines like [DEBUG] will not match and will be skipped
                match = pattern.search(line)
                if match:
                    # Extract data
                    epoch, _, _, kld, diverse, recon, total = match.groups()
                    epoch_num = int(epoch)
                    
                    # Accumulate all the "running average" values from each line
                    epoch_data_accumulator[epoch_num]['kld'].append(float(kld))
                    epoch_data_accumulator[epoch_num]['diverse'].append(float(diverse))
                    epoch_data_accumulator[epoch_num]['recon'].append(float(recon))
                    epoch_data_accumulator[epoch_num]['total_loss'].append(float(total))
    
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file}'")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

    if not epoch_data_accumulator:
        print("No matching log entries found.")
        return None

    # --- Calculate Averages ---
    print("Parsing complete. Calculating epoch averages...")
    
    plot_data = {
        'epochs': [],
        'kld': [],
        'diverse': [],
        'recon': [],
        'total_loss': []
    }

    # Sort epochs to ensure correct plot order
    sorted_epochs = sorted(epoch_data_accumulator.keys())
    
    for epoch in sorted_epochs:
        plot_data['epochs'].append(epoch)
        
        # Get all losses for this epoch
        losses = epoch_data_accumulator[epoch]
        
        # Calculate the mean for each loss type and append
        plot_data['kld'].append(np.mean(losses['kld']))
        plot_data['diverse'].append(np.mean(losses['diverse']))
        plot_data['recon'].append(np.mean(losses['recon']))
        plot_data['total_loss'].append(np.mean(losses['total_loss']))

    print(f"Calculated averages for {len(plot_data['epochs'])} epochs.")
    return plot_data

def plot_epoch_losses(data):
    """
    Generates and saves a 2x2 grid of *per-epoch* average loss plots.
    """
    if data is None:
        return

    print("Generating plots...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Average Training Loss per Epoch', fontsize=16)
    
    x_axis_data = data['epochs']
    x_label = 'Epoch'

    # --- Plot 1: Total Loss ---
    axs[0, 0].plot(x_axis_data, data['total_loss'], 'bo-', label='Total Loss') # blue
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel(x_label)
    axs[0, 0].set_ylabel('Avg. Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # --- Plot 2: KLD Loss ---
    axs[0, 1].plot(x_axis_data, data['kld'], 'ro-', label='KLD Loss') # red
    axs[0, 1].set_title('KLD Loss')
    axs[0, 1].set_xlabel(x_label)
    axs[0, 1].set_ylabel('Avg. Loss')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # --- Plot 3: Recon Loss ---
    axs[1, 0].plot(x_axis_data, data['recon'], 'go-', label='Recon Loss') # green
    axs[1, 0].set_title('Reconstruction Loss')
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel('Avg. Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Plot 4: Diverse Loss ---
    axs[1, 1].plot(x_axis_data, data['diverse'], 'mo-', label='Diverse Loss') # magenta
    axs[1, 1].set_title('Diversity Loss')
    axs[1, 1].set_xlabel(x_label)
    axs[1, 1].set_ylabel('Avg. Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Changed filename to avoid overwriting previous plots
    output_filename = 'epoch_loss_curves_v2.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    
    # Show the plot
    plt.show()

def main():
    log_file_name = 'log.txt' 
    
    if len(sys.argv) > 1:
        log_file_name = sys.argv[1]

    loss_data = parse_log_by_epoch(log_file_name)
    plot_epoch_losses(loss_data)

if __name__ == "__main__":
    main()