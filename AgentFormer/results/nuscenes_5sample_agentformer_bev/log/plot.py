import re
import matplotlib.pyplot as plt
import sys
from collections import defaultdict
import numpy as np

def parse_log_by_epoch(log_file):
    """
    Parses the new log file format and calculates the average loss for each epoch.
    """
    
    # Updated Regex to capture:
    # Group 1: Current Epoch (e.g., "00")
    # Group 2: Current Iteration (e.g., "0021")
    # Group 3: Total Iterations (e.g., "1034")
    # Group 4: mse avg loss (e.g., "10.455")
    # Group 5: kld avg loss (e.g., "325.103")
    # Group 6: sample avg loss (e.g., "9.543")
    # Group 7: total_loss avg loss (e.g., "345.100")
    pattern = re.compile(
        r"Epo: (\d+)/\d+, It: (\d+)/(\d+).*"  # Epoch and Iteration
        r"mse: [\d\.]+ \((\d+\.\d+)\).*"      # mse loss
        r"kld: [\d\.]+ \((\d+\.\d+)\).*"      # kld loss
        r"sample: [\d\.]+ \((\d+\.\d+)\).*"   # sample loss
        r"total_loss: [\d\.]+ \((\d+\.\d+)\)" # total_loss
    )

    # Use defaultdict to automatically create empty lists for new epochs
    # Updated keys for new losses
    epoch_data_accumulator = defaultdict(lambda: {
        'mse': [],
        'kld': [],
        'sample': [],
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
                    epoch, _, _, mse, kld, sample, total = match.groups()
                    epoch_num = int(epoch)
                    
                    # Accumulate all the "running average" values
                    # Updated to use new loss names
                    epoch_data_accumulator[epoch_num]['mse'].append(float(mse))
                    epoch_data_accumulator[epoch_num]['kld'].append(float(kld))
                    epoch_data_accumulator[epoch_num]['sample'].append(float(sample))
                    epoch_data_accumulator[epoch_num]['total_loss'].append(float(total))
    
    except FileNotFoundError:
        print(f"Error: Log file not found at '{log_file}'")
        return None
    except Exception as e:
        print(f"An error occurred during parsing: {e}")
        return None

    if not epoch_data_accumulator:
        print("No matching log entries found. Check your log file format.")
        return None

    # --- Calculate Averages ---
    print("Parsing complete. Calculating epoch averages...")
    
    # Updated keys for new losses
    plot_data = {
        'epochs': [],
        'mse': [],
        'kld': [],
        'sample': [],
        'total_loss': []
    }

    sorted_epochs = sorted(epoch_data_accumulator.keys())
    
    for epoch in sorted_epochs:
        plot_data['epochs'].append(epoch)
        
        losses = epoch_data_accumulator[epoch]
        
        # Calculate the mean for each new loss type
        plot_data['mse'].append(np.mean(losses['mse']))
        plot_data['kld'].append(np.mean(losses['kld']))
        plot_data['sample'].append(np.mean(losses['sample']))
        plot_data['total_loss'].append(np.mean(losses['total_loss']))

    print(f"Calculated averages for {len(plot_data['epochs'])} epochs.")
    return plot_data

def plot_epoch_losses(data):
    """
    Generates and saves a 2x2 grid of *per-epoch* average loss plots.
    (Updated for new losses)
    """
    if data is None:
        return

    print("Generating plots...")
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Average Training Loss per Epoch (BEV)', fontsize=16)
    
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

    # --- Plot 3: MSE Loss (NEW) ---
    axs[1, 0].plot(x_axis_data, data['mse'], 'go-', label='MSE Loss') # green
    axs[1, 0].set_title('MSE Loss')
    axs[1, 0].set_xlabel(x_label)
    axs[1, 0].set_ylabel('Avg. Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # --- Plot 4: Sample Loss (NEW) ---
    axs[1, 1].plot(x_axis_data, data['sample'], 'mo-', label='Sample Loss') # magenta
    axs[1, 1].set_title('Sample Loss')
    axs[1, 1].set_xlabel(x_label)
    axs[1, 1].set_ylabel('Avg. Loss')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_filename = 'epoch_loss_curves_new.png'
    plt.savefig(output_filename)
    print(f"Plot saved to {output_filename}")
    
    plt.show()

def main():
    # Set the name of your log file here
    log_file_name = 'log.txt' 
    
    if len(sys.argv) > 1:
        log_file_name = sys.argv[1]

    loss_data = parse_log_by_epoch(log_file_name)
    plot_epoch_losses(loss_data)

if __name__ == "__main__":
    main()