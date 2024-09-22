# -*- coding: utf-8 -*-
"""
Created on Wed Jul 1 04:43:45 2024

@author: ab
"""
#%% importing modules
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import cv2 
from scipy.ndimage import median_filter
import Vin_UNI_HFs as vf

#%% PARAMETERS
freqs = np.arange(1800,1880,0.5) #range of frequency of acquired dataset

#%%
#FUNCTIONS to process and analyze datasets
def extract_frequency(filename):
    parts = filename.split('_')
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    raise ValueError(f"Could not extract frequency from filename: {filename}")

def load_and_process_data(base_directory, experiment):
    
    """LOADS the raw dat files from the given directory {base_directory} and 
    derives the respective In-phase and Out-of-phase matrices. This includes I echo,
    I no-echo, Q echo, and Q no-echo. These matrices for all the frequencies are stored
    in modes_data. Furthermore, the magnitdue, and phase are also computed and stored 
    in modes_data.
    
    
    """
    experiment_path = os.path.join(base_directory, experiment)
    modes_data = {}

    for mode_folder in os.listdir(experiment_path):
        mode_path = os.path.join(experiment_path, mode_folder)
        frequency_data = {}
        if os.path.isdir(mode_path):
            for timestamp in os.listdir(mode_path):
                timestamp_path = os.path.join(mode_path, timestamp)
                for data_file in glob(os.path.join(timestamp_path, '*.dat')):
                    try:
                        frequency = extract_frequency(data_file)
                    except ValueError as e:
                        print(f"Error extracting frequency from {data_file}: {e}")
                        continue

                    try:
                        _, _, i_volts, q_volts = vf.loadSavedRawData(data_file)
                        if i_volts is None or q_volts is None:
                            print(f"Warning: No data loaded from {data_file}.")
                            continue
                    except Exception as e:
                        print(f"Error loading data from {data_file}: {e}")
                        continue

                    if frequency not in frequency_data:
                        frequency_data[frequency] = {'I_Echo': None, 'Q_Echo': None, 'I_NoEcho': None, 'Q_NoEcho': None}

                    data_type = "Echo" if "Frequencyecho_" in data_file else "NoEcho"
                    frequency_data[frequency][f'I_{data_type}'] = i_volts
                    frequency_data[frequency][f'Q_{data_type}'] = q_volts

        # Calculate magnitude and phase for each frequency in the specified range
        data_store = {
            'frequencies': [],
            'Magnitude': [],
            'Phase': []
        }

        for freq in freqs:  # Ensuring we cover the complete range
            if freq in frequency_data and all(frequency_data[freq][key] is not np.nan for key in ['I_Echo', 'Q_Echo', 'I_NoEcho', 'Q_NoEcho']):
                freq_data = frequency_data[freq]
                i_echo = np.array(freq_data['I_Echo'])
                q_echo = np.array(freq_data['Q_Echo'])
                i_noecho = np.array(freq_data['I_NoEcho'])
                q_noecho = np.array(freq_data['Q_NoEcho'])

                delta_i = i_echo - i_noecho
                delta_q = q_echo - q_noecho

                magnitude = np.sqrt(np.square(delta_i) + np.square(delta_q))
                phase = np.arctan2(delta_q, delta_i)

                data_store['frequencies'].append(freq)
                data_store['Magnitude'].append(magnitude)
                data_store['Phase'].append(phase)
            else:
                print(f"Missing complete data set for frequency: {freq}")
                data_store['frequencies'].append(freq)
                data_store['Magnitude'].append(None)
                data_store['Phase'].append(None)

        modes_data[mode_folder] = data_store

    return modes_data

def calculate_derived_parameters(data):
    
    """
    This function is used to calculate baseline adjusted Magnitdue and Phase from the
    processed dataset that is obtained using load_and_process_data function. It isolates the AIR/Baseline dataset and uses this to compute
    the baseline adjusted parameters for all the MODES, across all frequency frames. 
    
    It then computes the Reflection coefficient for each MODE, and Acoustic Impedance
    
    This is stored in derived_data. 
    """
    
    air_data = data['Air Frequency Sweep']
    derived_data = {}

    for mode, mode_data in data.items():
        if mode == 'Air Frequency Sweep':
            continue  # Skip the air mode data

        frequencies = mode_data['frequencies']
        RCOEF, Phase_adj, Mag_adj, Impedance = [], [], [], []

        for idx, freq in enumerate(frequencies):
            # Fetch corresponding index from air data (assuming same frequency order)
            air_idx = air_data['frequencies'].index(freq)
            if mode_data['Magnitude'][idx] is None:
                continue
                
            # Ensure all calculations use high-precision floats
            magnitude_sample = np.array(mode_data['Magnitude'][idx], dtype=np.float64)
            magnitude_air = np.array(air_data['Magnitude'][air_idx], dtype=np.float64)

            # Calculate reflection coefficient safely
            rcoef = np.divide(magnitude_sample, magnitude_air, out=np.zeros_like(magnitude_sample), where=magnitude_air!=0)
            RCOEF.append(rcoef)

            phase_sample = np.array(mode_data['Phase'][idx], dtype=np.float64)
            phase_air = np.array(air_data['Phase'][air_idx], dtype=np.float64)
            phase_adj = phase_sample - phase_air
            Phase_adj.append(phase_adj)

            mag_adj = magnitude_sample - magnitude_air
            Mag_adj.append(mag_adj)

            # Assuming mf.impedance_si is a method that calculates impedance
            impedance = vf.impedance_si(rcoef, array=True)
            Impedance.append(impedance)

        derived_data[mode] = {
            'frequencies': frequencies,
            'RCOEF': RCOEF,
            'Phase_adj': Phase_adj,
            'Mag_adj': Mag_adj,
            'Impedance': Impedance
        }

    return derived_data


def plot_parameter_vs_frequency_at_pixel(derived_data, mode, parameter, pixel=(64, 64)):
    if mode not in derived_data:
        print(f"Mode '{mode}' not found in the data.")
        return

    frequencies = derived_data[mode]['frequencies']
    values = derived_data[mode][parameter]

    # Extract values for the specific pixel, assuming values are stored in 2D array format for each frequency
    pixel_values = [val[pixel[0], pixel[1]] if val is not np.nan else None for val in values]

    # Plotting
    
    plt.plot(frequencies, pixel_values, linestyle='-',label = mode)
    plt.title(f'{parameter} at Pixel {pixel} vs Frequency for {mode}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(parameter)
    plt.legend()
    plt.grid(True)
    plt.show()
    

def fast_plot_and_video(matrix_list, parameter_label, mode, directory, vmin, vmax):
    # Create directories for saving outputs
    folder_name = f"{parameter_label.replace(' ', '_')}"
    mode_directory = os.path.join(directory, mode, folder_name)
    os.makedirs(mode_directory, exist_ok=True)
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 6))
    image_files = []

    # Plot and save images
    for idx, matrix in enumerate(matrix_list):
        if matrix is None:
            continue
        ax.clear()
        freq = 1800 + idx * 0.5
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap='Spectral_r', interpolation = 'bilinear')
        ax.set_title(f"{parameter_label} {freq:.1f} MHz")
        ax.axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Save the frame to file
        frame_filename = os.path.join(mode_directory, f"{folder_name}_{freq:.1f}MHz.png")
        plt.savefig(frame_filename, bbox_inches='tight', format='png')
        image_files.append(frame_filename)
        cbar.remove()

    plt.close(fig)

    # Create video from images, ensuring no color conversion issues
    if image_files:
        first_frame = cv2.imread(image_files[0], cv2.IMREAD_COLOR)  # Read the first image to determine size
        height, width, _ = first_frame.shape
        video_filename = os.path.join(mode_directory, f"{folder_name}.avi")
        video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'DIVX'), 7, (width, height))

        for image_file in image_files:
            img = cv2.imread(image_file, cv2.IMREAD_COLOR)  # Read in RGB format directly
            video_writer.write(img)  # Add the image to the video as is

        video_writer.release()

    print(f"Images and video saved in {mode_directory}")

    
    
def plot_parameter_for_mode(data, mode, parameter, vmin, vmax):
    parameter_labels = {
        'Magnitude_adj': 'Magnitude',
        'Phase_adj': 'Phase',
        'RCOEF': 'Reflection Coefficient',
        'Impedance': 'Acoustic Impedance'
    }
    
    parameter_label = parameter_labels.get(parameter, parameter)
    
    # Ensure the mode and parameter are valid
    if mode not in data or parameter not in data[mode]:
        print(f"Data for mode '{mode}' and parameter '{parameter}' not available.")
        return
    
    matrix_list = data[mode][parameter]
    if matrix_list[0] is None:
        print(f"No data available for {parameter} in mode {mode}.")
        return
    
    directory = 'C:/Users/anujb/Downloads/HealthyLUNG/'  # Base directory for output
    fast_plot_and_video(matrix_list, parameter_label, mode, directory, vmin, vmax)
    
#FUNCTIONS FOR POST PROCESSING 
#AREA OF TISSUE


# Calculate the area by thresholding and counting pixels

#     """Calculate the tissue areas where reflection coefficients are below the threshold."""
#     return [np.sum(image < threshold) for image in images]

def calculate_tissue_area(images, threshold=0.96, pixel_size=50):
    filter_size = 3
    """Calculate the tissue areas where reflection coefficients are below the threshold.
    0.96 is set to separate air pixels from the tissue pixels """
    # Apply median filter to reduce noise
    filtered_images = [median_filter(image, size=filter_size) for image in images]
    # Calculate area in pixels based on the threshold
    areas_pixels = [(filtered_image < threshold).sum() for filtered_image in filtered_images]
    # Convert area from pixels to square millimeters (each pixel is 50 microns)
    areas_mm2 = [area * (pixel_size * 1e-3)**2 for area in areas_pixels]
    # Normalize areas relative to the first area (baseline normalization)
    normalized_areas = [area / areas_mm2[0] for area in areas_mm2]
    # Create binary masks for visualization
    masks = [filtered_image < threshold for filtered_image in filtered_images]
    return areas_mm2, normalized_areas, masks


def extract_masked_values_at_frequency(dataset, parameter, experiment, mask, frequency_index):
    """
    Extracts all the reflection coefficients within a masked area for a specified frequency.
    
    Parameters:
        dataset (dict): A dictionary containing nested dictionaries and numpy arrays representing the data.
        parameter (str): The key for the parameter of interest in the dataset.
        experiment (str): The key for the experiment in the dataset.
        mask (numpy.ndarray): A 2D boolean array of shape (height, width) where True values indicate the desired area.
        frequency_index (int): The index of the frequency layer to analyze.
        
    Returns:
        list: A list of reflection coefficients within the masked area for the specified frequency.
    """
    
    
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    reflection_data = dataset[experiment][parameter]

    # Access the specific frequency layer
    frequency_layer = reflection_data[frequency_index]

    # Extract values where the mask is True
    masked_values = frequency_layer[mask]

    return list(masked_values)

def average_reflection_at_frequencies(dataset, parameter, experiment, mask):
    """
    Calculate the average reflection coefficient within a masked area for each frequency.
    
    Parameters:
        reflection_data (numpy.ndarray): A 3D array of shape (num_frequencies, height, width)
                                         containing reflection coefficient data.
        mask (numpy.ndarray): A 2D boolean array of shape (height, width) where True values
                              indicate the tissue area.
                              
    Returns:
        numpy.ndarray: An array of average reflection coefficients for each frequency.
    """
    # Ensure the mask is boolean
    mask = mask.astype(bool)
    reflection_data = dataset[experiment][parameter]
    # Calculate the average within the mask for each frequency layer
    averages = []
    for frequency_layer in reflection_data:
        masked_data = frequency_layer[mask]
        if masked_data.size > 0:
            average = np.mean(masked_data)
        else:
            average = np.nan  # Avoid division by zero if mask is empty for some reason
        averages.append(average)
    
    return np.array(averages)
#%% EXAMPLE OF PROCESSING, functions usage. SHOWN FOR 2 EXAMPLES directory

# Usage
def convert_2_Image(data, attr_name, output_path, keys='RCOEF', cmap='Spectral'):
    print("CONVERT DATA TO IMAGE")
    for attr in attr_name:
        dataAttr = data[attr][keys]
        lenFrequency = len(dataAttr)
        for frequency in range(lenFrequency):
            dataImg = dataAttr[frequency]
            fileNameOutput = os.path.join(output_path, attr)
            if not os.path.exists(fileNameOutput):
                os.makedirs(fileNameOutput)
            fig = plt.imshow(dataImg, cmap=cmap, vmin=0.75, vmax=0.97)
            plt.axis('off')
            plt.savefig(os.path.join(fileNameOutput, str(data[attr]['frequencies'][frequency])+'.jpg'))

def main():
    base_directory = './Cornell_Vinmec_Data/Pilot_240409/' #directory with main experiment folders
    attr_name = ['Sample Frequency Sweep - Weight 1',
                 'Sample Frequency Sweep - Weight 2',
                 'Sample Frequency Sweep - Weight 3',
                 'Sample Frequency Sweep - Weight 1 - repeat',
                 'Sample Frequency Sweep - Weight 2 - repeat',
                 'Sample Frequency Sweep - Weight 3 - repeat',
                 'Sample Frequency Sweep - Post sample',
                 'Air Frequency Sweep - Post sample'
                ]
    typeData = "lung_healthy"
    output_path = '/Users/chivinhduong/Project/TissueImaging-VINUNI/output'
    #load and process primary dataset
    data = load_and_process_data(base_directory, typeData)
    # calculate the derived dataset 
    dataDE = calculate_derived_parameters(data)
    convert_2_Image(dataDE, attr_name, output_path)

if __name__ == '__main__':
    main()