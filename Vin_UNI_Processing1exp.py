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
            if freq in frequency_data and all(frequency_data[freq][key] is not None for key in ['I_Echo', 'Q_Echo', 'I_NoEcho', 'Q_NoEcho']):
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
    pixel_values = [val[pixel[0], pixel[1]] if val is not None else None for val in values]

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
base_directory = 'C:/Users/anujb/Downloads' #directory with main experiment folders

healthy_exp = "lung_healthy-20240411T043210Z-001"
tumor_exp = 'Lung_tumor_redo-20240411T043207Z-001'


#load and process primary dataset
data_healthy = load_and_process_data(base_directory, healthy_exp)
data_tumor = load_and_process_data(base_directory, tumor_exp)

# calculate the derived dataset 

dr_healthy = calculate_derived_parameters(data_healthy)
dr_tumor = calculate_derived_parameters(data_tumor)

#%% VISUALIZE THE 2D MAPS at 1 frequency: F = 1859.5 MHz, which is 119th index: HEALTHY
#PLOT ALL 8 MAPS AT 119th index

#this frequency was selected by plotting individual pixel value (few location at tissue, and air) vs frequency, 
#and determining the frequency where the magnitude is maximum
#each frame represents different modes + post experiment air frame of param_to_plot

param_to_plot = 'RCOEF'
HealthyW1 = dr_healthy['Sample Frequency Sweep - Weight 1'][param_to_plot][119]
HealthyW2 = dr_healthy['Sample Frequency Sweep - Weight 2'][param_to_plot][119]
HealthyW3 = dr_healthy['Sample Frequency Sweep - Weight 3'][param_to_plot][119]
HealthyW1_r = dr_healthy['Sample Frequency Sweep - Weight 1 - repeat'][param_to_plot][119]
HealthyW2_r = dr_healthy['Sample Frequency Sweep - Weight 2 - repeat'][param_to_plot][119]
HealthyW3_r = dr_healthy['Sample Frequency Sweep - Weight 3 - repeat'][param_to_plot][119]
Healthy_PF = dr_healthy['Sample Frequency Sweep - Post sample'][param_to_plot][119]
Healthy_POST_AIR = dr_healthy['Air Frequency Sweep - Post sample'][param_to_plot][119]


plot_list_healthy = [HealthyW1,HealthyW2,HealthyW3,HealthyW1_r,HealthyW2_r, HealthyW3_r, Healthy_PF, Healthy_POST_AIR]
str_list_healthy = ['H_W1', 'H_W2', 'H_W3', 'H_W1_r', 'H_W2_r', 'H_W3_r', 'H_PS_FLUID', 'P_AIR']


vf.multi_image_ROWCOL(plot_list_healthy, str_list_healthy, 1, 8, vmin=0.75, vmax=0.97, cmap='Spectral') 

#%% TUMOR
param_to_plot = 'RCOEF'
TumorW1 = dr_tumor['Sample Frequency Sweep - Weight 1'][param_to_plot][119]
TumorW2 = dr_tumor['Sample Frequency Sweep - Weight 2'][param_to_plot][119]
TumorW3 = dr_tumor['Sample Frequency Sweep - Weight 3'][param_to_plot][119]
TumorW1_r = dr_tumor['Sample Frequency Sweep - Weight 1 - repeat'][param_to_plot][119]
TumorW2_r = dr_tumor['Sample Frequency Sweep - Weight 2 - repeat'][param_to_plot][119]
TumorW3_r = dr_tumor['Sample Frequency Sweep - Weight 3 - repeat'][param_to_plot][119]
Tumor_PF = dr_tumor['Sample Frequency Sweep - Post sample'][param_to_plot][119]
Tumor_POST_AIR = dr_tumor['Air Frequency Sweep - Post sample'][param_to_plot][119]

plot_list_tumor = [TumorW1,TumorW2,TumorW3,TumorW1_r,TumorW2_r, TumorW3_r, Tumor_PF, Tumor_POST_AIR]
str_list_tumor = ['T_W1', 'T_W2', 'T_W3', 'T_W1_r', 'T_W2_r', 'T_W3_r', 'T_PS_FLUID', 'P_AIR']

vf.multi_image_ROWCOL(plot_list_tumor, str_list_tumor, 1, 8, vmin=0.75, vmax=0.97, cmap='Spectral') 

#%% PLOTTING CHANGE OF TISSUE IMAGES WITH LARGER LOAD

W2r_W1r_healthy = HealthyW2_r - HealthyW1_r
W2r_W1r_tumor = TumorW2_r - TumorW1_r


#plot_list = [W3_W1r_healthy, W3_W1r_tumor, W3_W1r_mixed]
plot_list = [W2r_W1r_healthy,W2r_W1r_tumor ]
str_list = ['W2_r - W1_r: Healthy', 'W2_r - W1_r: Tumor', 'W2_r - W1_r: Mixed']

vf.multi_image(plot_list, str_list, vmin=-0.2, vmax=0.2, cmap='Spectral')
#%%
W3r_W1r_healthy = HealthyW3_r - HealthyW1_r
W3r_W1r_tumor = TumorW3_r - TumorW1_r


#plot_list = [W3_W1r_healthy, W3_W1r_tumor, W3_W1r_mixed]
plot_list = [W3r_W1r_healthy,W3r_W1r_tumor]
str_list = ['W3_r - W1_r: Healthy', 'W3_r - W1_r: Tumor', 'W3_r - W1_r: Mixed']

vf.multi_image(plot_list, str_list, vmin=-0.2, vmax=0.2, cmap='Spectral')

#%%
#ANALYZING THE TISSUE AREA

# Load your image data
images_healthy, images_tumor= plot_list_healthy[:6], plot_list_tumor[:6]

# Calculate and normalize tissue areas under different loads
areas_healthy, normalized_area_healthy, masks_healthy = calculate_tissue_area(images_healthy)
areas_tumor, normalized_area_tumor, masks_tumor = calculate_tissue_area(images_tumor)

# Prepare plots
# Load labels and titles
load_labels = ['50', '100', '200', '50 repeat', '100 repeat', '200 repeat']
titles = ['Healthy Tissue', 'Tumor Tissue', 'Mixed Tissue']
load_labels = ['0.49', '0.98', '1.96'] #in Newtons
loads = ['0.49','0.98','1.96', '0.49','0.98','1.96'] #in Newtons


#%%PLOTTING THE MASKS first to see if it represents the tissue area well by checking original RCOEF maps
#The goal of this is to make sure we are only selecting the tissue region for accurate calculation of 2D area

#for healthy
vf.multi_image_ROWCOL(masks_healthy, str_list_healthy, 1, 6, vmin=0.75, vmax=0.97, cmap='Wistia') 

#%% for tumor
vf.multi_image_ROWCOL(masks_tumor, str_list_tumor, 1, 6, vmin=0.75, vmax=0.97, cmap='Wistia') 


#%%then plot the normalized area vs loads
#this was done for the repeated experiments only as the first trial had large non-contactign area

#plt.subplot(1, 2, 1)
plt.plot(load_labels[:3], areas_healthy[3:]/areas_healthy[3],  'o--', markersize = 10, label='Healthy Tissue', color='seagreen', linewidth = 3)
plt.plot(load_labels[:3], areas_tumor[3:]/areas_tumor[3],  'D--', markersize = 10,  label='Tumor Tissue', color='indianred', linewidth = 3)
#plt.plot(load_labels, areas_mixed,  'o-', label='Mixed Tissue', color='gold', linewidth = 3)
plt.title('Tissue area changing with load')
plt.xlabel('Load (N)')
plt.ylabel(r'Normalized Area ($mm^2$)')
plt.legend()
plt.grid(True)

plt.xticks(rotation = 25)

#%%
#plotting the actual area
#plt.subplot(1, 2, 2)
plt.plot(load_labels[:3], normalized_area_healthy[3:], 'o--', markersize = 10, label='Healthy Tissue', color='seagreen', linewidth = 3)
plt.plot(load_labels[:3], normalized_area_tumor[0]+normalized_area_tumor[3:], 'D--', markersize = 10, label='Tumor Tissue', color='indianred', linewidth = 3)
#plt.plot(load_labels, normalized_area_mixed, 'o-', label='Mixed Tissue', color='gold', linewidth = 3)
plt.title('Tissue area changing with load')
plt.xlabel('Load (N)')
plt.ylabel(r'Area ($mm^2$)')
plt.legend()
plt.grid(True)

plt.xticks(rotation = 25)
plt.tight_layout()
plt.show()

#%% COMPARING TISSUES WITH FREQUENCY EXAMPLE WITH PHASE

H_RCOEF_W3 = average_reflection_at_frequencies(dr_healthy, 'Phase_adj', 'Sample Frequency Sweep - Weight 3 - repeat', masks_healthy[5])

T_RCOEF_W3 = average_reflection_at_frequencies(dr_tumor, 'Phase_adj', 'Sample Frequency Sweep - Weight 3 - repeat', masks_tumor[5])


plt.plot(freqs, abs(H_RCOEF_W3), label = "Healthy 1.96N load", color = 'indianred', linewidth = 3)
plt.plot(freqs, abs(T_RCOEF_W3), label = "Tumor 1.96N load",  color = 'seagreen', linewidth = 3)

plt.title(r'Average Phase vs frequency')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Phase (radians)')
plt.legend()


