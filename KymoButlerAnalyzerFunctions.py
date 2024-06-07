import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import math

'''
KymoGrid Functions: Make grid of all the kymograph images
'''

# Function to resize images to 1024x512
def resize_function_rectangle(image):
    return image.resize((2048, 512), Image.BICUBIC)

# Function to resize images based on their aspect ratio
def resize_images(images_dir, overlay, brightness_factor):
    resized_images = []
    resize_functions = []  # Keep track of which resize function was used for each image
    
    if overlay:
        for filename in os.listdir(images_dir):
            if filename.endswith('.png'):
                img_path = os.path.join(images_dir, filename)
                img = Image.open(img_path)
                img = adjust_brightness(img, brightness_factor)
                resized_img = resize_function_rectangle(img)
                resize_functions.append("rectangle")
                resized_images.append(resized_img)
    
    else:
        for root, dir, files in os.walk(images_dir):
            for filename in files:
                if filename.endswith("Kymograph.png"):
                    img_path = os.path.join(root, filename)
                    img = Image.open(img_path)

                    # Adjust brightness for certain channels
                    if "Kymograph" in filename:
                        img = adjust_brightness_gray(img, brightness_factor)
                    
                    resized_img = resize_function_rectangle(img)
                    resize_functions.append("rectangle")
                    resized_images.append(resized_img)

    return resized_images, resize_functions

# Function to adjust brightness of an image without changing colors
def adjust_brightness_gray(image, brightness_factor):
    # Convert image to grayscale
    grayscale_image = image.convert('L')

    # Adjust brightness by multiplying pixel values by the brightness factor
    adjusted_image = Image.eval(grayscale_image, lambda x: x * brightness_factor)

    # Convert the adjusted grayscale image back to RGB
    return adjusted_image.convert('RGB')

# Function to adjust brightness of an image while maintaining colors
def adjust_brightness(image, brightness_factor):
    # Apply brightness factor to each pixel in the image
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness_factor)




# Function to plot images in a grid while maintaining aspect ratio
def plot_images(images, resize_functions, output_dir, num_cols=4, title="Image Grid"):
    num_images = len(images)
    num_rows = math.ceil(num_images / num_cols)

    # Calculate figure size based on the number of columns
    fig_width = 40
    fig_height = fig_width / num_cols 

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axs.flat):
        if i < num_images:
            img = images[i]
            resize_function = resize_functions[i]
            ax.imshow(img)
            ax.axis('off')
            if resize_function == "square":
                ax.set_aspect('auto')  # Set aspect ratio to 'auto' for square images
        else:
            ax.axis('off')  # Hide the subplot if there are fewer images than subplots

    fig.suptitle(title, fontsize=70, wrap=True)  # Enable text wrapping for the title
    save_path = os.path.join(output_dir, title + ".pdf")
    plt.savefig(save_path)

# Make grid of all the kymos
def kymogrid(images_dir, output_dir, title, overlay, brightness_factor=3):
    resized_images, resize_functions = resize_images(images_dir, overlay, brightness_factor)
    plot_images(resized_images, resize_functions, output_dir, title=title)

'''
Create Grid of All Plots
'''
# Image grid for plots
def imagegrid(title, imagename, directory, num_cols=5):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(imagename):
            img_path = os.path.join(directory, filename)
            img = Image.open(img_path)
            images.append(img)
    
    num_images = len(images)
    if num_images == 0:
        print(f"No images found with the suffix '{imagename}' in directory '{directory}'")
        return
    
    num_rows = math.ceil(num_images / num_cols)
    
    # Calculate figure size based on the number of columns
    fig_width = 60
    fig_height = fig_width / num_cols * num_rows
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.subplots_adjust(hspace=.2, wspace=0.15)
    
    for i, ax in enumerate(axs.flat):
        if i < num_images:
            img = images[i]
            ax.imshow(img)
            ax.axis('off')
            ax.set_aspect('auto')  # Set aspect ratio to 'auto' for all images
        else:
            ax.axis('off')  # Hide the subplot if there are fewer images than subplots
    
    fig.suptitle(title, fontsize=70, wrap=True)  # Enable text wrapping for the title
    save_path = os.path.join(directory, title + ".pdf")
    plt.savefig(save_path)
    plt.show()
    plt.close()

'''
Determine Bins for Histograms
'''
# Function to calculate the number of bins using different methods (can edit to choose different methods)
def calculate_bins(data, method='sqrt'):
    data = data[~np.isnan(data)]  # Remove NaN values
    n = len(data)
    if n == 0:
        return 1  # Default to 1 bin if no data left after removing NaNs
    if method == 'sqrt':
        return int(np.sqrt(n))
    elif method == 'sturges':
        return int(np.log2(n) + 1)
    elif method == 'rice':
        return int(2 * (n ** (1 / 3)))
    elif method == 'fd':  # Freedman-Diaconis
        q25, q75 = np.percentile(data, [25, 75])
        bin_width = 2 * (q75 - q25) / (n ** (1 / 3))
        if bin_width == 0 or np.isnan(bin_width):
            return 1  # Return 1 bin if bin_width is 0 or NaN
        return max(1, int((data.max() - data.min()) / bin_width))  # Ensure at least 1 bin
    else:
        raise ValueError("Unknown method")
    

'''
Extract and compile data from kymobutler results files
'''
# Go through and process all the KymoButler results files, extracting each data metric by direction and counting number of tracks per direction
def process_data(path, conditionnames, dfs, columnmapping, results_folder, grid_folder, kymo_grid, row_number=0):
    for condition in conditionnames:
        condition_path = os.path.join(path, condition)

        print(f"  Processing condition: {condition}")

        directory_name = os.path.basename(path)
        if kymo_grid:
            titlegrid = f'{directory_name}_{condition}'
            kymogrid(condition_path, grid_folder, titlegrid, overlay=False)

            overlaytitlegrid=f'{directory_name}_{condition}_overlay'
            overlayfolder= os.path.join(condition_path, 'kymo_overlays')
            kymogrid(overlayfolder, grid_folder, overlaytitlegrid, overlay=True, brightness_factor=2)

        for filename in os.listdir(condition_path):
            if filename.endswith(('.xlsx', '.xls')):
                filepath = os.path.join(condition_path, filename)
                df = pd.read_excel(filepath)

                if 'Direction' not in df.columns:
                    print(f"Warning: 'Direction' column not found in {filename}. Skipping this file.")
                    continue

                # Calculate number of tracks per direction
                total_tracks = len(df)
                anterograde_tracks = len(df[df['Direction'] == 1])
                retrograde_tracks = len(df[df['Direction'] == -1])
                stationary_tracks = len(df[df['Direction'] == 0])

                percent_anterograde = (anterograde_tracks / total_tracks) * 100 if total_tracks > 0 else 0
                percent_retrograde = (retrograde_tracks / total_tracks) * 100 if total_tracks > 0 else 0
                percent_stationary = (stationary_tracks / total_tracks) * 100 if total_tracks > 0 else 0

                directions_data = {
                    'anterograde tracks': [anterograde_tracks],
                    'retrograde tracks': [retrograde_tracks],
                    'stationary tracks': [stationary_tracks],
                    'total tracks': [total_tracks],
                    'percent anterograde': [percent_anterograde],
                    'percent retrograde': [percent_retrograde],
                    'percent stationary': [percent_stationary]
                }
                
                directions_df = pd.DataFrame(directions_data)
                dfs[condition][f'{condition}_directions'].loc[row_number] = directions_df.iloc[0]
                row_number += 1

                #Extract each metric by direction
                for direction, label in zip([1, -1, 0], ['ANT', 'RET', 'STAT']):
                    df_direction = df[df['Direction'] == direction]
                    for column, suffix in columnmapping.items():
                        if column in df_direction.columns:
                            dataframe_key = f'{condition}_{label}_{suffix}'
                            column_data = df_direction[column].reset_index(drop=True)
                        
                            if dfs[condition][dataframe_key].empty:
                                dfs[condition][dataframe_key] = column_data.to_frame()
                            else:
                                dfs[condition][dataframe_key] = pd.concat([dfs[condition][dataframe_key], column_data], axis=1)

                # Combine all directions for TOTAL dataframes
                for column, suffix in columnmapping.items():
                    if column in df.columns:
                        dataframe_key = f'{condition}_TOTAL_{suffix}'
                        column_data = df[column].reset_index(drop=True)

                        if dfs[condition][dataframe_key].empty:
                            dfs[condition][dataframe_key] = column_data.to_frame()
                        else:
                            dfs[condition][dataframe_key] = pd.concat([dfs[condition][dataframe_key], column_data], axis=1)

    # Save all compiled data to excel sheets
    for condition in conditionnames:
        output_filepath = os.path.join(results_folder, f'{condition}_compiled_results.xlsx')
        with pd.ExcelWriter(output_filepath) as writer:
            for key, df in dfs[condition].items():
                if 'direction' not in key:
                    df.columns = [f'File_{i+1}' for i in range(df.shape[1])]
                    sheet_name = key.replace(f'{condition}_', '')
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    sheet_name = key.replace(f'{condition}_', '')
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Dataframes created and saved successfully!")
    return row_number