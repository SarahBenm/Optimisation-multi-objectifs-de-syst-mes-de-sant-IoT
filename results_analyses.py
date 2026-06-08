import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns


def extract_info_from_path(file_path):
    # Normalize path to handle both forward and backward slashes
    normalized_path = os.path.normpath(file_path)

    # Split the normalized path
    parts = normalized_path.split(os.sep)

    # Number of clouds from folder name (e.g., '10clouds')
    cloud_info = [part for part in parts if "clouds" in part][0]
    number_of_clouds = int(cloud_info.replace("clouds", ""))

    # Algorithm name (e.g., 'pcn' or 'mpmoql')
    algo_name = parts[-2]

    # Comp number (e.g., 'comp_5')
    comp_info = parts[-1]
    comp_number = comp_info.split("_")[2].replace(".csv", "")

    return algo_name, number_of_clouds, comp_number


def generate_file_paths(base_dir, algorithms, cloud_numbers, comp_numbers):
    file_paths = []
    # Use itertools.product to generate all combinations of algorithms, cloud numbers, and comp numbers
    for algo, cloud, comp in itertools.product(algorithms, cloud_numbers, comp_numbers):
        file_path = os.path.join(
            base_dir, f"{cloud}clouds", algo, f"{algo}_comp_{comp}.csv"
        )
        file_paths.append(file_path)
    return file_paths


def calculate_and_save_averages(file_paths, output_file):
    all_results = []  # List to accumulate results

    for file_path in file_paths:
        # Extract algorithm name, number of clouds, and comp number
        algo_name, number_of_clouds, comp_number = extract_info_from_path(file_path)

        # Read the CSV file into a DataFrame, skip if the file doesn't exist
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        df = pd.read_csv(file_path)

        # Select the columns to calculate averages
        numeric_columns = [
            "HV",
            "Sparsity",
            "Expected Utility",
            "Card",
            "IGD",
            "MUL",
            "execution_time",
        ]

        # Calculate the average for each column
        averages = df[numeric_columns].mean()

        # Add the algorithm name, number of clouds, and comp number to the results
        result = averages.to_frame().T  # Convert Series to DataFrame
        result["Algorithm"] = algo_name
        result["Number of Clouds"] = number_of_clouds
        result["Comp"] = comp_number  # Store the actual composition number

        # Append the result to the list
        all_results.append(result)

    # Concatenate all the results into one DataFrame
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)

        # Save the final results to the output CSV file
        final_results.to_csv(output_file, index=False)
    else:
        print("No results to save.")


# Define the parameters
base_dir = "./results/Execution_results/"  # Base directory where results are stored
algorithms = ["mpmoql", "pcn", "envelope"]
cloud_numbers = [5, 10, 20]
comp_numbers = [3, 5, 8]

# Generate file paths based on all possible combinations
file_paths = generate_file_paths(base_dir, algorithms, cloud_numbers, comp_numbers)

# Define the output file
output_file = "./results/output_averages/all_averages.csv"

# Calculate and save averages
calculate_and_save_averages(file_paths, output_file)

# Read the CSV data into a DataFrame
data = pd.read_csv(output_file)

# List of metrics you want to plot
metrics = ["HV", "Sparsity", "Expected Utility", "Card", "IGD", "MUL", "execution_time"]

# Get the unique 'Comp' values
comp_values = data["Comp"].unique()

# Loop through each metric
for metric in metrics:
    # Create a directory for each metric if it doesn't exist
    if not os.path.exists(f"./results/graphs/{metric}"):
        os.makedirs(f"./results/graphs/{metric}")

    # Loop through each 'Comp' value to create individual plots for each metric
    for comp in comp_values:
        plt.figure(figsize=(10, 6))

        # Filter the data for the current 'Comp' value
        comp_data = data[data["Comp"] == comp]

        # Use seaborn to create a bar plot
        sns.barplot(
            x="Number of Clouds",
            y=metric,  # Change y-axis to the current metric
            hue="Algorithm",
            data=comp_data,
            palette="viridis",
        )

        # Set the title and labels
        plt.title(f"{metric} vs Number of Clouds for Comp {comp}")
        plt.xlabel("Number of Clouds")
        plt.ylabel(metric)

        # Show the legend for algorithms
        plt.legend(title="Algorithm")

        # Define the filename and save the figure
        filename = os.path.join(
            f"./results/graphs/{metric}/", f"{metric}_Comp_{comp}.png"
        )
        plt.savefig(filename)

        # Close the plot to avoid display
        plt.close()

print("Graphs have been successfully saved.")
