import csv
import matplotlib.pyplot as plt

# Read the CSV file using the csv module
files = ["seq.csv", "omp_frag.csv", "omp_symm.csv", "mpi_frag.csv", "mpi_symm.csv"]
for file in files:
    with open(file, 'r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Initialize dictionaries to store data for each option
        option_data = {}

        # Process each row in the CSV file
        for row in csvreader:
            option = row['Option']
            dataset = row['Dataset']
            execution_time = float(row['ExecutionTime'])

            # Initialize lists for each option if not already present
            if option not in option_data:
                option_data[option] = {
                    'datasets': [],
                    'execution_times': []
                }

            option_data[option]['datasets'].append(dataset)
            option_data[option]['execution_times'].append(execution_time)

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Plot each option as a separate line
    for option, data in option_data.items():
        datasets = data['datasets']
        execution_times = data['execution_times']
        if option == "-n":
            plt.plot(datasets, execution_times, label=f'nested join')
        elif option == "-m":
            plt.plot(datasets, execution_times, label=f'merge join')
        elif option == "-h":
            plt.plot(datasets, execution_times, label=f'hash join')
            
            

    # Customize the plot
    plt.title('Execution Times for Different Options')
    plt.xlabel('Dataset')
    plt.ylabel('Execution Time(ms)')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.savefig("{} graph.png".format(file[:-4]))
