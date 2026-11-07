#!/bin/bash

# Function to calculate required GPL steps for a desired average entries per query
calculate_required_steps() {
    local gpl_batch_size=$1
    local num_train_queries=$2
    local desired_average_entries=$3

    # Calculate required GPL steps
    local size_gpl_train_data=$(echo "scale=0; $desired_average_entries * $num_train_queries" | bc)
    local gpl_steps=$(echo "scale=0; $size_gpl_train_data / $gpl_batch_size" | bc)

    echo "To achieve an average of $desired_average_entries entries per query with $num_train_queries queries, you should use approximately $gpl_steps GPL steps."
}

# Check for required arguments
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <num_train_queries> <desired_average_entries>"
    exit 1
fi

# Read user inputs
num_train_queries=$1
desired_average_entries=$2

# Fixed parameter
gpl_batch_size=32

# Call the function with inputs
calculate_required_steps "$gpl_batch_size" "$num_train_queries" "$desired_average_entries"

