# Step 1: Initialize an Associative Array for Defaults
# Define your default arguments in an associative array:
declare -A defaults=(
    [--source_url]="XURLs.LF_AMAZON_131K"
    [--all_pairs_file_path]="all_pairs.txt"
    [--labels_file_path]="lbl.json"
    [--train_file_path]="trn.json"
    [--test_file_path]="tst.json"
)

declare -A args=()  # This array will store the final arguments

# Step 2: Parse Command-Line Arguments
# Loop through the command-line arguments and store them in the args associative array, allowing command-line values to overwrite defaults:
while [[ $# -gt 0 ]]; do
    case "$1" in
        --*=*)
            key="${1%%=*}"  # Extract the key (everything before '=')
            value="${1#*=}"  # Extract the value (everything after '=')
            args[$key]="$value"  # Assign to the array, overwriting the default if it exists
            shift
            ;;
        --*)
            key="$1"
            # Check if the next argument is another flag or a value
            if [[ "$2" == --* ]] || [[ -z "$2" ]]; then
                args[$key]=""  # No value after the flag, assume flag-only
            else
                args[$key]="$2"  # Assign the value to the key in the args
                shift
            fi
            shift
            ;;
        *)
            echo "Warning: Ignoring unrecognized argument format: $1"
            shift
            ;;
    esac
done

# Step 3: Merge Command-Line Arguments with Defaults
# After parsing the command-line arguments, merge them with the defaults to ensure all necessary parameters are set:
# Merge defaults with args, defaults only used if args does not already have the key
for key in "${!defaults[@]}"; do
    if [[ -z "${args[$key]}" ]]; then  # If key is not already set by command-line args
        args[$key]="${defaults[$key]}"
    fi
done

# Step 4: Prepare Arguments for Python Script
# Finally, you'll want to construct the list of arguments to pass to the Python script:
final_args=()
for key in "${!args[@]}"; do
    if [[ -z "${args[$key]}" ]]; then  # Check if the value for this key is empty
        final_args+=("$key")  # Add only the key as a flag
    else
        final_args+=("$key=${args[$key]}")  # Add key=value
    fi
done

# Print all arguments to check
echo "All arguments:"
for arg in "${final_args[@]}"; do
    echo "$arg"
done

# Step 5: Run the Python Script with Constructed Arguments
# Set environment variables if needed

# Run the Python script with all arguments
python ./process_xmlzeroshot.py "${final_args[@]}"

# Unset environment variables after the script runs
