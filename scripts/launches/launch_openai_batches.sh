#!/bin/bash

# Check if the requests_filepath was passed as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <requests_filepath>"
    exit 1
fi

REQUESTS_FILEPATH=$1

# Check if the requests file exists
if [ ! -f "$REQUESTS_FILEPATH" ]; then
    echo "Error: File '$REQUESTS_FILEPATH' does not exist."
    exit 1
fi

# Record the start time (in seconds since the epoch)
start_time=$(date +%s)

# Define variables for the Python script arguments
REQUEST_URL="https://api.openai.com/v1/chat/completions"
MAX_REQUESTS_PER_MINUTE=3750
MAX_TOKENS_PER_MINUTE=2000000
TOKEN_ENCODING_NAME="cl100k_base"
MAX_ATTEMPTS=5
LOGGING_LEVEL=20

# Execute the Python script with the specified parameters
python api_request_parallel_processor.py \
  --requests_filepath "$REQUESTS_FILEPATH" \
  --request_url "$REQUEST_URL" \
  --max_requests_per_minute "$MAX_REQUESTS_PER_MINUTE" \
  --max_tokens_per_minute "$MAX_TOKENS_PER_MINUTE" \
  --token_encoding_name "$TOKEN_ENCODING_NAME" \
  --max_attempts "$MAX_ATTEMPTS" \
  --logging_level "$LOGGING_LEVEL"

# Check if the Python script exited successfully
if [ $? -eq 0 ]; then
    echo "API requests processed successfully."
else
    echo "Failed to process API requests."
fi

# Record the end time and compute the elapsed time
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
# echo "Time to completion: ${elapsed} seconds."

# Optionally, format the time more nicely:
elapsed_minutes=$((elapsed / 60))
elapsed_seconds=$((elapsed % 60))
echo "Elapsed time: ${elapsed_minutes}m ${elapsed_seconds}s"