#!/bin/bash
# Usage: ./extract_user_content.sh <filename> <row_id>
# Example: ./extract_user_content.sh api_requests_part4.jsonl 34629

# Check if exactly 2 arguments are provided
#if [ "$#" -ne 2 ]; then
    #echo "Usage: $0 <filename> <row_id>"
    #exit 1
#fi

#FILENAME="$1"
#ROW_ID="$2"
#TARGET="request-$ROW_ID"

## Use jq to extract and print the user message content for the given row_id
#jq -r --arg rid "$TARGET" 'select(.metadata.row_id == $rid) | .messages[] | select(.role == "user") | .content' "$FILENAME"

#!/bin/bash
# Usage: ./extract_messages.sh --filename <filename> --requestid <requestid>
# Example: ./extract_messages.sh --filename api_requests_part1_results.jsonl --requestid 1739

# usage() {
#   echo "Usage: $0 --filename <filename> --requestid <requestid>"
#   exit 1
# }

# # Parse command-line arguments
# while [[ "$#" -gt 0 ]]; do
#   case $1 in
#     --filename)
#       FILENAME="$2"
#       shift 2
#       ;;
#     --requestid)
#       REQUESTID="$2"
#       shift 2
#       ;;
#     *)
#       echo "Unknown parameter: $1"
#       usage
#       ;;
#   esac
# done

# # Ensure both filename and request id are provided
# if [[ -z "$FILENAME" || -z "$REQUESTID" ]]; then
#   usage
# fi

# TARGET="request-$REQUESTID"

# # Use jq to extract and print messages with newlines
# jq -r --arg rid "$TARGET" '
#   select(.[2].row_id == $rid) |
#   "User Message:\n" +
#   (.[0].messages[] | select(.role=="user") | .content) +
#   "\n\nAssistant Message:\n" +
#   (.[1].choices[0].message.content)
# ' "$FILENAME"

#!/bin/bash
# Usage: ./extract_messages.sh --filename <filename> [--requestid <requestid>]
# If --requestid is not provided, a random one from the file will be used.

usage() {
  echo "Usage: $0 --filename <filename> [--requestid <requestid>]"
  exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --filename)
      FILENAME="$2"
      shift 2
      ;;
    --requestid)
      REQUESTID="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      usage
      ;;
  esac
done

if [[ -z "$FILENAME" ]]; then
  usage
fi

# If no request id provided, pick a random one from the file
if [[ -z "$REQUESTID" ]]; then
  RANDOM_RECORD=$(shuf -n 1 "$FILENAME")
  REQUESTID=$(echo "$RANDOM_RECORD" | jq -r '.[2].row_id')
else
  # If provided externally, prepend "request-" if not already present
  if [[ "$REQUESTID" != request-* ]]; then
    REQUESTID="request-$REQUESTID"
  fi
fi

# TARGET="request-$REQUESTID"

echo "Using Request ID: $REQUESTID"
echo "----------------------------------"

# Use jq to extract and print messages with actual newlines
jq -r --arg rid "$REQUESTID" '
  select(.[2].row_id == $rid) |
  "User Message:\n" +
  (.[0].messages[] | select(.role=="user") | .content) +
  "\n\nAssistant Message:\n" +
  (.[1].choices[0].message.content)
' "$FILENAME"