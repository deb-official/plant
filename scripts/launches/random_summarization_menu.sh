#!/bin/bash

# Check if a file argument was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <jsonl_file>"
  exit 1
fi

FILE="$1"

# Check if the file exists
if [ ! -f "$FILE" ]; then
  echo "File '$FILE' not found."
  exit 1
fi

while true; do
    # Select a random record from the file
    random_record=$(shuf -n 1 "$FILE")
    
    # Extract the _id from the random record (first object with an _id key)
    random_id=$(echo "$random_record" | jq -r 'map(select(has("_id")))[0]._id')
    
    # If no _id was found, report and try again
    if [ -z "$random_id" ]; then
      echo "No _id found in the selected record. Trying again..."
      continue
    fi
    
    echo "------------------------------------"
    echo "Random _id selected: $random_id"
    echo "------------------------------------"
    
    # Inner loop: allow the user to choose an option for the current random id
    while true; do
         echo ""
         echo "Options for _id: $random_id"
         echo "1: Display the entire API request prompt and CHATGPT response"
         echo "2: Display only the API request prompt"
         echo "3: Display only the CHATGPT response"
         echo "q: Quit"
         echo "Press Enter without input to select a new random _id."
         read -p "Enter your choice: " choice

         # If the user just presses enter, break out to get a new random _id
         if [ -z "$choice" ]; then
              break
         fi

         case "$choice" in
             1)
                 echo "Executing Option 1 for _id: $random_id"
                 jq 'select(any(.[]; ._id? == "'"$random_id"'"))' "$FILE"
                 ;;
             2)
                 echo "Executing Option 2 for _id: $random_id"
                 jq -r 'select(any(.[]; ._id? == "'"$random_id"'")) | .[0].messages[1].content' "$FILE" | less
                 ;;
             3)
                 echo "Executing Option 3 for _id: $random_id"
                 jq 'select(any(.[]; ._id? == "'"$random_id"'")) | .[1].choices[0].message.content' "$FILE"
                 ;;
             q|Q)
                 echo "Exiting..."
                 exit 0
                 ;;
             *)
                 echo "Invalid option. Please try again."
                 ;;
         esac
         echo ""
         # After execution, the loop continues allowing another option for the current id
    done
done

