#!/bin/bash

# 1. Verify subdir checkpoints and logs exists. Complain otherwise. Checkpoints might be a softlink to a dir.
# 2. Check if there are any files in the length of 88c in the logs dir.
# 3. If any such files are found, report them and exit, unless:
# 4. If a flag is provided, say -f, delete these files.
# 5. At this point check if any empty direct sub directories exist.
# 6. If found report them and exit, unless:
# 7. A flag is provided, say -f, and in that case delete the empty directories.
# 8. Only consume the flag once. That is, if you use it to delete 88c files, in step 7 the flag is not provided.

# Parse options
FORCE=0
while getopts "f" opt; do
  case ${opt} in
    f)
      FORCE=1
      ;;
    \?)
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Verify directories exist
if [ ! -d checkpoints ] || [ ! -d logs ]; then
  echo "Error: 'checkpoints' and/or 'logs' directory does not exist."
  exit 1
fi

# Check for 88-byte files in logs
FILES=$(find logs -type f -size 88c)
if [ -n "$FILES" ]; then
  echo "Found 88-byte files in 'logs':"
  echo "$FILES"
  echo "Use -f to delete."
  if [ $FORCE -eq 1 ]; then
    echo "Deleting files..."
    find logs -type f -size 88c -delete
    FORCE=0
  else
    exit 1
  fi
fi

# Check for empty directories in 'logs' and 'checkpoints'
DIRS=$(find logs checkpoints -mindepth 1 -type d -empty)
if [ -n "$DIRS" ]; then
  echo "Found empty directories:"
  echo "$DIRS"
  echo "Use -f to delete."
  if [ $FORCE -eq 1 ]; then
    echo "Deleting directories..."
    find logs checkpoints -mindepth 1 -type d -empty -delete
  else
    exit 1
  fi
fi
