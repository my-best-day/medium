#!/bin/bash

set -e
set -x

TARGET=$1

# if target is missing, print usage
if [ -z "${TARGET}" ]; then
    echo "Usage: $0 <target>"
    exit 1
fi

# check if target is reachable
if ! ping -c 1 -W 1 "${TARGET}" > /dev/null; then
    echo "Target host ${TARGET} is not reachable."
    exit 1
fi

# list of files to copy
files=(
    ~/.ssh/git_rsa
    ~/dev/torch/medium/ignore/datasets32.tar
    ~/dev/torch/medium/bert-it-1/bert-it-vocab.txt
    ~/dev/torch/medium/src/config.py
)

# copy each file
for file in "${files[@]}"; do
    if [ -f "${file}" ]; then
        scp "${file}" "${TARGET}:"
    else
        echo "File ${file} does not exist."
    fi
done

cat << EOF
touch .no_auto_tmux
mv git_rsa .ssh/
eval "\$(ssh-agent -s)"
ssh-add ~/.ssh/git_rsa
git clone git@github.com:my-best-day/medium.git
mkdir medium/bert-it-1
mv bert-it-vocab.txt medium/bert-it-1/
cd medium/
tar xfv ~/datasets32.tar
mv ~/config.py src/
sudo apt update
apt install vim
vi src/config.py
pip install --upgrade pip
pip install -r requirements.txt
mkdir logs checkpoints
EOF