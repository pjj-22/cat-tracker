#!/bin/bash
PI=${PI_HOST:-"patrickjamesdev@10.0.0.209"}
REMOTE=~/Projects/cat-tracker/recordings/
LOCAL=./recordings/

mkdir -p "$LOCAL"
rsync -avz --progress "$PI:$REMOTE" "$LOCAL"
