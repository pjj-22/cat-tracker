#!/bin/bash
PI=${PI_HOST:-"patrickjamesdev@10.0.0.209"}
REMOTE=~/Projects/cat-tracker/captures/
LOCAL=./captures/

rsync -avz --progress --include="*/" --include="labels.json" --exclude="*" "$LOCAL" "$PI:$REMOTE"
