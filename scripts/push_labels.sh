#!/bin/bash
PI=${PI_HOST:-"patrickjamesdev@10.0.0.209"}
REMOTE=~/Projects/cat-tracker/captures/
LOCAL=./captures/

rsync -avz --progress --include="*/" --include="labels.json" --exclude="*" "$LOCAL" "$PI:$REMOTE"

ssh "$PI" "cd ~/Projects/cat-tracker && rm -f cat_profiles.json && python3 build_profiles.py captures/session_*"
