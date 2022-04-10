#!/bin/bash
git config --global user.name "seiichiinoue"
git config --global user.email "d35inou108@gmail.com"

git remote set-url origin https://seiichiinoue:${GITHUB_TOKEN}@github.com/tmu-nlp/NLPtutorial2022.git
git checkout -b master

git log -1
last_commit_message="$(git log -1 | tail -1)"
echo $last_commit_message

if [ -z "$(echo $last_commit_message | grep updater)" ]; then
    python3 make_progress.py
    git add progress.png
    git commit -m '[updater] update progress bar'
    git push origin HEAD
else
    echo "nothing updated"
fi