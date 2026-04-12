#!/bin/zsh
set -e

COMMIT_MSG=${1:-"Update blog"}
REMOTE="git@github.com:eschizoid/eschizoid.github.io.git"

echo "🔨 Building site..."
hugo

echo "📦 Committing source to main..."
git add content/ config.toml static/ deploy.sh
git commit -m "$COMMIT_MSG" || echo "Nothing new to commit on main"
git push origin main

echo "🚀 Deploying public/ to gh-pages..."
cd public
git add -A
git commit -m "$COMMIT_MSG" || echo "Nothing new to commit on gh-pages"
git push -f origin HEAD:gh-pages && git fetch origin && git branch --set-upstream-to=origin/gh-pages gh-pages
cd ..

echo "✅ Done! Site published to gh-pages."

