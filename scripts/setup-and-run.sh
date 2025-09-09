#!/usr/bin/env bash
set -euo pipefail

# Optional: pin Node version with nvm/volta here if desired
# export NVM_DIR="$HOME/.nvm" && . "$NVM_DIR/nvm.sh" && nvm use 18

if command -v npm >/dev/null 2>&1; then
  if [ -f package-lock.json ]; then
    npm ci
  else
    npm install
  fi
  npm run dev
else
  echo "Error: npm is not installed. Install Node.js 18+ (which includes npm)." >&2
  exit 1
fi
