#!/bin/bash
set -e

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

EXPLOITS=(
  "e0a.sh"
  "e1a.sh"
  "e2a.sh"
  "e3a.sh"
  "e4a.sh"
  "e5a.sh"
  "e6a.sh"
  "e7a.sh"
  "e8a.sh"
  "e9a.sh"
  "e10a.sh"
  "e11a.sh"
  "e12a.sh"
  "e13a.sh"
  "e14a.sh"
  "e15a.sh"
  "e16a.sh"
  "e17a.sh"
  "e18a.sh"
)



echo "Running all vulnerability exploits..."
echo ""

for exploit in "${EXPLOITS[@]}"; do
  echo "Running: $exploit"
  
  echo "Resetting environment..."
  ./setup.bash
  
  cd exploits
  ./"$exploit"
  cd ..
  
  echo ""
  echo "Press enter to continue to next exploit, or ctrl c to exit"
  read -r
  echo ""
done

