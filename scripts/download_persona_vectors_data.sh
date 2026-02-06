#!/bin/bash
#
# download_persona_vectors_data.sh - Download trait data from Persona Vectors repository
#
# Downloads trait data files (evil, sycophantic, hallucinating) from the
# Persona Vectors repository (https://github.com/safety-research/persona_vectors).
#
# These files are not included in this repository due to licensing considerations.
#
# Usage:
#   ./scripts/download_persona_vectors_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL="https://raw.githubusercontent.com/safety-research/persona_vectors/main/data_generation"

TRAITS=("evil" "sycophantic" "hallucinating")
DIRS=("trait_data_eval" "trait_data_extract")

# Create directories if they don't exist
for dir in "${DIRS[@]}"; do
    mkdir -p "$REPO_ROOT/data_generation/$dir"
done

echo "Downloading trait data from Persona Vectors repository..."
echo "Repository: https://github.com/safety-research/persona_vectors"
echo ""

failed=()
success=0

for trait in "${TRAITS[@]}"; do
    for dir in "${DIRS[@]}"; do
        url="${BASE_URL}/${dir}/${trait}.json"
        output_path="$REPO_ROOT/data_generation/${dir}/${trait}.json"
        
        echo "Downloading: ${dir}/${trait}.json"
        
        if curl -sSfL "$url" -o "$output_path"; then
            echo "  ✓ Saved to: $output_path"
            success=$((success + 1))
        else
            echo "  ✗ Failed to download: $url"
            failed+=("${dir}/${trait}.json")
        fi
    done
done

echo ""
echo "=== Download Summary ==="
echo "Successfully downloaded: $success files"
echo "Failed: ${#failed[@]} files"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo ""
    echo "Failed files:"
    for f in "${failed[@]}"; do
        echo "  - $f"
    done
    exit 1
else
    echo ""
    echo "All files downloaded successfully!"
    exit 0
fi

