#!/usr/bin/env bash
# Copyright (c) 2026 Lu-Xing Yang.
# No project-wide open-source license is granted; see repository COPYRIGHT_AND_LICENSE.md.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/reference_repositories_upstream}"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

clone_or_update() {
  local url="$1"
  local name="$2"
  if [ -d "$name/.git" ]; then
    echo "Updating $name"
    git -C "$name" pull --ff-only
  else
    echo "Cloning $name"
    git clone --depth 1 "$url" "$name"
  fi
}

clone_or_update "https://github.com/XiaojuanCheng/OpinionMalware_TIFS_2025_Code.git" "OpinionMalware_TIFS_2025_Code"
clone_or_update "https://github.com/XiaojuanCheng/PropagandaWar_TIFS_2024_Code.git" "PropagandaWar_TIFS_2024_Code"
clone_or_update "https://github.com/XiaojuanCheng/Propaganda_TCSS_2025_Code.git" "Propaganda_TCSS_2025_Code"

echo "Reference repositories are available in $TARGET_DIR"
echo "Use with: python3 run_reference_smoke.py --package-dir <directory-containing-reference_repositories>"
