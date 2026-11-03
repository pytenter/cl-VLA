#!/usr/bin/env bash
# Portable entry point. Historical launchers under legacy_runs need path review.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/env.sh" "${1:-openvla-robotwin}"
