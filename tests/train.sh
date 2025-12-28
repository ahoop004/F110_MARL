#!/bin/bash
# Convenience script for running v2 training scenarios

PYTHONPATH=/home/aaron/F110_MARL:$PYTHONPATH python3 v2/run.py "$@"
