#!/usr/bin/env bash
# entrypoint.sh

source /opt/conda/etc/profile.d/conda.sh
conda activate sync
cd /home/sarpba/AI_Dubbing
exec python main_app.py
