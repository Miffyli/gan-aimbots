#!/bin/bash
# Train one aimbot gan per data directory.
# Hardcoded number of groups.
mkdir -p gan_models
for group_id in 0 1; do
    python3 aimbots/humanlike_aimbot_gan.py train gan_train_data/group${group_id}/* --model gan_models/gan_group${group_id} --epochs 100
    # Also convert the generator into numpython3 arrays so we can use it for aimbotting.
    python3 aimbots/humanlike_aimbot_gan.py params_to_numpython3 dummy --model gan_models/gan_group${group_id}_G
done
