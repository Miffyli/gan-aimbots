#!/usr/bin/env python3
#
# feature_extraction.py 
# Turn player trajectories into features for 
# ML methods to classify to non-cheater / cheater
#
import math as m
import numpy as np
import json
import argparse
import os

parser = argparse.ArgumentParser("Player trajectories into ML features for classification")
parser.add_argument("method", choices=["vacnet"])
parser.add_argument("--vacnet_num_shots_per_feature", type=int, default=1)
parser.add_argument("--vacnet_hor_only", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("gameplay_files", nargs="+")
parser.add_argument("output")

WIDTH = 640
HEIGHT = 480

# Match with CROSSHAIR_X/Y defined
# during recording
CROSSHAIR_Y = 201
CROSSHAIR_X = 320

# Maps weapon number to ammo per shot
# (e.g. super-shotgun uses two per shot)
WEAPON_TO_AMMO_PER_SHOT = {
    1: 1, # Fists
    2: 1, # Pistol
    3: 2, # Shotgun
    4: 1, # Minigun
    5: 1, # Rocket launcher
    6: 1, # Plasmagun
}

# Skip these weapons (do not include their
# shots in the features)
SKIP_WEAPONS = [
    5, # Rocket launcher
    6, # Plasmagun
]

# VACNet extraction default params
VACNET_SECONDS_BEFORE_SHOT = 1/2
VACNET_SECONDS_AFTER_SHOT  = 1/4

# Aimangle delta indexes in actions
AIMANGLE_DELTA_YAW_IDX = 4
AIMANGLE_DELTA_PITCH_IDX = 5

def extract_vacnet(episode_data, shots_per_feature, hor_only=False):
    """
    Do vacnet type of extraction on one episode of data.
    Returns numpy array of N x D of input features.

    Note: Features are in form 
        (dx1, dx2, dx3, ... , dy1, dy2, dy3, ... , is_hit)
    
    Source: (https://www.youtube.com/watch?v=ObhK8lUfIlc).
    Note: We do not include selected weapon
    """
    
    episode_length = len(episode_data["ammos"])
    frames_before_shot = int(VACNET_SECONDS_BEFORE_SHOT * 35)
    frames_after_shot  = int(VACNET_SECONDS_AFTER_SHOT  * 35)

    # Find indexes where shots happened
    shot_idxs = []
    ammos = episode_data["ammos"]
    weaps = episode_data["weapons"]
    last_ammo = None
    for i in range(1,episode_length):
        current_weapon = int(weaps[i])
        # Do not continue checking if this is a weapon
        # we should skip
        if current_weapon in SKIP_WEAPONS: continue
        # A shot: When number of 
        # ammo changed according per ammo_per_shot
        if (ammos[i - 1] - ammos[i]) == WEAPON_TO_AMMO_PER_SHOT[current_weapon]:
            # A shot
            shot_idxs.append(i)

    # For each shot, gather mouse movement before and 
    # after the shot, and include if it was a hit
    features = []
    yaw_deltas =   [action[AIMANGLE_DELTA_YAW_IDX] for action in  episode_data["actions"]]
    pitch_deltas = [action[AIMANGLE_DELTA_PITCH_IDX] for action in  episode_data["actions"]]
    damages = episode_data["damages"]
    for shot_idx in shot_idxs:
        a_hit = int(damages[shot_idx] != damages[shot_idx - 1])
        # Gather angle changes
        yaws = yaw_deltas[shot_idx - frames_before_shot : shot_idx + frames_after_shot]
        pitches = pitch_deltas[shot_idx - frames_before_shot : shot_idx + frames_after_shot]
        # Make sure we always have same-length frames
        if len(yaws) == (frames_before_shot + frames_after_shot):
            if hor_only:
                features.append(yaws + [a_hit])
            else:
                features.append(yaws + pitches + [a_hit])

    # Now take features of successive shots and combine them into
    # one feature (takes up space but easiest this way)
    successive_features = []
    for i in range(shots_per_feature, len(features)):
        successive_features.append(features[i - shots_per_feature:i])
    successive_features = np.array(successive_features)
    # Ravel extra dimension we have
    successive_features = successive_features.reshape((len(successive_features), -1))

    return successive_features

def main(args):
    gameplay_files = args.gameplay_files

    all_features   = []
    # 1 = cheating, 0 = legit
    all_labels     = []
    aimbot_classes = []

    for gameplay_file in gameplay_files:
        features = None
        data = json.load(open(gameplay_file, "r"))
        if args.method == "vacnet":
            features = extract_vacnet(data, args.vacnet_num_shots_per_feature, args.vacnet_hor_only)
        cheating = int(data["aimbot"] is not None and data["aimbot"] != "none")
        
        aimbot_class = 0
        if cheating:
            if data["aimbot"] == "ease_light":
                aimbot_class = 1
            elif data["aimbot"] == "ease_strong":
                aimbot_class = 2
            # GAN splitting
            elif data["aimbot"] == "gan_group0":
                aimbot_class = 10
            elif data["aimbot"] == "gan_group1":
                aimbot_class = 11

        all_features.append(features)
        all_labels.extend([cheating]*len(features))
        aimbot_classes.extend([aimbot_class]*len(features))

        if args.verbose:
            print("%s: %d features of label %d" % (gameplay_file, len(features), cheating))

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels).astype(np.int)
    aimbot_classes = np.array(aimbot_classes).astype(np.int)

    if args.verbose:
        print("Total legit:    %d (%.2f)" % (np.sum(all_labels == 0), 1 - np.mean(all_labels)))
        print("Total cheating: %d (%.2f)" % (np.sum(all_labels), np.mean(all_labels)))
        print("Total samples:  %d" % len(all_labels))

    np.savez(args.output, features=all_features, labels=all_labels, aimbot_class=aimbot_classes)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
