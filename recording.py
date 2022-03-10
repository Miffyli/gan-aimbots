#!/usr/bin/env python3
#
# main.py
# Play ViZDoom against bots with aimbots
#
import math as m
import numpy as np
import vizdoom as vzd
from vizdoom import Button
from aimbots.aimbots import *
from utils.mouse_emulation import move_mouse
from time import time
from uuid import getnode
import json
import argparse
import cv2

parser = argparse.ArgumentParser("Play Vizdoom with aimbots, and record footage")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--bots", type=int, default=6)
parser.add_argument("--timeout", type=int, default=1000)
parser.add_argument("--record_frames", action="store_true")
parser.add_argument("--aimbot", type=str, default="none",
                    choices=["none", "snap", "ease", "max_step", "ease_light", "ease_strong", "gan", "gan_light"])
parser.add_argument("--map", type=str, default="map03")
parser.add_argument("--model", type=str, default="", help="Path to stored model for aimbots that trained parameters.")
parser.add_argument("output", type=str, help="The name of output recordings")

WIDTH = 640
HEIGHT = 480

# Not in the center if HUD is enabled
CROSSHAIR_Y = 201
CROSSHAIR_X = 320

# If True, aimbot will always aim at stuff
AIMBOT_ALWAYS_ON = True

# Maximum distance to any possible target
# this is to avoid jittering
MAX_TARGET_DIST = 800

DEBUG_PRINTS = False
DEBUG_FRAMES = False

FOVX = 90
FOVY = FOVX * (WIDTH/HEIGHT)

# mouse_move delta x -> amount of degrees in game
# Calibrated for my mouse
# (doing mouse_move(1,0) will cause TURN_LEFT_RIGHT_DELTA to be this)
X_TO_GAME = 0.17578125
# Same for delta y
Y_TO_GAME = 0.087890625
GAME_TO_X = 1/X_TO_GAME
GAME_TO_Y = 1/Y_TO_GAME

# Enable buttons here rather than config file,
# just for tiny bit of more security
AVAILABLE_BUTTONS = [
    Button.MOVE_LEFT,
    Button.MOVE_RIGHT,
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.TURN_LEFT_RIGHT_DELTA,
    Button.LOOK_UP_DOWN_DELTA,
    Button.MOVE_FORWARD_BACKWARD_DELTA,
    Button.SPEED,
    Button.ATTACK,
    Button.SELECT_NEXT_WEAPON,
    Button.SELECT_PREV_WEAPON,
]

def get_visible_enemies(state, player_pos, max_distance=MAX_TARGET_DIST):
    """
    Returns list of centers of visible enemies, and another
    list of bounding boxes of the visible enemies (x0,y0,x1,y1).

    Also do not select targets too far away
    """
    enemy_objs = [obj for obj in state.labels if obj.object_name == "DoomPlayer"]
    img = state.labels_buffer
    enemy_middle_points = []
    enemy_bounding_boxes = []
    enemy_velocities = []
    for obj in enemy_objs:
        # Do not aim at yourself, dummy!
        if obj.value == 255: continue
        
        # Check distance to opponent
        distance_to_opponent = (
            (player_pos[0]-obj.object_position_x)**2 + 
            (player_pos[1]-obj.object_position_y)**2 +
            (player_pos[2]-obj.object_position_z)**2
        )
        if m.sqrt(distance_to_opponent) >= max_distance: continue

        # Use stuff from label info
        bbox = (obj.x, obj.y, obj.x + obj.width, obj.y + obj.height)
        enemy_bounding_boxes.append(bbox)
        enemy_velocities.append((obj.object_velocity_x, 
                                 obj.object_velocity_y, 
                                 obj.object_velocity_z))

    return enemy_bounding_boxes, enemy_velocities

def move_aim(dx,dy):
    # Move mouse by delta x/y amount of degrees
    dx = dx * GAME_TO_X
    dy = dy * GAME_TO_Y
    
    move_mouse(dx, dy)

def play_and_record_episodes(num_episodes, maps, aimbots, output,
                             timeout = 1000, num_bots = 6, store_frames = DEBUG_FRAMES,
                             record_video = False, additional_data = None):
    """
    num_episodes (int): Number of episodes to play
    maps (List): List of maps to play, one per episode
    aimbots (List): List of aimbots to use, one per episodfe
    output (str): Prefix of the output files
    timeout (int or List): Timeout of one episode (or list of timeouts, one per episode)
    record_video (bool): If true, record .avi video of the gameplay
    additional_data (dict): Additional data to put into saved json files
    """
    aimbot_args = (WIDTH, HEIGHT, FOVX, FOVY, CROSSHAIR_X, CROSSHAIR_Y)

    if isinstance(timeout, int):
        timeout = [timeout] * num_episodes
    else:
        assert len(timeout) == num_episodes

    game = vzd.DoomGame()

    game.set_doom_scenario_path("scenarios/cig.wad")
    game.set_labels_buffer_enabled(True)
    # Enables freelook in engine
    game.add_game_args("+freelook 1 +norawinput 1 -deathmatch")

    # Set settings here, just in case somebody starts
    # messing with the config file
    for button in AVAILABLE_BUTTONS:
        game.add_available_button(button)

    game.set_doom_map(maps[0])
    game.set_sound_enabled(True)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_render_hud(True)
    game.set_render_weapon(True)
    game.set_render_crosshair(True)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)

    game.init()

    # Set buttons
    game.send_game_command("bind w +forward")
    game.send_game_command("bind a +moveleft")
    game.send_game_command("bind s +back")
    game.send_game_command("bind d +moveright")

    # Deaths / damagecount does not update
    # between episodes, so keep track of it here
    last_ep_damages = 0.0
    last_ep_deaths = 0
    aimbot = None
    aimbot_name = None
    for episode_i in range(num_episodes):

        video_recorder = None
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            filename = output + ("_episode%d.mp4" % episode_i)
            video_recorder = cv2.VideoWriter(
                filename, fourcc, 35.0, (640, 480)
            )

        # Select new aimbot
        aimbot_name = aimbots[episode_i]
        if aimbot_name is None or aimbot_name == "none":
            aimbot = None
        elif aimbot_name == "ease_strong":
            aimbot = RelativeAimbot(*aimbot_args, max_aim_angle=15, hop_frac=0.6)
        elif aimbot_name == "ease_light":
            aimbot = RelativeAimbot(*aimbot_args, max_aim_angle=5, hop_frac=0.4)
        elif aimbot_name == "gan_group0":
            # Fixed model paths
            aimbot = GANAimbotNP(*aimbot_args, model_path="./shared_parameters/gan_group0_parameters.npz", max_aim_angle=15)
        elif aimbot_name == "gan_group1":
            # Fixed model paths
            aimbot = GANAimbotNP(*aimbot_args, model_path="./shared_parameters/gan_group1_parameters", max_aim_angle=15)


        if aimbot_name == "ease_strong":
            print("Game #{} , Aimbot ON (strong)".format(episode_i + 1))
        elif aimbot_name == "ease_light":
            print("Game #{} , Aimbot ON (light)".format(episode_i + 1))
        elif aimbot_name is not None and "gan" in aimbot_name:
            print("Game #{} , Aimbot ON".format(episode_i + 1))
        else:
            print("Game #{} , Aimbot {}".format(episode_i + 1, "OFF" if aimbot is None else "ON"))
        
        game.set_episode_timeout(timeout[episode_i])

        # Select new map
        game.set_doom_map(maps[episode_i])

        game.new_episode()

        game.send_game_command("removebots")
        for i in range(num_bots):
            game.send_game_command("addbot")

        # Stored actions, enemy locations
        # and enemy bbs
        stored_actions = []
        stored_damages = []
        stored_ammos = []
        stored_weaps = []
        stored_frags = []
        stored_deaths = []
        stored_vels = []
        stored_enemies = []
        stored_enemy_bbs = []
        stored_enemy_vels = []
        # For debuggin
        stored_frames = []

        last_action = []
        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()
            
            state = game.get_state()

            if state is None:
                game.advance_action()
                continue
            
            last_action = game.get_last_action()
            
            #start_time = time()
            # Measured max. 5ms latency here (not too bad)
            enemy_bbs = []
            if aimbot is not None:
                aimbot.process_last_action(last_action)
            if AIMBOT_ALWAYS_ON:
                player_pos = (
                    game.get_game_variable(vzd.GameVariable.POSITION_X),
                    game.get_game_variable(vzd.GameVariable.POSITION_Y),
                    game.get_game_variable(vzd.GameVariable.POSITION_Z),
                )
                enemy_bbs, enemy_vels = get_visible_enemies(state, player_pos)
                
                if len(enemy_bbs) > 0 and aimbot is not None:
                    dx,dy = aimbot.get_aiming(enemy_bbs)
                    if dx != 0 or dy != 0:
                        move_aim(dx, dy)
            
            #print("ms for aimbot: %f" % ((time() - start_time)*1000))
            
            reward = game.get_last_reward()
            damage_done = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT) - last_ep_damages
            ammo = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
            weap = game.get_game_variable(vzd.GameVariable.SELECTED_WEAPON)
            frags = game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
            deaths = game.get_game_variable(vzd.GameVariable.DEATHCOUNT) - last_ep_deaths

            velocities = (
                game.get_game_variable(vzd.GameVariable.VELOCITY_X),
                game.get_game_variable(vzd.GameVariable.VELOCITY_Y),
                game.get_game_variable(vzd.GameVariable.VELOCITY_Z),
            )

            if DEBUG_PRINTS:
                print("State #" + str(state.number))
                print("Game variables: ", state.game_variables)
                print("Action:", last_action)
                print("Reward:", reward)
                print("Dmgcount: ", damage_done)
                print("Frags: ", frags)
                print("Deaths: ", deaths)
                print("Weap: ", weap)
                print("Ammo: ", ammo)
                print("=====================")

            if store_frames:
                stored_frames.append(state.screen_buffer)
            if video_recorder is not None:
                image = state.screen_buffer
                # Move channel to last and flip RGB to BGR
                image = image.transpose([1, 2, 0])
                video_recorder.write(image[..., ::-1])
            stored_actions.append(last_action)
            stored_damages.append(damage_done)
            stored_ammos.append(ammo)
            stored_weaps.append(weap)
            stored_frags.append(frags)
            stored_deaths.append(deaths)
            stored_vels.append(velocities)
            stored_enemy_bbs.append(enemy_bbs)
            stored_enemy_vels.append(enemy_vels)

            game.advance_action()

        last_ep_damages = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        last_ep_deaths = game.get_game_variable(vzd.GameVariable.DEATHCOUNT)

        filename = output + ("_episode%d.json" % episode_i)
        save_dict = {
            "timestamp": int(time()),
            "hwid": int(getnode()),
            "aimbot": aimbot_name,
            "frames": stored_frames, # For debugging
            "actions": stored_actions,
            "damages": stored_damages,
            "ammos": stored_ammos,
            "weapons": stored_weaps,
            "frags": stored_frags,
            "deaths":  stored_deaths,
            "velocities": stored_vels,
            "enemy_bbs": stored_enemy_bbs,
            "enemy_vels": stored_enemy_vels,
        }
        if additional_data is not None:
            assert isinstance(additional_data, dict)
            save_dict.update(additional_data)

        with open(filename, "w") as f:
            json.dump(save_dict, f)

        if video_recorder is not None:
            video_recorder.release()

    game.close()

if __name__ == '__main__':
    args = parser.parse_args()
    play_and_record_episodes(args.episodes, 
                             [args.map] * args.episodes,
                             [args.aimbot] * args.episodes,
                             args.output,
                             num_bots = args.bots, 
                             timeout = args.timeout,
                             store_frames = args.record_frames)
