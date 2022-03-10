#!/usr/bin/env python3
#
# data_collection_gan.py
# Main entrypoint for datacollection
#
import os
import random

from recording import play_and_record_episodes
from time import sleep

AGREEMENT = """
 Machine learning versus machine learning: Case of aimbots.
 Organizer: Anssi Kanervisto (anssk@uef.fi)
            School of Computing, University of Eastern Finland
 ------------------------------------------------------------------------------
 BACKGROUND

 Terms:
     "Cheating": When a video-game player uses software ("cheat") to gain
                 an unfair advantage over other players.
     "Aimbot":   Particular cheat which assists player with aiming at
                 enemy players.

 Purpose of this study is two-fold:
     1) Detecting aimbot software from player's mouse
        movement using machine learning techniques
     2) Applying machine learning techniques to create
        aimbot that is not distinguishable from human players.

     Goal is to obtain public knowledge on such cheats, which can then
     be used to create better anti-cheating software.

 What data will be collected and how:
     In these experiments, you will play a game of Doom (1993)
     against computer-players. In some of the games, your aim
     will be assisted by software ("aimbot").
     The software will record your keypresses and mouse movement
     in the game.

     A hardware identifier of your machine will be included in the data.

     All data will be anonymous and no private information will be
     collected.

 How this data will be used:
     The recorded data will be used to...

     1) ... evaluate the performance of machine learning based aimbots.

     This data may be released to the public.

     This data may be used in future research.

 Requirements:
     - A separate mouse (not a trackpad/mousepad)
     - 30 minutes of uninterrupted time (you can not pause the experiment)

 ------------------------------------------------------------------------------
 AGREEMENT

 By agreeing you:
     - ... confirm you recognize what data will be collected and how
       it will be used (scroll up).
     - ... are 18 years old or above.
     - ... fulfil the requirements (see above).
     - ... acknowledge there is no compensation.
     - ... acknowledge your participation is voluntary and you may withdraw at
       any point before end of the experiment.

 Input 'I agree' to continue, "quit" to close this software."""

INSTRUCTIONS = """
 INSTRUCTIONS

     This experiment will take 30 minutes, and consists
     of six games of Doom deathmatch against bots.

     Your goal is to score frags by shooting the enemies.
     You will find different weapons on the map, as well
     ammo for them and health/armor kits.

     For each game you will aimbot on or off, automatically
     set by this software. If aimbot is enabled, your aim will
     "home in" in to enemy targets once they are close enough
     to your crosshair.

     Once all games are finished, this script will upload
     recorded files to a server.

     NOTE: Avoid opening the menu (ESC) and unfocusing the window
           (clicking outside the game window or changing window)

 BUTTONS

     WASD:             Forward, left, backward and right
     SHIFT:            Run (hold down to move faster)
     Mouse:            Aim
     Left-mouse click: Shoot
     Scroll up/down:   Change weapon

     Note that you can also aim up/down.

 Press ENTER to proceed"""

AFTER_GAMES = """
 Games finished.
 Press ENTER to upload recordings to server.
 Note that these files will be locally removed after upload.
"""

FINISHED = """
 Everything done.
 Thank you for participating! Closing in 5 seconds.
"""

PLAYER_PERFORMANCE_QUERY = """
 Please provide your estimated familiarity with first-person shooter (FPS) games.
    1: I have very little experience with FPS games (less than a hour of experience)
    2: I am familiar with FPS games but do not actively play them (1 - 10 hours of experience)
    3: I am an experienced FPS player (10 - 100 hours of experience)
    4: I am a very experienced FPS player (above 100 hours of experience)
"""

NUMBER_OF_EPISODES = 6
# 10 minutes
EPISODE_LENGTH_10MIN = 35 * 60 * 10
# 2.5 minutes
EPISODE_LENGTH_2ANDHALFMIN = int(35 * 60 * 2.5)

MAPS = ["map03"] * NUMBER_OF_EPISODES

def main_data_collection():
    os.system("cls")
    print(AGREEMENT)
    agreement = ""
    while agreement != "i agree" and agreement != "quit":
        agreement = input(" >>> ").lower()
    if agreement == "quit":
        exit(0)

    # Continue to instrunctions etc
    os.system("cls")
    print(INSTRUCTIONS)
    input()

    # Ask for the player's performance-level
    os.system("cls")
    print(PLAYER_PERFORMANCE_QUERY)
    player_fps_familiarity = None
    while player_fps_familiarity is None:
        input_str = input(" Give your answer (single digit in 1 - 4): ")
        try:
            input_int = int(input_str)
            if input_int in [1, 2, 3, 4]:
                player_fps_familiarity = input_int
        except Exception:
            pass

    os.system("cls")
    print(
        " You will now proceed to play the games. You will start with two practice games:\n"
        "    1. Without aimbot (10min)\n"
        "    2. With aimbot (10min)\n"
        " The remaining four games (2.5min) will have random aimbots enabled.\n"
        " Your performance will only be recorded in the last four games.\n"
        " Press ENTER to start the first game."
    )
    input()

    # Create the set of aimbots
    recording_aimbots = [None, "ease_light", "ease_strong", "gan_group0"]
    random.shuffle(recording_aimbots)
    aimbots = [None, "ease_light"] + recording_aimbots

    episode_lengths = [
        EPISODE_LENGTH_10MIN,
        EPISODE_LENGTH_10MIN,
        EPISODE_LENGTH_2ANDHALFMIN,
        EPISODE_LENGTH_2ANDHALFMIN,
        EPISODE_LENGTH_2ANDHALFMIN,
        EPISODE_LENGTH_2ANDHALFMIN,
    ]

    # Play games to create recordings
    play_and_record_episodes(
        NUMBER_OF_EPISODES,
        MAPS,
        aimbots,
        "recording",
        timeout=episode_lengths,
        additional_data={"player-fps-familiarity": player_fps_familiarity}
    )

    print(AFTER_GAMES)
    input()

    # Upload recordings to the server
    print("[NOTE] No data uploading in shared code")

    print(FINISHED)
    sleep(5)


if __name__ == '__main__':
    import sys

    # Workaround for pyinstaller:
    # Some of the local files are in
    # the temp folder used by pyinstaller,
    # so we need to navigate there.
    # Stackoverflow #57480958
    if hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)

    main_data_collection()
