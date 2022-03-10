#!/usr/bin/env python3
#
# data_collection_gan.py
# Main entrypoint for datacollection
#
import os
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

     1) ... evaluate the quality of the machine-learning based aimbots
     2) ... used to train machine learning models to detect these aimbots

     This data may be released to the public.

     This data may be used in future research.

 Requirements:
     - A separate mouse (not a trackpad/mousepad)
     - 20 minutes of uninterrupted time (you can not pause the experiment)

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

     This experiment will take 20 minutes, and consists
     of four 5-minute games of Doom (1993).

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

 Press ENTER to start the game"""

AFTER_GAMES = """
 Games finished.
 Press ENTER to upload recordings to server.
 Note that these files will be locally removed after upload.
"""

FINISHED = """
 Everything done.
 Thank you for participating! Closing in 5 seconds."""

NUMBER_OF_EPISODES = 4
# 5 minutes
EPISODE_LENGTH = 35 * 60 * 5

MAPS = ["map03"] * NUMBER_OF_EPISODES

AIMBOTS = [None, None, "gan_group0", "gan_group1"]


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

    # Play games to create recordings
    play_and_record_episodes(
        NUMBER_OF_EPISODES,
        MAPS,
        AIMBOTS,
        "recording",
        timeout=EPISODE_LENGTH,
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
