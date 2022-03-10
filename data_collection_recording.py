#!/usr/bin/env python3
#
# data_collection_recording.py
# Main entrypoint for datacollection (for recording videos with different videos)
#
import os
from recording import play_and_record_episodes
from time import sleep

import random

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

     1) ... evaluate the quality of the aimbots against human judges.
            The (anonymous) recorded gameplay will be shown to experienced
            game server admins and they will judge if gameplay should be
            considered suspicious (i.e. player is using a cheat) or not.

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

     Try to appear as innocent as possible. That is: even with
     aimbot enabled, try to appear as if no aimbot were enabled.
     Pretend you are playing against other human players and
     you are cheating with the aimbot, and you have to avoid
     getting caught for using a cheat.

     Once all games are finished, you will have to provide
     the host of the experiment with the files this records.

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
 [NOTE] There is no data uploading in shared code.
 Thank you for participating!
"""


NUMBER_OF_EPISODES = 4
# 5 minutes
EPISODE_LENGTH = 35 * 60 * 5

MAPS = ["map03"] * NUMBER_OF_EPISODES

AIMBOTS = [None, "ease_light", "ease_strong", "gan_group0"]
# Shuffle the aimbots so the ordering is not known
random.shuffle(AIMBOTS)


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
        record_video=True
    )

    print(AFTER_GAMES)


if __name__ == '__main__':
    import sys
    import glob
    import shutil

    original_dir = os.path.abspath(os.curdir)

    # Workaround for pyinstaller:
    # Some of the local files are in
    # the temp folder used by pyinstaller,
    # so we need to navigate there.
    # Stackoverflow #57480958
    if hasattr(sys, "_MEIPASS"):
        os.chdir(sys._MEIPASS)

    main_data_collection()

    # Move all data back to original location
    recordings = glob.glob("*_episode*")
    for recording in recordings:
        shutil.move(recording, os.path.join(original_dir, os.path.basename(recording)))
