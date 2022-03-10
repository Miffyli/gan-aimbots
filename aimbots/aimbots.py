#!/usr/bin/env python3
#
# aimbots.py 
# Implements bunch of different aimbots
#
import math as m
from random import uniform, gauss
from collections import deque
import numpy as np

TARGET_Y_RANGE = [0.32,  0.32]
TARGET_X_RANGE = [0.5,  0.5]

# If dx/dy angle is larger than this, 
# do not aim (i.e. does not aim at enemies
# too far away)
DEFAULT_MAX_ANGLE = 7

def pick_random_spot(bb):
    # Return random-ish spot from inside enemy bounding box
    # according to TARGET_X/Y_RANGE
    x_frac = uniform(*TARGET_X_RANGE)
    y_frac = uniform(*TARGET_Y_RANGE)
    return (bb[0] + (bb[2] - bb[0]) * x_frac, bb[1] + (bb[3] - bb[1]) * y_frac)

def select_closest_enemy(enemy_bbs, crosshair_x, crosshair_y):
    # Returns x,y of the closest enemy to crosshair
    # (x,y is the center of enemy).
    # enemy_bbs are bounding boxes of enemies ()
    min_dist = 1e99
    closest_bb = None
    # Select enemy whose bb is closest to ours
    for bb in enemy_bbs:
        # Distance to the center of the player
        dist = ((bb[0] + bb[2])//2 - crosshair_x)**2 + ((bb[1] + bb[3])//2 - crosshair_y)**2
        if dist < min_dist:
            min_dist = dist
            # Hurrr
            closest_bb = bb
    # No valid target, return bogus target
    if closest_bb is None:
        return [-100, -100]

    target_point = pick_random_spot(closest_bb)
    return target_point


def sign(x):
    """ 
    Return sign of x:
        -1 if x < 0 else 1
    """
    return -1 if x < 0 else 1


def ELU(x):
    """
    Exponential Linear Unit,
    as implemented in PyTorch:
        https://pytorch.org/docs/master/generated/torch.nn.ELU.html
    """
    return np.clip(x, 0, None) + np.clip(np.exp(x) - 1, None, 0)


def fnn_with_elus(x, weights, biases):
    """
    Run input through a fully-connected network
    with ELU activations, with layers
    with given weights and biases
    """
    for i in range(len(weights) - 1):
        x = (x @ weights[i].T) + biases[i]
        x = ELU(x)
    # Final mapping is linear
    x = (x @ weights[-1].T) + biases[-1]
    return x


class Aimbot:
    """ Abstract class for guidance """
    def __init__(self, width, height, fovx, fovy, crosshair_x, crosshair_y,
                 max_aim_angle = DEFAULT_MAX_ANGLE):
        self.width = width
        self.height = height
        self.fovx = fovx
        self.fovy = fovy
        self.crosshair_x = crosshair_x
        self.crosshair_y = crosshair_y

        self.max_aim_angle = max_aim_angle

    def pixels_to_deg(self, x, y):
        """ Returns degrees of deviation from center to target pixel """
        x_degrees = ((x - self.crosshair_x)/self.width)*self.fovx
        y_degrees = ((y - self.crosshair_y)/self.height)*self.fovy
        return [x_degrees, y_degrees]

    def deg_to_pixels(self, x_degrees, y_degrees):
        """ Returns screen location of given degrees of deviation from center """
        x = (x_degrees / self.fovx)*self.width + self.crosshair_x
        y = (y_degrees / self.fovy)*self.height + self.crosshair_y
        return [x, y]

    def add_jitter(self, dx, dy, noise_std=0.2):
        """
        Add small amount of gaussian jitter to the movement, 
        depending on the strength of the jump
        """
        dx += gauss(0, noise_std * dx + 0.00001)
        dy += gauss(0, noise_std * dy + 0.00001)
        return dx, dy

    def is_inside_allow_angle(self, dx, dy):
        """
        Check if the amount of movement we are about to do
        is inside allowed limits (self.max_aim_angle).
        """
        if abs(dx) < self.max_aim_angle and abs(dy) < self.max_aim_angle:
            return True

    def get_aiming(self, enemies):
        """ 
        Returns dx,dy where dx,dy is amount of degrees
        to move aim
        
        Parameters:
            enemies - List of [x,y] of visible enemies,
                      where x,y are the coordinates of 
                      enemy on screen
        Returns:
            dx,dy - Amount of degrees move should be moved
        """
        raise NotImplementedError

    def process_last_action(self, actions):
        """
        Receive last_actions from the game, 
        in case we want to keep track of something
        """
        pass

class SnapAimbot(Aimbot):
    """ Snap aim: Find closest target, and aim to it """

    def get_aiming(self, enemies):
        target = select_closest_enemy(enemies, self.crosshair_x, self.crosshair_y)
        return self.pixels_to_deg(target[0], target[1])


class MaxStepAimbot(Aimbot):
    """ 
    Snap with max steps: Find closest target and aim towards it 
    at most with some step size 
    """
    def __init__(self, width, height, fovx, fovy, crosshair_x, crosshair_y, max_hop=5,
                 max_aim_angle=DEFAULT_MAX_ANGLE):
        super().__init__(width, height, fovx, fovy, crosshair_x, crosshair_y, max_aim_angle)
        self.max_hop = max_hop

    def get_aiming(self, enemies):
        target = select_closest_enemy(enemies, self.crosshair_x, self.crosshair_y)
        dx,dy = self.pixels_to_deg(target[0], target[1])
        if not self.is_inside_allow_angle(dx, dy):
            return 0, 0

        # Limit amount of movement
        dx = sign(dx)*min(abs(dx), self.max_hop)
        dy = sign(dy)*min(abs(dy), self.max_hop)
        return self.add_jitter(dx,dy)


class RelativeAimbot(Aimbot):
    """ 
    Relative moving towards target:
        - Find closest
        - Move towards target relative to distance
    """
    def __init__(self, width, height, fovx, fovy, crosshair_x, crosshair_y, hop_frac=0.4, min_hop=1, 
                 max_aim_angle=DEFAULT_MAX_ANGLE):
        super().__init__(width, height, fovx, fovy, crosshair_x, crosshair_y, max_aim_angle)
        self.hop_frac = hop_frac
        self.min_hop = min_hop

    def get_aiming(self, enemies):
        target = select_closest_enemy(enemies, self.crosshair_x, self.crosshair_y)
        dx,dy = self.pixels_to_deg(target[0], target[1])
        if not self.is_inside_allow_angle(dx, dy):
            return 0, 0
        # Move relative to distance to target
        dx = dx*self.hop_frac if dx > self.min_hop else dx
        dy = dy*self.hop_frac if dy > self.min_hop else dy
        return self.add_jitter(dx,dy)


class GANAimbot(Aimbot):
    """
    Aimbot based on the GANs trained on the datasets.
    """
    def __init__(self, width, height, fovx, fovy, crosshair_x, crosshair_y, model_path,
                 num_previous_steps=20, num_outputs=5, max_aim_angle=DEFAULT_MAX_ANGLE):
        super().__init__(width, height, fovx, fovy, crosshair_x, crosshair_y, max_aim_angle)
        # Wonky stuff here because pyinstaller catched these imports and included whole of
        # PyTorch in the mix...
        raise RuntimeError("Remove commented imports below this")
        #from aimbots.humanlike_aimbot_gan import AimbotGenerator, LATENT_SIZE
        #import torch
        self.aimbot_generator = AimbotGenerator(
            LATENT_SIZE,
            num_previous_steps * 2 + 2,
            num_outputs * 2
        )
        self.aimbot_generator.load_state_dict(torch.load(model_path, map_location="cpu"))

        # Fixed random latent for generator
        self.latent = np.random.normal(size=(LATENT_SIZE,))

        self.num_outputs = num_outputs
        self.num_previous_steps = num_previous_steps

        self.last_steps = deque(maxlen=num_previous_steps)
        # Fill with zeros
        for i in range(num_previous_steps + 1):
            self.last_steps.append((0,0))

    def get_aiming(self, enemies):
        target = select_closest_enemy(enemies, self.crosshair_x, self.crosshair_y)
        dx, dy = self.pixels_to_deg(target[0], target[1])
        if not self.is_inside_allow_angle(dx, dy):
            return 0, 0
        # Build input vector
        input_features = np.array(self.last_steps).ravel()
        target = np.array((dx, dy))
        condition =  np.concatenate((input_features, target))

        prediction = self.aimbot_generator.forward_np(self.latent[None], condition[None])[0]

        # Take first step of the prediction as the
        # step we will take
        dx = prediction[0]
        dy = prediction[1]

        return dx, dy

    def process_last_action(self, actions):
        # Store the last mouse movements 
        # TODO hardcoded indices
        self.last_steps.append((
            actions[4],
            actions[5]
        ))


class GANAimbotNP(Aimbot):
    """
    Aimbot based on the GANs trained on the datasets.
    """
    def __init__(self, width, height, fovx, fovy, crosshair_x, crosshair_y, model_path,
                 num_previous_steps=20, num_outputs=5, max_aim_angle=DEFAULT_MAX_ANGLE):
        super().__init__(width, height, fovx, fovy, crosshair_x, crosshair_y, max_aim_angle)
        parameters = np.load(model_path, allow_pickle=True)
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

        # Fixed random latent for generator (known size)
        self.latent = np.random.normal(size=(16,))

        self.num_outputs = num_outputs
        self.num_previous_steps = num_previous_steps

        self.last_steps = deque(maxlen=num_previous_steps)
        # Fill with zeros
        for i in range(num_previous_steps + 1):
            self.last_steps.append((0,0))

    def get_aiming(self, enemies):
        target = select_closest_enemy(enemies, self.crosshair_x, self.crosshair_y)
        dx, dy = self.pixels_to_deg(target[0], target[1])
        if not self.is_inside_allow_angle(dx, dy):
            return 0, 0
        # Build input vector
        input_features = np.array(self.last_steps).ravel()
        target = np.array((dx, dy))
        x = np.concatenate((self.latent, input_features, target))

        prediction = fnn_with_elus(x, self.weights, self.biases)

        # Take first step of the prediction as the
        # step we will take
        dx = prediction[0]
        dy = prediction[1]

        return dx, dy

    def process_last_action(self, actions):
        # Store the last mouse movements 
        # TODO hardcoded indices
        self.last_steps.append((
            actions[4],
            actions[5]
        ))
