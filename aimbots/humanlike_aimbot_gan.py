#
# humanlike_aimbot_gan.py
# The good ol' GAN-solution
#
import numpy as np
import json
import argparse
import torch
from torch import nn
import torch.distributions
import random
from tqdm import tqdm
from collections import deque

parser = argparse.ArgumentParser("Train humanlike aimbots (GAN-style)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--model", type=str, help="Path to input/output model")
parser.add_argument("operation", choices=["train", "visualize", "params_to_numpy"])
parser.add_argument("human_recordings", nargs="+", help=".pkl files containing human gameplay (from recording.py)")

# Trajectories are in format (dx1, dy1, dx2, dy2, ...)

BATCH_SIZE = 64
USE_GPU = True

# How many previous dx/dy will be included
# in the condition
NUM_PREVIOUS_FRAMES = 20

# For loading data from recordings
AIMANGLE_DELTA_YAW_IDX = 4
AIMANGLE_DELTA_PITCH_IDX = 5

# How many steps will be generated by GAN at any time
# In practice we would just use the first generated step
NUM_GENERATED_STEPS = 5

# Epsilon for numerical stability
EPSILON = 1e-5

# Random noise vector size
LATENT_SIZE = 16
# How many iterations between updating generator
GENERATOR_UPDATE_RATE = 5

# WGAN's weight clipping for discriminator.
# Taken from the original paper https://arxiv.org/pdf/1701.07875.pdf
WGAN_WEIGHT_CLIP = 0.01
WGAN_LEARNING_RATE = 0.00005


class AimbotGenerator(nn.Module):
    """
    The generator part of GAN, which will provide
    next hops.
    """

    def __init__(self,
                 latent_size,
                 condition_size,
                 dim_out,
                 use_gpu=False,
                 optimizer=torch.optim.RMSprop,
                 discriminator_loss_weight=1,
                 aimbot_loss_weight=1
                 ):
        """
        Args:
            latent_size (int): Size of the random input vector
            condition_size (int): Size of the condition vector
            dim_out (int): Number of features out
        """
        super().__init__()

        self.latent_size = latent_size
        self.condition_size = condition_size
        self.dim_out = dim_out

        self.discriminator_loss_weight = discriminator_loss_weight
        self.aimbot_loss_weight = aimbot_loss_weight

        # TODO hardcoded network
        # "+2" for the target
        self.head = torch.nn.Sequential(
            torch.nn.Linear(self.latent_size + self.condition_size, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, self.dim_out)
        )

        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

        self.optimizer = optimizer(self.parameters(), lr=WGAN_LEARNING_RATE)

        self.to(self.device)

    def forward(self, latent, condition):
        return self.head(torch.cat((latent, condition), dim=1))

    def forward_np(self, latent, condition):
        """ Forward but for numpy arrays """
        return self(torch.from_numpy(latent).float().to(self.device),
                    torch.from_numpy(condition).float().to(self.device)).cpu().detach().numpy()

    def train_on_batch(self, conditions, discriminator):
        if self.use_gpu:
            conditions = conditions.to(self.device)

        # Predict some "hacking" movement
        latents = torch.randn((conditions.shape[0], LATENT_SIZE)).to(self.device)
        predictions = self(latents, conditions)

        # Ask discriminator what it thinks about the generated stuff,
        # and we want to minimize this (think it is human)
        discriminator_scores = discriminator(predictions, conditions)
        D_loss = torch.mean(discriminator_scores) * self.discriminator_loss_weight

        # Aimbot-encouragement: After the generated steps we should
        # be close to the target
        # Predictions are mouse movements x1, y1, x2, y2, x3, ...
        # TODO nasty hardcoding: The target is last part of the condition
        targets = conditions[:, -2:]
        dist_to_target_x = torch.sum(predictions[:, ::2], dim=1) - targets[:, 0]
        dist_to_target_y = torch.sum(predictions[:, 1::2], dim=1) - targets[:, 1]
        total_distance_to_target = torch.sqrt(dist_to_target_x**2 + dist_to_target_y**2)

        aimbot_loss = torch.mean(total_distance_to_target) * self.aimbot_loss_weight

        total_loss = D_loss + aimbot_loss

        # Update params
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return D_loss.item(), aimbot_loss.item()


class AimbotDiscriminator(nn.Module):
    """
    The discriminator part of the GAN, attempts to distinguish
    between human players and whatever generator generates.
    Outputs 1 for "hack", 0 for "bona fide". Note that
    this flipped of the general notion with GANs, but
    done here to stay consistent with rest of the code.
    """

    def __init__(self,
                 dim_in,
                 condition_size,
                 use_gpu=False,
                 optimizer=torch.optim.RMSprop,
                 human_loss_weight=1,
                 bot_loss_weight=1):
        """
        Args:
            dim_in (int): Number of features in
        """
        super().__init__()

        self.human_loss_weight = human_loss_weight
        self.bot_loss_weight = bot_loss_weight

        # Hardcoded network based on the
        # classification network
        self.head = torch.nn.Sequential(
            torch.nn.Linear(dim_in + condition_size, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 1),
        ).cuda()

        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu else "cpu"

        self.optimizer = optimizer(self.parameters(), lr=WGAN_LEARNING_RATE)

        self.to(self.device)

    def forward(self, x, condition):
        # Return classification for trajectory
        # 0: Human
        # 1: Bot
        x = self.head(torch.cat((x, condition), dim=1))
        return x

    def forward_np(self, x, condition):
        """ Forward but for numpy arrays """
        return self(torch.from_numpy(x).float().to(self.device),
                    torch.from_numpy(condition).float().to(self.device)).cpu().detach().numpy()

    def train_on_batch(self, human_datas, conditions, generator):
        if self.use_gpu:
            human_datas = human_datas.to(self.device)
            conditions = conditions.to(self.device)

        human_predictions = self(human_datas, conditions)

        # Generate hack movement
        latents = torch.randn((human_datas.shape[0], LATENT_SIZE)).to(self.device)
        bot_trajectories = generator(latents, conditions)

        bot_predictions = self(bot_trajectories, conditions)

        # Minimize humans, maximize bots
        total_loss = (
            torch.mean(human_predictions) * self.human_loss_weight -
            torch.mean(bot_predictions) * self.bot_loss_weight
        )

        # Provide some kind of metrics how we are learning
        human_scores = torch.mean(human_predictions).item()
        bot_scores = torch.mean(bot_predictions).item()

        # Update params
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # WGAN weight clipping
        for p in self.parameters():
            p.data.clamp_(min=-WGAN_WEIGHT_CLIP, max=WGAN_WEIGHT_CLIP)

        return total_loss.item(), (human_scores, bot_scores)


def load_human_data(files):
    """
    Load samples of human trajectories from recorded files.
    Returns two arrays:
        condition (N x (NUM_PREVIOUS_STEPS*2 + 2)):
            previous mouse movements + target point
            (where player landed after NUM_GENERATED_STEPS)
        human_data (N x NUM_GENERATED_STEPS*2):
            How player moved to reach that target point
    Automatically skips files with aimbot enabled.
    """
    conditions = []
    human_data = []

    condition_size = NUM_PREVIOUS_FRAMES * 2
    human_data_size = NUM_GENERATED_STEPS * 2
    full_slice_size = condition_size + human_data_size
    for filename in tqdm(files):
        data = json.load(open(filename, "rb"))
        # Skip aimbotting ones
        if data["aimbot"] is not None:
            continue

        # Interleave yaws and pitches
        yaw_pitches = []
        for action in data["actions"]:
            yaw_pitches.append(action[AIMANGLE_DELTA_YAW_IDX])
            yaw_pitches.append(action[AIMANGLE_DELTA_PITCH_IDX])

        for idx, i in enumerate(range(full_slice_size, len(yaw_pitches), 2)):
            full_slice = yaw_pitches[i - full_slice_size:i]
            human_data_slice = full_slice[-human_data_size:]
            target_condition = [sum(human_data_slice[::2]), sum(human_data_slice[1::2])]
            condition = full_slice[:condition_size]

            conditions.append(np.array(condition + target_condition))
            human_data.append(np.array(human_data_slice))

    return np.array(conditions), np.array(human_data)


def main_visualize(args):
    from matplotlib import pyplot
    # Clip human data (we do not need much)
    human_data, targets = load_human_data(random.sample(args.human_recordings, 5))

    generator = AimbotGenerator(human_data.shape[1], NUM_GENERATED_STEPS * 2)
    generator.load_state_dict(torch.load(args.model + "_G"))

    # Pick random human trajectories, and compare predictions to
    # real steps
    idxs = list(range(len(human_data)))
    for rep in range(10):
        random_idx = random.choice(idxs)
        random_traj = human_data[random_idx]
        random_target = targets[random_idx]

        print(random_target)

        prediction = generator.forward_np(random_traj[None], random_target[None])[0]

        print(random_traj.reshape((-1, 2)))
        print(prediction.reshape((-1, 2)))

        input_traj = np.cumsum(random_traj.reshape((-1, 2)), axis=0)
        prediction = np.cumsum(prediction.reshape((-1, 2)), axis=0) + input_traj[-1]
        pyplot.axis("equal")
        pyplot.scatter(input_traj[:,0], input_traj[:,1])
        pyplot.scatter(input_traj[-1, 0] + random_target[0],
                       input_traj[-1, 1] + random_target[1],
                       c="g")
        pyplot.scatter(prediction[:, 0], prediction[:, 1], c="r")
        pyplot.show()
        pyplot.close()


def main_train(args):
    print("Loading human data...")
    conditions, human_data = load_human_data(args.human_recordings)
    human_data = human_data.astype(np.float32)
    conditions = conditions.astype(np.float32)

    human_data = torch.from_numpy(human_data)
    conditions = torch.from_numpy(conditions)

    condition_size = NUM_PREVIOUS_FRAMES * 2 + 2
    generated_size = NUM_GENERATED_STEPS * 2
    generator = AimbotGenerator(
        LATENT_SIZE,
        condition_size,
        generated_size,
        aimbot_loss_weight=1,
        discriminator_loss_weight=1,
        optimizer=torch.optim.Adam,
        use_gpu=USE_GPU
    )
    discriminator = AimbotDiscriminator(
        generated_size,
        condition_size,
        optimizer=torch.optim.Adam,
        human_loss_weight=1,
        bot_loss_weight=1,
        use_gpu=USE_GPU
    )

    # Two sets of random indices
    # one for D and one for G
    rand_idxs = np.arange(len(human_data))
    rand_idxs2 = np.arange(len(human_data))

    g_d_losses = deque(maxlen=100)
    g_aimbot_losses = deque(maxlen=100)
    d_losses = deque(maxlen=100)
    d_human_preds = deque(maxlen=100)
    d_bot_preds = deque(maxlen=100)

    for epoch in range(args.epochs):
        print("Epoch %d" % epoch)

        np.random.shuffle(rand_idxs)
        np.random.shuffle(rand_idxs2)

        iteration = 0
        for i in tqdm(range(0, len(rand_idxs) - BATCH_SIZE, BATCH_SIZE)):
            # ---Update D---
            batch_human_data = human_data[rand_idxs[i:i + BATCH_SIZE]]
            batch_conditions = conditions[rand_idxs[i:i + BATCH_SIZE]]

            d_loss, d_accs = discriminator.train_on_batch(
                batch_human_data,
                batch_conditions,
                generator
            )

            d_losses.append(d_loss)
            d_human_preds.append(d_accs[0])
            d_bot_preds.append(d_accs[1])

            # ---Update G---
            # Generate some dummy targets for aimbot to aim at
            if (iteration % GENERATOR_UPDATE_RATE) == 0:
                batch_conditions = conditions[rand_idxs2[i:i + BATCH_SIZE]]

                # First update generator
                g_d_loss, g_aimbot_loss = generator.train_on_batch(
                    batch_conditions,
                    discriminator
                )

                g_d_losses.append(g_d_loss)
                g_aimbot_losses.append(g_aimbot_loss)

            if (iteration % 100) == 0:
                tqdm.write("Iteration {}".format(iteration))
                tqdm.write("\tG_D loss (v):        {:.3f}".format(np.mean(g_d_losses)))
                tqdm.write("\tG_aimbot loss (v):   {:.3f}".format(np.mean(g_aimbot_losses)))
                tqdm.write("\tD loss (v):          {:.3f}".format(np.mean(d_losses)))
                tqdm.write("\tD human pred:        {:.3f}".format(np.mean(d_human_preds)))
                tqdm.write("\tD bot pred:          {:.3f}".format(np.mean(d_bot_preds)))

            iteration += 1

        if args.model is not None:
            torch.save(generator.state_dict(), args.model + "_G")
            torch.save(discriminator.state_dict(), args.model + "_D")


def params_to_numpy(args):
    """
    Take args.model, load the parameters
    and convert into nice numpy thing of
    weights and biases
    """
    params = torch.load(args.model)
    # Hardcoded generator parameter names
    weights = [
        params["head.0.weight"].cpu().numpy(),
        params["head.2.weight"].cpu().numpy(),
        params["head.4.weight"].cpu().numpy(),
    ]
    biases = [
        params["head.0.bias"].cpu().numpy(),
        params["head.2.bias"].cpu().numpy(),
        params["head.4.bias"].cpu().numpy(),
    ]
    output_file = args.model + "_numpy_params"
    np.savez(output_file, weights=weights, biases=biases)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.operation == "train":
        main_train(args)
    elif args.operation == "params_to_numpy":
        params_to_numpy(args)
    else:
        main_visualize(args)
