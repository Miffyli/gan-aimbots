#
# plot_paper.py
# Hardcoded functions for plotting the final
# plots for the paper
#
import os
import pickle
from argparse import ArgumentParser
from glob import glob
import random
import json

import numpy as np
from matplotlib import pyplot
import sklearn.metrics
import scipy
from scipy.stats import ttest_ind

# Different directories.
# The non-GAN folders contain data from the first data collection
RECORDINGS_DIR = "data/data_collection_1"
GAN_RECORDINGS_DIR = "data/data_collection_2"
PERFORMANCE_RECORDINGS_DIR = "data/data_collection_3"
FEATURES_DIR = "features"
GAN_FEATURES_DIR = "gan_features"
HUMAN_GRADING_DIR = "data/data_collection_5"

# Aimangle delta indexes in actions
AIMANGLE_DELTA_YAW_IDX = 4
AIMANGLE_DELTA_PITCH_IDX = 5

# Mapping from aimbot_class integer to name of the aimbot
AIMBOT_FILE_NAMES = {
    0: None,
    1: "ease_light",
    2: "ease_strong",
    3: "gan",
    4: "gan_light",
    10: "gan_group0",
    11: "gan_group1"
}
AIMBOT_NAMES = {
    0: "None",
    1: "Light",
    2: "Strong",
    3: "GAN Strong",
    4: "GAN Light",
    10: "GAN (Group1)",
    11: "GAN (Group2)"
}

# Plotting constants
TITLE_KWARGS = dict(fontsize=27)
LEGEND_KWARGS = dict(fontsize=22)
TICK_PARAMS_KWARGS = dict(axis='both', which='both', labelsize=23)
LABEL_KWARGS = dict(fontsize=27)

# Colors the worst case -> known attack -> oracle -> best case lines
SCENARIO_COLORS = ["C3", "C1", "C0", "C2"]


def compute_fpr_fnr(bona_fide_scores, aimbot_scores):
    """
    Compute and return FPR and FNR points
    for system with given bona_fide (non-target)
    and aimbot (target) scores.
    Returns two arrays: fpr and fnr.
    """
    labels = np.concatenate(
        (
            np.zeros((bona_fide_scores.shape[0],)),
            np.ones((aimbot_scores.shape[0],))
        )
    ).astype(np.int64)
    all_scores = np.concatenate((bona_fide_scores, aimbot_scores))

    fpr, fnr, thresholds = sklearn.metrics.det_curve(labels, all_scores)
    return fpr, fnr


def compute_mindcf_eer(bona_fide_scores, aimbot_scores, hacker_prior):
    """
    Compute min DCF and EER of given bona fide (non-target) and hacker
    scores (target) under the given hacker_prior.
    Returns minDCF and eer (scalars).
    """
    # Import SIDEKIT here to avoid importing it when library is imported
    import sidekit

    # fast_minDCF function will take sigmoid of the prior,
    # so we take the inverse here (logit)
    logit_hacker_prior = np.log(hacker_prior / (1 - hacker_prior))

    results = sidekit.bosaris.fast_minDCF(aimbot_scores, bona_fide_scores, logit_hacker_prior, normalize=True)
    mindcf = results[0]
    eer = results[-1]
    return mindcf, eer


def print_metrics():
    """
    Calculate EERs and MinDCFs for the
    different scenarios
    """
    original_data = np.load("classification_results/dnn_scores.npz")
    worst_case = np.load("evaluation_scores/worst_case.npz")

    worst_case_scores = np.concatenate([data["test_scores"] for data in [original_data, worst_case]], axis=0)
    worst_case_aimbots = np.concatenate([data["test_aimbots"] for data in [original_data, worst_case]], axis=0)
    worst_case_data = {"test_scores": worst_case_scores, "test_aimbots": worst_case_aimbots}

    group1_data = np.load("evaluation_scores/known_attack_group1.npz")
    group2_data = np.load("evaluation_scores/known_attack_group2.npz")

    best_case_data = np.load("evaluation_scores/best_case.npz")

    train_light_data = np.load("evaluation_scores/trained_on_light.npz")
    train_strong_data = np.load("evaluation_scores/trained_on_strong.npz")
    best_case_original = np.load("evaluation_scores/best_case_original.npz")

    # We need to go over:
    # - Different aimbots (light, strong, gan1 and gan2)
    # - Different scenarios (worst-case, best case etc)
    # - EER and DCF
    # - Different priors for DCF
    # EER and DCF on x-axis
    # aimbots and scenarios on y-axis
    P_HACKERS = [0.5, 0.25, 0.1, 0.01]

    header_print_template = "{:<15}  {:<15}  {:<15}  {:<15}  {:<15}  {:<15}  {:<15}"
    print_template = "{:<15}& {:<15}& {:<15.2f}& {:<15.4f}& {:<15.4f}& {:<15.4f}& {:<15.4f}"

    # Print header
    print(header_print_template.format(
        *[
            "Aimbot",
            "Scenario",
            "EER(%)",
        ] + ["minDCF(p={})".format(p) for p in P_HACKERS]
    ))

    # Maps scenario name to mapping, that tells
    # which data should be used for aimbot
    scenarios = {
        "Worst-case": {
            # Nothing for light and strong aimbot here
            10: worst_case_data,
            11: worst_case_data
        },
        "Known-attack": {
            1: train_strong_data,
            2: train_light_data,
            10: group2_data,
            11: group1_data
        },
        "Oracle": {
            1: train_light_data,
            2: train_strong_data,
            10: group1_data,
            11: group2_data
        },
        "Train-on-test": {
            1: best_case_original,
            2: best_case_original,
            10: best_case_data,
            11: best_case_data
        },
    }

    for aimbot_class in [1, 2, 10, 11]:
        for scenario_name, scenario_mapping in scenarios.items():
            data = scenario_mapping.get(aimbot_class)
            if data is None:
                # Print emptys
                print(print_template.format(AIMBOT_NAMES[aimbot_class], scenario_name, *([np.nan] * (len(P_HACKERS) + 1))))
                continue
            bona_fide_scores = data["test_scores"][data["test_aimbots"] == 0, 1]
            aimbot_scores = data["test_scores"][data["test_aimbots"] == aimbot_class, 1]

            eer = None
            mindcfs = []
            for p_hacker in P_HACKERS:
                # EER will always be same so we can just
                # use the latest
                mindcf, eer = compute_mindcf_eer(bona_fide_scores, aimbot_scores, p_hacker)
                mindcfs.append(mindcf)
            eer = eer * 100

            # Remove whitespaces for spreadsheets not to flip out
            aimbot_name = AIMBOT_NAMES[aimbot_class].replace(" ", "")
            print(print_template.format(aimbot_name, scenario_name, eer, *mindcfs))


def plot_dets():
    """
    Plot the DET curves for classifier
    accuracy with and without GAN aimbots etc.

    Assume GAN classifiers have been trained and evaluated,
    and that results are in evaluation_scores.

    DET plotting code and adjustments are taken from scikit-learn:
    https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_plot/det_curve.py
    """

    DET_TICKS = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    DET_TICKS_LOCATIONS = scipy.stats.norm.ppf(DET_TICKS)

    def adjust_ax_for_det(ax):
        """Adjust given axis to pretty-show DET plots"""
        # Code copied directly from the scikit-learn det_curve.py
        tick_labels = [
            '{:.0f}'.format(100 * s) for s in DET_TICKS
        ]
        ax.set_xticks(DET_TICKS_LOCATIONS)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(-3, 3)
        ax.set_yticks(DET_TICKS_LOCATIONS)
        ax.set_yticklabels(tick_labels)
        ax.set_ylim(-3, 3)

    original_data = np.load("classification_results/dnn_scores.npz")
    worst_case = np.load("evaluation_scores/worst_case.npz")

    worst_case_scores = np.concatenate([data["test_scores"] for data in [original_data, worst_case]], axis=0)
    worst_case_aimbots = np.concatenate([data["test_aimbots"] for data in [original_data, worst_case]], axis=0)
    worst_case_data = {"test_scores": worst_case_scores, "test_aimbots": worst_case_aimbots}

    group1_data = np.load("evaluation_scores/known_attack_group1.npz")
    group2_data = np.load("evaluation_scores/known_attack_group2.npz")

    best_case_data = np.load("evaluation_scores/best_case.npz")

    train_light_data = np.load("evaluation_scores/trained_on_light.npz")
    train_strong_data = np.load("evaluation_scores/trained_on_strong.npz")
    best_case_original = np.load("evaluation_scores/best_case_original.npz")

    fig, axs = pyplot.subplots(
        nrows=1,
        ncols=3,
        sharey="row",
        sharex="row",
        figsize=[3 * 6.4, 1 * 6.4]
    )

    # First plot: Original aimbots + GANs without training
    ax = axs[0]
    ax.grid(alpha=0.2)
    # Human scores are the second scores.
    for aimbot_class in [1, 2]:
        for i, data in enumerate([train_light_data, train_strong_data, best_case_original]):
            bona_fide_scores = data["test_scores"][data["test_aimbots"] == 0, 1]
            aimbot_scores = data["test_scores"][data["test_aimbots"] == aimbot_class, 1]
            fpr, fnr = compute_fpr_fnr(bona_fide_scores, aimbot_scores)
            style = "-" if aimbot_class == 1 else "--"
            # Special handling: For light aimbot
            # we need to flip the known-attack/oracle colors
            c = SCENARIO_COLORS[i + 1]
            if aimbot_class == 1:
                if i == 0:
                    # Oracle
                    c = SCENARIO_COLORS[2]
                elif i == 1:
                    # Known attack
                    c = SCENARIO_COLORS[1]

            ax.plot(
                scipy.stats.norm.ppf(fpr),
                scipy.stats.norm.ppf(fnr),
                c=SCENARIO_COLORS[i + 1],
                linestyle=style
            )
    ax.tick_params(**TICK_PARAMS_KWARGS)
    # Create bit wonkier legends
    lines = []
    legends = []
    legend_lines = [
        {"c": SCENARIO_COLORS[1], "style": "-", "name": "Known-attack"},
        {"c": SCENARIO_COLORS[2], "style": "-", "name": "Oracle"},
        {"c": SCENARIO_COLORS[3], "style": "-", "name": "Train-on-test"},
        # Super pretty way of doing an empty space in legend
        # Stackoverflow #28078846
        {"c": "w", "style": "-", "name": ""},
        {"c": "k", "style": "-", "name": "Light"},
        {"c": "k", "style": "--", "name": "Strong"},
    ]
    for legend_line in legend_lines:
        line, = ax.plot(fpr, fnr, c=legend_line["c"], linestyle=legend_line["style"])
        # Do not show in the plot
        line.remove()
        lines.append(line)
        legends.append(legend_line["name"])
    adjust_ax_for_det(ax)
    ax.legend(lines, legends, **LEGEND_KWARGS)
    ax.set_xlabel("False Positive Rate (%)", **LABEL_KWARGS)
    ax.set_ylabel("False Negative Rate (%)", **LABEL_KWARGS)
    ax.set_title("Heuristic aimbots", **TITLE_KWARGS)

    # Second plot: Group 1 results
    ax = axs[1]
    ax.grid(alpha=0.2)
    for aimbot_class in [10]:
        for i, data in enumerate([worst_case, group2_data, group1_data, best_case_data]):
            bona_fide_scores = data["test_scores"][data["test_aimbots"] == 0, 1]
            aimbot_scores = data["test_scores"][data["test_aimbots"] == aimbot_class, 1]
            fpr, fnr = compute_fpr_fnr(bona_fide_scores, aimbot_scores)
            ax.plot(
                scipy.stats.norm.ppf(fpr),
                scipy.stats.norm.ppf(fnr),
                c=SCENARIO_COLORS[i]
            )
    adjust_ax_for_det(ax)
    ax.tick_params(**TICK_PARAMS_KWARGS)
    ax.legend(["Worst-case", "Known attack", "Oracle", "Train-on-test"], **LEGEND_KWARGS)
    ax.set_xlabel("False Positive Rate (%)", **LABEL_KWARGS)
    ax.set_title("GAN, Group 1", **TITLE_KWARGS)

    # Third plot: Group 2 results
    ax = axs[2]
    ax.grid(alpha=0.2)
    for aimbot_class in [11]:
        for i, data in enumerate([worst_case_data, group1_data, group2_data, best_case_data]):
            bona_fide_scores = data["test_scores"][data["test_aimbots"] == 0, 1]
            aimbot_scores = data["test_scores"][data["test_aimbots"] == aimbot_class, 1]
            fpr, fnr = compute_fpr_fnr(bona_fide_scores, aimbot_scores)
            ax.plot(
                scipy.stats.norm.ppf(fpr),
                scipy.stats.norm.ppf(fnr),
                c=SCENARIO_COLORS[i]
            )
    adjust_ax_for_det(ax)
    ax.tick_params(**TICK_PARAMS_KWARGS)
    ax.legend(["Worst-case", "Known attack", "Oracle", "Train-on-test"], **LEGEND_KWARGS)
    ax.set_xlabel("False Positive Rate (%)", **LABEL_KWARGS)
    ax.set_title("GAN, Group 2", **TITLE_KWARGS)

    fig.tight_layout()
    fig.savefig("figures/dets.pdf", bbox_inches="tight", pad_inches=0.0)


def print_player_stats():
    """
    Go through recordings and extracted features, and print
    out player accuracy/performance (frags) with and without
    different aimbots
    """
    from feature_extraction import extract_vacnet

    # Assumes:
    #  - Performance recordings are in "performance_recordings/..."

    data_files = glob(os.path.join(PERFORMANCE_RECORDINGS_DIR, "*.json"))

    no_aimbot_frags = []
    light_aimbot_frags = []
    strong_aimbot_frags = []
    gan_aimbot_frags = []

    no_aimbot_accuracy = []
    light_aimbot_accuracy = []
    strong_aimbot_accuracy = []
    gan_aimbot_accuracy = []

    no_aimbot_weapon_distribution = []
    light_aimbot_weapon_distribution = []
    strong_aimbot_weapon_distribution = []
    gan_aimbot_weapon_distribution = []

    for filename in data_files:
        # Skip first two games which were used for warming up
        if "episode0" in filename or "episode1" in filename:
            continue
        data = json.load(open(filename, "rb"))
        frags = data["frags"][-1]
        aimbot = data["aimbot"]
        weapons = data["weapons"]
        weapons = np.eye(6)[np.array(weapons).astype(np.int) - 1]
        features = extract_vacnet(data, shots_per_feature=1, hor_only=False)
        hits = features[:, -1]
        accuracy = hits.mean()
        if aimbot == None:
            no_aimbot_frags.append(frags)
            no_aimbot_accuracy.append(accuracy)
            no_aimbot_weapon_distribution.append(weapons.mean(axis=0))
        elif aimbot == "ease_light":
            light_aimbot_frags.append(frags)
            light_aimbot_accuracy.append(accuracy)
            light_aimbot_weapon_distribution.append(weapons.mean(axis=0))
        elif aimbot == "ease_strong":
            strong_aimbot_frags.append(frags)
            strong_aimbot_accuracy.append(accuracy)
            strong_aimbot_weapon_distribution.append(weapons.mean(axis=0))
        elif aimbot == "gan_group0":
            gan_aimbot_frags.append(frags)
            gan_aimbot_accuracy.append(accuracy)
            gan_aimbot_weapon_distribution.append(weapons.mean(axis=0))
        else:
            raise ValueError("Unknown aimbot type {}".format(aimbot))

    assert len(no_aimbot_accuracy) == len(light_aimbot_accuracy) == len(strong_aimbot_accuracy) == len(gan_aimbot_accuracy)

    print("N={}".format(len(no_aimbot_frags)))

    print("no-aimbot frags:           {}".format(no_aimbot_frags))
    print("light-aimbot game frags:   {}".format(light_aimbot_frags))
    print("strong-aimbot game frags:  {}".format(strong_aimbot_frags))
    print("gan-aimbot game frags:     {}".format(gan_aimbot_frags))

    print("\nMean no-aimbot frags:           {:2.4f} +/- {:2.4f}".format(np.mean(no_aimbot_frags), np.std(no_aimbot_frags)))
    print("Mean light-aimbot game frags:   {:2.4f} +/- {:2.4f}".format(np.mean(light_aimbot_frags), np.std(light_aimbot_frags)))
    print("Mean strong-aimbot game frags:  {:2.4f} +/- {:2.4f}".format(np.mean(strong_aimbot_frags), np.std(strong_aimbot_frags)))
    print("Mean gan-aimbot game frags:     {:2.4f} +/- {:2.4f}".format(np.mean(gan_aimbot_frags), np.std(gan_aimbot_frags)))

    print("No vs. light-aimbot p-value:  {:.4f}".format(ttest_ind(no_aimbot_frags, light_aimbot_frags, equal_var=False, alternative="two-sided")[1]))
    print("No vs. strong-aimbot p-value: {:.4f}".format(ttest_ind(no_aimbot_frags, strong_aimbot_frags, equal_var=False, alternative="two-sided")[1]))
    print("No vs. gan-aimbot p-value:    {:.4f}".format(ttest_ind(no_aimbot_frags, gan_aimbot_frags, equal_var=False, alternative="two-sided")[1]))

    print("\nMean no-aimbot accuracy:           {:2.4f} +/- {:2.4f}".format(np.mean(no_aimbot_accuracy), np.std(no_aimbot_accuracy)))
    print("Mean light-aimbot game accuracy:   {:2.4f} +/- {:2.4f}".format(np.mean(light_aimbot_accuracy), np.std(light_aimbot_accuracy)))
    print("Mean strong-aimbot game accuracy:  {:2.4f} +/- {:2.4f}".format(np.mean(strong_aimbot_accuracy), np.std(strong_aimbot_accuracy)))
    print("Mean gan-aimbot game accuracy:     {:2.4f} +/- {:2.4f}".format(np.mean(gan_aimbot_accuracy), np.std(gan_aimbot_accuracy)))

    print("No vs. light-aimbot p-value:  {:.4f}".format(ttest_ind(no_aimbot_accuracy, light_aimbot_accuracy, equal_var=False, alternative="two-sided")[1]))
    print("No vs. strong-aimbot p-value: {:.4f}".format(ttest_ind(no_aimbot_accuracy, strong_aimbot_accuracy, equal_var=False, alternative="two-sided")[1]))
    print("No vs. gan-aimbot p-value:    {:.4f}".format(ttest_ind(no_aimbot_accuracy, gan_aimbot_accuracy, equal_var=False, alternative="two-sided")[1]))

    print("\n                            Fist Pist Shot Mini Rock Plas")
    print("Mean no-aimbot weapons:     {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(*np.mean(no_aimbot_weapon_distribution, axis=0).tolist()))
    print("Mean light-aimbot weapons:  {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(*np.mean(light_aimbot_weapon_distribution, axis=0).tolist()))
    print("Mean strong-aimbot weapons: {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(*np.mean(strong_aimbot_weapon_distribution, axis=0).tolist()))
    print("Mean gan-aimbot weapons:    {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}".format(*np.mean(gan_aimbot_weapon_distribution, axis=0).tolist()))



def player_performance_vs_detection():
    """
    Analyse correlation between player's bona fide performance
    and detectability with different aimbots (e.g., "will weaker players be easier to detect when using aimbot"?)
    """
    from feature_extraction import extract_vacnet
    import torch

    # Assumes:
    #  - Performance recordings are in "performance_recordings/..."

    TORCH_MODEL_PATH = "gan_classification_results/dnn_group0_model.pkl"
    DATA_NORMALIZATION_PATH = "gan_classification_results/feature_normalization.npz"

    data_files = glob(os.path.join(PERFORMANCE_RECORDINGS_DIR, "*.json"))
    normalization_stats = np.load(DATA_NORMALIZATION_PATH)
    model = torch.load(TORCH_MODEL_PATH)

    no_aimbot_frags = {}
    no_aimbot_accuracy = {}

    no_aimbot_detection_scores = {}
    light_aimbot_detection_scores = {}
    strong_aimbot_detection_scores = {}
    gan_aimbot_detection_scores = {}

    for filename in data_files:
        # Skip first two games which were used for warming up
        if "episode0" in filename or "episode1" in filename:
            continue
        player_id = "_".join(os.path.basename(filename).split("_")[:2])
        data = json.load(open(filename, "rb"))
        frags = data["frags"][-1]
        aimbot = data["aimbot"]
        weapons = data["weapons"]
        weapons = np.eye(6)[np.array(weapons).astype(np.int) - 1]
        features = extract_vacnet(data, shots_per_feature=1, hor_only=False)
        hits = features[:, -1]
        accuracy = hits.mean()
        normalized_features = (features - normalization_stats["means"]) / normalization_stats["stds"]
        scores = model(torch.from_numpy(normalized_features).float()).detach().numpy()[:, 1]
        mean_score = scores.mean()
        if aimbot == None:
            no_aimbot_frags[player_id] = frags
            no_aimbot_accuracy[player_id] = accuracy
            no_aimbot_detection_scores[player_id] = mean_score
        elif aimbot == "ease_light":
            light_aimbot_detection_scores[player_id] = mean_score
        elif aimbot == "ease_strong":
            strong_aimbot_detection_scores[player_id] = mean_score
        elif aimbot == "gan_group0":
            gan_aimbot_detection_scores[player_id] = mean_score
        else:
            raise ValueError("Unknown aimbot type {}".format(aimbot))

    assert len(no_aimbot_detection_scores) == len(light_aimbot_detection_scores) == len(strong_aimbot_detection_scores) == len(gan_aimbot_detection_scores)

    player_ids = list(no_aimbot_detection_scores.keys())
    no_aimbot_detection_scores = [no_aimbot_detection_scores[player_id] for player_id in player_ids]
    light_aimbot_detection_scores = [light_aimbot_detection_scores[player_id] for player_id in player_ids]
    strong_aimbot_detection_scores = [strong_aimbot_detection_scores[player_id] for player_id in player_ids]
    gan_aimbot_detection_scores = [gan_aimbot_detection_scores[player_id] for player_id in player_ids]
    no_aimbot_frags = [no_aimbot_frags[player_id] for player_id in player_ids]
    no_aimbot_accuracy = [no_aimbot_accuracy[player_id] for player_id in player_ids]

    fig, axs = pyplot.subplots(nrows=5, ncols=2, figsize=(5.0 * 2, 3.2 * 5))

    for colum_idx, frag_or_accuracy in enumerate(("Kills", "Accuracy")):
        no_aimbot_x_axis = no_aimbot_frags if frag_or_accuracy == "Kills" else no_aimbot_accuracy

        axs[0, colum_idx].scatter(no_aimbot_x_axis, no_aimbot_detection_scores, label="No Aimbot", color="blue")
        axs[0, colum_idx].scatter(no_aimbot_x_axis, light_aimbot_detection_scores, label="Light Aimbot", color="green")
        axs[0, colum_idx].scatter(no_aimbot_x_axis, strong_aimbot_detection_scores, label="Strong Aimbot", color="red")
        axs[0, colum_idx].scatter(no_aimbot_x_axis, gan_aimbot_detection_scores, label="GAN Aimbot", color="orange")
        axs[0, colum_idx].set_xlabel("{} (without aimbot)".format(frag_or_accuracy))
        axs[0, colum_idx].set_ylabel("Detection score\n(Higher = hacking)")
        axs[0, colum_idx].legend()

        axs[1, colum_idx].scatter(no_aimbot_x_axis, no_aimbot_detection_scores)
        axs[1, colum_idx].set_xlabel("{} (without aimbot)".format(frag_or_accuracy))
        axs[1, colum_idx].set_ylabel("Detection score\n(higher = hacking)")
        axs[1, colum_idx].set_title("No Aimbot (corr = {:.3f})".format(np.corrcoef(no_aimbot_x_axis, no_aimbot_detection_scores)[0, 1]))

        axs[2, colum_idx].scatter(no_aimbot_x_axis, light_aimbot_detection_scores)
        axs[2, colum_idx].set_xlabel("{} (without aimbot)".format(frag_or_accuracy))
        axs[2, colum_idx].set_ylabel("Detection score\n(higher = hacking)")
        axs[2, colum_idx].set_title("Light Aimbot (corr = {:.3f})".format(np.corrcoef(no_aimbot_x_axis, light_aimbot_detection_scores)[0, 1]))

        axs[3, colum_idx].scatter(no_aimbot_x_axis, strong_aimbot_detection_scores)
        axs[3, colum_idx].set_xlabel("{} (without aimbot)".format(frag_or_accuracy))
        axs[3, colum_idx].set_ylabel("Detection score\n(higher = hacking)")
        axs[3, colum_idx].set_title("Strong Aimbot (corr = {:.3f})".format(np.corrcoef(no_aimbot_x_axis, strong_aimbot_detection_scores)[0, 1]))

        axs[4, colum_idx].scatter(no_aimbot_x_axis, gan_aimbot_detection_scores)
        axs[4, colum_idx].set_xlabel("{} (without aimbot)".format(frag_or_accuracy))
        axs[4, colum_idx].set_ylabel("Detection score\n(higher = hacking)")
        axs[4, colum_idx].set_title("GAN Aimbot (corr = {:.3f})".format(np.corrcoef(no_aimbot_x_axis, gan_aimbot_detection_scores)[0, 1]))

    pyplot.tight_layout()
    fig.savefig("figures/player_performance_vs_detection.png", dpi=200)


def multi_vector_classification():
    """
    Analysis of doing classification with multiple data vectors
    """
    from classification import get_player_id
    import torch

    VECTOR_AMOUNTS = list(range(1, 81, 1))
    N_REPEATS = 200

    LINE_NAMES = [
        "Light",
        "Strong",
        "GAN"
    ]
    AIMBOT_CLASS = [
        1,
        2,
        10
    ]
    TORCH_MODEL_PATHS = [
        "gan_classification_results/dnn_aimbot1_model.pkl",
        "gan_classification_results/dnn_aimbot2_model.pkl",
        "gan_classification_results/dnn_group0_model.pkl"
    ]
    DATA_NORMALIZATION_PATH = "gan_classification_results/feature_normalization.npz"
    TRAIN_TEST_SPLIT_FILE = "gan_classification_results/train_test_split.pkl"
    normalization_stats = np.load(DATA_NORMALIZATION_PATH)

    testing_ids = None
    with open(TRAIN_TEST_SPLIT_FILE, "rb") as f:
        split_data = pickle.load(f)
        testing_ids = split_data["testing_ids"]

    # Line names -> {"bonafide": bonafide_scores, "hacking":aimbot_scores}
    line_scores = {}
    for line_name, aimbot_class, torch_model_path in zip(LINE_NAMES, AIMBOT_CLASS, TORCH_MODEL_PATHS):
        model = torch.load(torch_model_path)
        feature_files = glob(os.path.join("gan_classification_data", "*"))
        bonafide_player_scores = []
        hacking_player_scores = []

        for feature_file in feature_files:
            player_id = get_player_id(feature_file)
            if player_id not in testing_ids:
                continue
            data = np.load(feature_file)
            # Aimbot is same over all samples
            aimbot_type = int(data["aimbot_class"][0])
            features = data["features"]

            if aimbot_type not in [0, aimbot_class]:
                continue

            normalized_features = (features - normalization_stats["means"]) / normalization_stats["stds"]
            scores = model(torch.from_numpy(normalized_features).float()).detach().numpy()[:, 1]

            if aimbot_type == 0:
                bonafide_player_scores.append(scores.tolist())
            else:
                hacking_player_scores.append(scores.tolist())
        line_scores[line_name] = {"bonafide": bonafide_player_scores, "hacking": hacking_player_scores}

    # Now, for each "VECTOR_AMOUNTS" (number of points per player)
    # we repeat N_REPEATS times
    # we take vector_amount points per player by sampling, average scores and try to do classifying

    line_eers = dict((name, []) for name in LINE_NAMES)
    line_stds = dict((name, []) for name in LINE_NAMES)

    for n_vectors in VECTOR_AMOUNTS:
        for line_name in LINE_NAMES:
            bonafide_scores = line_scores[line_name]["bonafide"]
            hacking_scores = line_scores[line_name]["hacking"]
            eers = []
            for _ in range(N_REPEATS):
                average_bonafide_scores = [np.mean(random.sample(scores, n_vectors)) for scores in bonafide_scores]
                average_hacking_scores = [np.mean(random.sample(scores, n_vectors)) for scores in hacking_scores]
                mind_dcf, eer = compute_mindcf_eer(np.array(average_bonafide_scores), np.array(average_hacking_scores), 0.5)
                eers.append(eer * 100)
            line_eers[line_name].append(np.mean(eers))
            line_stds[line_name].append(np.std(eers))

    fig = pyplot.figure(figsize=[6.4 * 0.9, 4.8 * 0.55])
    ax = pyplot.gca()
    for line_name in LINE_NAMES:
        eers = np.array(line_eers[line_name])
        stds = np.array(line_stds[line_name])
        ax.plot(VECTOR_AMOUNTS, eers, label=line_name)
        ax.fill_between(VECTOR_AMOUNTS, np.clip(eers - stds, 0, None), eers + stds, alpha=0.2)
    ax.set_ylim(-2, 22)
    ax.grid(alpha=0.2)
    ax.legend(fontsize="large")
    ax.set_xlabel("Number of features per game", fontsize="x-large")
    ax.set_ylabel("Equal error rate (%)", fontsize="x-large")
    ax.tick_params(axis='both', which='both', labelsize="large")
    pyplot.tight_layout()
    fig.savefig("figures/multi_vector_classification.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_mouse_analysis():
    """
    Analyze mouse movement of bona fide and hacking players
    """
    import matplotlib.colors as mcolors
    import scipy.stats

    AXIS_RANGE = 5

    # Load data
    recordings = glob(os.path.join(RECORDINGS_DIR, "*"))
    gan_recordings = glob(os.path.join(GAN_RECORDINGS_DIR, "*"))

    bona_fide_mouse_movement = []
    heuristic_aimbot_mouse_movement = []
    gan_aimbot_mouse_movement = []

    for filename in (recordings + gan_recordings):
        data = json.load(open(filename, "rb"))
        actions = data["actions"]
        # Take yaw and pitch
        mouse_movements = np.array([
            (a[AIMANGLE_DELTA_YAW_IDX], a[AIMANGLE_DELTA_PITCH_IDX]) for a in actions
        ])
        if "episode0" in filename or "episode1" in filename:
            # Bona fide gameplay
            bona_fide_mouse_movement.append(mouse_movements)
        else:
            if filename in recordings:
                # Heuristic aimbot
                heuristic_aimbot_mouse_movement.append(mouse_movements)
            else:
                gan_aimbot_mouse_movement.append(mouse_movements)

    bona_fide_individual_data = bona_fide_mouse_movement
    heuristic_aimbot_individual_data = heuristic_aimbot_mouse_movement
    gan_aimbot_individual_data = gan_aimbot_mouse_movement

    bona_fide_mouse_movement = np.concatenate(bona_fide_mouse_movement, axis=0)
    heuristic_aimbot_mouse_movement = np.concatenate(heuristic_aimbot_mouse_movement, axis=0)
    gan_aimbot_mouse_movement = np.concatenate(gan_aimbot_mouse_movement, axis=0)

    # Plot and print out some results
    figure, axs = pyplot.subplots(
        nrows=1,
        ncols=3,
        sharey="all",
        figsize=[4.8 * 3, 4.8]
    )

    # Put data and names in lists we will index in loop
    datas = [
        bona_fide_mouse_movement,
        heuristic_aimbot_mouse_movement,
        gan_aimbot_mouse_movement
    ]

    individual_datas = [
        bona_fide_individual_data,
        heuristic_aimbot_individual_data,
        gan_aimbot_individual_data
    ]

    titles = [
        "Bona fide",
        "Heuristic aimbot",
        "GAN aimbot"
    ]

    for i in range(3):
        data = datas[i]
        individual_data = individual_datas[i]
        title = titles[i]
        ax = axs[i]

        yaws = data[:, 0]
        pitches = data[:, 1]

        # Print out some basic stats
        print("Statistics for {}".format(title))
        print("\tYaw       {:2.4f} +/- {:2.4f}".format(yaws.mean(), yaws.std()))
        print("\tPitch     {:2.4f} +/- {:2.4f}".format(pitches.mean(), pitches.std()))
        print("\t|Yaw|     {:2.4f} +/- {:2.4f}".format(np.abs(yaws).mean(), np.abs(yaws).std()))
        print("\t|Pitch|   {:2.4f} +/- {:2.4f}".format(np.abs(pitches).mean(), np.abs(pitches).std()))
        print("\tCorr + p  {:.5f} ({:.5f})".format(*scipy.stats.pearsonr(np.abs(yaws), np.abs(pitches))))

        yaw_diff_corr = np.mean([scipy.stats.pearsonr(x[:-1, 0], x[1:, 0])[0] for x in individual_data])
        pitch_diff_corr = np.mean([scipy.stats.pearsonr(x[:-1, 1], x[1:, 1])[0] for x in individual_data])
        print("\tStep Corr  {:.5f} {:.5f}".format(yaw_diff_corr, pitch_diff_corr))
        print("\tStep Corr avg. {:.5f}".format((yaw_diff_corr + pitch_diff_corr) / 2))

        # Remove zero-movements from the plot
        zeros = (yaws == 0) & (pitches == 0)
        yaws = yaws[~zeros]
        pitches = pitches[~zeros]

        ax.hist2d(
            yaws,
            pitches,
            bins=50,
            range=((-AXIS_RANGE, AXIS_RANGE), (-AXIS_RANGE, AXIS_RANGE)),
            norm=mcolors.PowerNorm(0.5),
            density=True
        )
        ax.set_title(title)

    pyplot.tight_layout()
    figure.savefig("figures/mouse_dist.png", dpi=200)


def plot_trajectories():
    """
    Plot bunch of example trajectories from each aimbot category.
    """

    EXAMPLES_PER_CATEGORY = 10

    # Load data
    recordings = glob(os.path.join(FEATURES_DIR, "*"))
    gan_recordings = glob(os.path.join(GAN_FEATURES_DIR, "*"))

    bona_fide_features = []
    heuristic_aimbot_features = []
    gan_aimbot_features = []

    for filename in (recordings + gan_recordings):
        data = np.load(filename)
        features = data["features"]
        if "episode0" in filename or "episode1" in filename:
            # Bona fide gameplay
            bona_fide_features.append(features)
        elif "episode3" in filename and filename in recordings:
            # Add strong aimbots to heuristic aimbots
            heuristic_aimbot_features.append(features)
        elif "episode2" in filename and filename in recordings:
            # Skip light aimbots
            pass
        else:
            # Recording is from gan_aimbot
            gan_aimbot_features.append(features)

    bona_fide_features = np.concatenate(bona_fide_features, axis=0)
    heuristic_aimbot_features = np.concatenate(heuristic_aimbot_features, axis=0)
    gan_aimbot_features = np.concatenate(gan_aimbot_features, axis=0)

    figure, axs = pyplot.subplots(
        nrows=3,
        ncols=EXAMPLES_PER_CATEGORY,
        figsize=[4.8 * (EXAMPLES_PER_CATEGORY / 3), 4.8],
        sharex="none",
        sharey="none"
    )

    titles = [
        "Bona fide",
        "Strong\naimbot",
        "GAN\naimbot"
    ]

    datas = [
        bona_fide_features,
        heuristic_aimbot_features,
        gan_aimbot_features
    ]

    for type_i in range(3):
        data = datas[type_i]
        for example_i in range(EXAMPLES_PER_CATEGORY):
            ax = axs[type_i, example_i]
            # Pick random feature
            random_pick = data[random.randint(0, len(data) - 1)]

            # Turn the feature vector back into trajectory.
            # Assuming VAC-net-like features with a ton of hardcoding
            trajectory = np.array([random_pick[:25], random_pick[25:50]]).T
            trajectory = np.cumsum(trajectory, axis=0)
            # Center around the point where we shot
            trajectory -= trajectory[16]
            # Plot
            ax.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)

            colors = ["g"] + (["b"] * 15) + ["m"] + (["b"] * 7) + ["r"]
            sizes = [50] + ([7] * 15) + [50] + ([7] * 7) + [50]
            ax.axis("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.scatter(trajectory[:, 0], trajectory[:, 1], s=sizes, c=colors)

            # Draw again but only important bits to overwrite over blues
            sizes = [50] + ([0] * 15) + [50] + ([0] * 7) + [50]
            ax.scatter(trajectory[:, 0], trajectory[:, 1], s=sizes, c=colors)

            if example_i == 0:
                # Add titles
                ax.set_ylabel(titles[type_i], fontsize=20)

    pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=0.03)
    figure.savefig("figures/mouse_trajectories.pdf", bbox_inches="tight", pad_inches=0.0)


def plot_human_grading():
    """
    Plot the opinion-scores of the recordings being hackers.
    """

    ground_truth_data = json.load(open(os.path.join(HUMAN_GRADING_DIR, "ground_truth.json")))
    ground_truth_aimbots = [x["aimbot-name"] for x in ground_truth_data]
    ground_truth_aimbots = [x if x != "none" else None for x in ground_truth_aimbots]
    aimbot_file_names_transposed = dict((v, k) for k, v in AIMBOT_FILE_NAMES.items())
    ground_truth_aimbot_labels = [aimbot_file_names_transposed[x] for x in ground_truth_aimbots]
    ground_truth_aimbot_labels = np.array(ground_truth_aimbot_labels)

    # Load answers.
    # Assume answers in same order as the ground-truth items.
    # Also offset results to [0, 2].
    experienced_judge_answers = []
    fps_gamer_answers = []
    for filepath in glob(os.path.join(HUMAN_GRADING_DIR, "answers", "experienced_judges", "*")):
        experienced_judge_answers.append(np.loadtxt(filepath)[:, 1] - 1)
    for filepath in glob(os.path.join(HUMAN_GRADING_DIR, "answers", "fps_gamers", "*")):
        fps_gamer_answers.append(np.loadtxt(filepath)[:, 1] - 1)

    # Average/std of the grading
    experienced_judge_means = []
    experienced_judge_stds = []
    fps_gamer_means = []
    fps_gamer_stds = []
    aimbot_names = [
        "None",
        "GAN",
        "Light",
        "Strong",
    ]
    aimbot_labels = [
        aimbot_file_names_transposed[None],
        aimbot_file_names_transposed["gan_group0"],
        aimbot_file_names_transposed["ease_light"],
        aimbot_file_names_transposed["ease_strong"],
    ]

    # Matrix of percentages (aimbot_type, answer).
    # Hard-coded three answers
    experienced_judge_grading_ratios = np.zeros((len(aimbot_labels), 3))
    fps_gamer_grading_ratios = np.zeros((len(aimbot_labels), 3))

    # Get mean answers per aimbot.
    # We might want to change this to showing proportions...
    for aimbot_i, aimbot_label in enumerate(aimbot_labels):
        # Mask for the answers for this specific aimbot
        mask = ground_truth_aimbot_labels == aimbot_label
        experienced_judge_label_answers = np.concatenate([
            x[mask] for x in experienced_judge_answers
        ])
        fps_gamer_label_answers = np.concatenate([
            x[mask] for x in fps_gamer_answers
        ])
        experienced_judge_means.append(experienced_judge_label_answers.mean())
        experienced_judge_stds.append(experienced_judge_label_answers.std())
        fps_gamer_means.append(fps_gamer_label_answers.mean())
        fps_gamer_stds.append(fps_gamer_label_answers.std())
        # Also store ratio of different gradings
        for grade_i, grade in enumerate([0, 1, 2]):
            experienced_judge_grading_ratios[aimbot_i, grade_i] = np.mean(
                experienced_judge_label_answers == grade
            )
            fps_gamer_grading_ratios[aimbot_i, grade_i] = np.mean(
                fps_gamer_label_answers == grade
            )

    print("Experienced judge ratios (y = grade, y = aimbot)")
    print(experienced_judge_grading_ratios.T * 100)
    print("\nFPS gamer ratios")
    print(fps_gamer_grading_ratios.T * 100)

    # Plot results
    # Taking guidance from matplotlib tutorial
    #   https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    fig = pyplot.figure(figsize=[6.4 * 1.2, 4.8 * 1.2])
    x_range = np.arange(len(aimbot_names))
    width = 0.35
    pyplot.bar(
        x_range - width / 2,
        experienced_judge_means,
        width,
        yerr=experienced_judge_stds,
        label="Experienced\njudges"
    )
    pyplot.bar(
        x_range + width / 2,
        fps_gamer_means,
        width,
        yerr=fps_gamer_stds,
        label="FPS players"
    )

    ax = pyplot.gca()
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(("Not\nSuspicious", "Suspicious", "Cheating"))
    ax.set_xticks(x_range)
    ax.set_xticklabels(aimbot_names)
    ax.tick_params(**TICK_PARAMS_KWARGS)
    pyplot.legend(loc="upper left", **LEGEND_KWARGS)
    pyplot.grid(axis="y", alpha=0.2)

    pyplot.tight_layout()

    pyplot.savefig("figures/human_grading.pdf")


def print_dataset_statistics():
    """
    Print out statistics of our datasets (how many participants, how much data,
    how many features etc etc).

    NOTE: This assumes that we have ran all classification etc. code to produce
        train-test splits and whatnot.
    """
    from classification import get_player_id

    # This file exists after running GAN-classification stuff
    TRAIN_TEST_SPLIT_FILE = "gan_classification_results/train_test_split.pkl"
    HEURISTIC_FEATURES_DIR = "features"
    GAN_FEATURES_DIR = "gan_features"

    training_ids = None
    testing_ids = None
    with open(TRAIN_TEST_SPLIT_FILE, "rb") as f:
        split_data = pickle.load(f)
        training_ids = split_data["training_ids"]
        testing_ids = split_data["testing_ids"]

    print("Total number of IDs: {} for training, {} for testing".format(len(training_ids), len(testing_ids)))

    data_collections = ["heuristic", "gan"]
    data_feature_dirs = [HEURISTIC_FEATURES_DIR, GAN_FEATURES_DIR]
    for data_collection, data_feature_dir in zip (data_collections, data_feature_dirs):
        print("Results for collection '{}'".format(data_collection))

        feature_files = glob(os.path.join(data_feature_dir, "*"))
        train_feature_sizes = []
        test_feature_sizes = []
        train_aimbot_feature_sizes = {}
        test_aimbot_feature_sizes = {}
        train_participants = set()
        test_participants = set()

        for feature_file in feature_files:
            player_id = get_player_id(feature_file)
            data = np.load(feature_file)
            num_features = len(data["features"])
            # Aimbot is same over all samples
            aimbot_type = data["aimbot_class"][0]

            if player_id in training_ids:
                # Sanity check
                if player_id in testing_ids:
                    raise RuntimeError("A player id exists both in testing and training set!")
                train_feature_sizes.append(num_features)
                train_participants.add(player_id)
                train_aimbot_feature_sizes[aimbot_type] = train_aimbot_feature_sizes.get(aimbot_type, []) + [num_features]
            elif player_id in testing_ids:
                test_feature_sizes.append(num_features)
                test_participants.add(player_id)
                test_aimbot_feature_sizes[aimbot_type] = test_aimbot_feature_sizes.get(aimbot_type, []) + [num_features]
            else:
                raise RuntimeError("A player ID was not assigned to testing or training set!")

        print("\tTraining set: {} participants, {} features".format(len(train_participants), sum(train_feature_sizes)))
        for aimbot_class, feature_counts in train_aimbot_feature_sizes.items():
            print("\t\tAimbot {}: {} features".format(aimbot_class, sum(feature_counts)))
        print("\tTesting set:  {} participants, {} features".format(len(test_participants), sum(test_feature_sizes)))
        for aimbot_class, feature_counts in test_aimbot_feature_sizes.items():
            print("\t\tAimbot {}: {} features".format(aimbot_class, sum(feature_counts)))


AVAILABLE_OPERATIONS = {
    "all": None,
    "dets": plot_dets,
    "classification-metrics": print_metrics,
    "player-stats": print_player_stats,
    "mouse-analysis": plot_mouse_analysis,
    "plot-trajectories": plot_trajectories,
    "human-grading": plot_human_grading,
    "data-statistics": print_dataset_statistics,
    "player-stats-detection": player_performance_vs_detection,
    "multi-vector-classification": multi_vector_classification
}

if __name__ == '__main__':
    parser = ArgumentParser("Different plot utils")
    parser.add_argument("operation", choices=list(AVAILABLE_OPERATIONS.keys()), help="Operation to run")
    args = parser.parse_args()

    if args.operation == "all":
        for plot_function in AVAILABLE_OPERATIONS.values():
            if plot_function is not None:
                plot_function()
    else:
        operation_fn = AVAILABLE_OPERATIONS[args.operation]
        operation_fn()
