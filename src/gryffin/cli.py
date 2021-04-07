#!/usr/bin/env python

import numpy as np
import pandas as pd
import os
import shutil
import json
import argparse
from gryffin import Gryffin


# =============
# Parse Options
# =============
def parse_options():

    parser = argparse.ArgumentParser()

    parser.add_argument('-f',
                        dest='file',
                        type=str,
                        help='Excel/CSV file with all previous experiments.',
                        required=True)
    parser.add_argument('-c',
                        dest='json',
                        type=str,
                        help='Json configuration file.',
                        required=True)
    parser.add_argument('--num_experiments',
                        dest='num_experiments',
                        type=int,
                        help='Number of experiments requested. Default is 1.',
                        default=1)
    parser.add_argument('--num_cpus',
                        dest='num_cpus',
                        type=int,
                        help='Number of CPUs to use. Default is 1.',
                        default=1)
    parser.add_argument('--seed',
                        dest='random_seed',
                        type=int,
                        help='Random seed used for initialization. Default is 42.',
                        default=42)
    args = parser.parse_args()
    return args


# ====
# Main
# ====
def main(args):

    # load past experiments
    infile_extension = args.file.split('.')[-1]
    if infile_extension == 'csv':
        df_in = pd.read_csv(args.file)
    elif infile_extension in ['xls', 'xlsx']:
        df_in = pd.read_excel(args.file)

    print("    ================")
    print("    Past Experiments")
    print("    ================")
    print(df_in)

    # load params/objectives
    with open(args.json, 'r') as jsonfile:
        config = json.load(jsonfile)

    # check we have all right params/objs in the csv file
    obj_names = [obj['name'] for obj in config["objectives"]]
    param_names = [param['name'] for param in config["parameters"]]  # N.B. order matters
    for obj_name in obj_names:
        if obj_name not in df_in.columns:
            raise ValueError(f"Expected objective '{obj_name}' missing from {args.file}")
    for param_name in param_names:
        if param_name not in df_in.columns:
            raise ValueError(f"Expected parameter '{param_name}' missing from {args.file}")

    # init gryffin
    gryffin = init_objects(args, config)

    # ---------------------
    # Run Gryffin recommend
    # ---------------------
    print()
    print("    =====================================")
    print("    Running Experiment Planning Algorithm")
    print("    =====================================")

    if len(df_in) == 0:
        observations = []
        samples = suggest_next_experiments(gryffin, observations, args.num_experiments)
    else:
        # build observation list for Gryffin
        observations = build_observations(df_in, param_names, obj_names)

        # ask for next experiments
        samples = suggest_next_experiments(gryffin, observations, args.num_experiments)

    # ---------------------
    # Save samples to file
    # ---------------------

    # create df_samples
    df_samples = pd.DataFrame(columns=df_in.columns)

    for param_name in param_names:
        param_values = [sample[param_name][0] for sample in samples]
        df_samples.loc[:, param_name] = param_values

    print()
    print("    ====================")
    print("    Proposed Experiments")
    print("    ====================")
    print(df_samples)
        
    # append df_samples to df_in
    df_out = df_in.append(df_samples, ignore_index=True, sort=False)

    # make backup of result file
    bkp_file = f"backup_{args.file}"
    if os.path.isfile(bkp_file):
        os.remove(bkp_file)
    shutil.copy(args.file, bkp_file)

    # save new result file
    if infile_extension == 'csv':
        df_out.to_csv(args.file, index=False)
    elif infile_extension in ['xls', 'xlsx']:
        df_out.to_excel(args.file, index=False)


# =========
# Functions
# =========
def init_objects(args, config):

    config["general"] = {
                        "num_cpus": args.num_cpus,
                        "boosted": True,  # <----
                        "caching": True,  # <----
                        "batches": args.batches,
                        "sampling_strategies": args.sampling_strategies,
                        "auto_desc_gen": False,  # <----
                        "feas_approach": "fwa",  # <----
                        "feas_param": 1,  # < -----
                        "random_seed": args.random_seed,
                        "save_database": False,
                        "acquisition_optimizer": "adam",  # < -------
                        "verbosity": 3
                        }

    gryffin = Gryffin(config_dict=config)
    return gryffin


def suggest_next_experiments(gryffin, observations, num_experiments):

    samples = gryffin.recommend(observations=observations)

    # i.e. sampling_strategies * batches == num_experiments
    if len(samples) == num_experiments:
        return samples
    # i.e. sequential experiments with sampling_strategies==2, batches==1
    elif len(samples) > num_experiments:
        select = len(observations) % len(samples)
        return [samples[select]]
    else:
        raise ValueError()


def build_observations(df_observations, param_names, obj_names):

    observations = []

    for obs in df_observations.to_numpy():
        d = {}
        for param_name, value in zip(param_names, obs):
            d[param_name] = np.array([value])
        for obj_name, value in zip(obj_names, obs):
            d[obj_name] = value
        observations.append(d)

    return observations


def entry_point():
    args = parse_options()
    main(args)


if __name__ == "__main__":
    entry_point()


