import pickle
import numpy as np
import pandas as pd
import os

from gryffin import Gryffin
budget = 192
sampling_strategies = np.array([-1, 1])
with_desc = True
dynamic = True
random_seed = 2020

# the categorical options corresponding to the minimum bandgap in the dataset (optimum)
optimum = ['hydrazinium', 'I', 'Sn'] # value = 1.5249 eV

# load in the perovskites dataset as a pandas DataFrame
lookup_df = pickle.load(open('./perovskites.pkl', 'rb'))

# helper functions
def measure(param):
    ''' lookup the HSEO6 bandgap for given perovskite composition
    '''
    match = lookup_df.loc[
                (lookup_df.organic == param['organic']) &
                (lookup_df.anion == param['anion']) &
                (lookup_df.cation == param['cation'])
        ]
    assert len(match)==1
    bandgap = match.loc[:, 'hse06'].to_numpy()[0]
    return bandgap

def get_descriptors(element, kind):
    ''' retrive the descriptors for a given categorical variable option
    '''
    return lookup_df.loc[(lookup_df[kind]==element)].loc[:, lookup_df.columns.str.startswith(f'{kind}-')].values[0].tolist()


# prepare descriptors
organic_options = lookup_df.organic.unique().tolist()
anion_options = lookup_df.anion.unique().tolist()
cation_options = lookup_df.cation.unique().tolist()

if with_desc:
    # use physicochemical descriptors - static or dynamic gryffin
    desc_organic = {option: get_descriptors(option, 'organic') for option in organic_options}
    desc_anion = {option: get_descriptors(option, 'anion') for option in anion_options}
    desc_cation = {option: get_descriptors(option, 'cation') for option in cation_options}
else:
    # no descriptors - naive gryffin
    desc_organic = {option: None for option in organic_options}
    desc_anion = {option: None for option in anion_options}
    desc_cation = {option: None for option in cation_options}

# gryffin config
config = {
    "general": {
        #"num_cpus": 'all',
        "num_cpus": 4,
        "auto_desc_gen": dynamic,
        "batches": 1,
        "sampling_strategies": 1,
        "boosted":  False,
        "caching": True,
        "random_seed": random_seed,
        "acquisition_optimizer": "genetic",
        # "verbosity": 3
    },
    "parameters": [
        {"name": "organic", "type": "categorical", 'options': organic_options, 'category_details': desc_organic},
        {"name": "anion", "type": "categorical", 'options': anion_options, 'category_details': desc_anion},
        {"name": "cation", "type": "categorical",  'options': cation_options, 'category_details': desc_cation},
    ],
    "objectives": [
        {"name": "bandgap", "goal": "min"},
    ]
}

observations = []

# initialize gryffin
gryffin =  Gryffin(config_dict=config)#, silent=True)

for num_iter in range(budget):
    print('-'*20, 'Iteration:', num_iter+1, '-'*20)

    # alternating sampling strategies, assuming batch size of 1
    idx = num_iter % len(sampling_strategies)
    sampling_strategy = sampling_strategies[idx]

    # ask Gryffin for a new sample
    samples = gryffin.recommend(observations=observations, sampling_strategies=[sampling_strategy])

    measurement = measure(samples[0])
    samples[0]['bandgap'] = measurement
    observations.extend(samples)
    print(f'SAMPLES : {samples}')
    print(f'MEASUREMENT : {measurement}')
#     print(f'ITER : {num_iter}\tSAMPLES : {samples}\t MEASUREMENT : {measurement}')


    # check for convergence
    if [samples[0]['organic'], samples[0]['anion'], samples[0]['cation']] == optimum:
        print(f'FOUND OPTIMUM AFTER {num_iter+1} ITERATIONS!')
        break
