#!/usr/bin/env python 
from gryffin import Gryffin


def test_import():
    from gryffin import Gryffin 


def test_recommend():
    observations = [
        {'param_0': [-1.0, -1.0], 'param_1': [1.0], 'obj_0': 0.1, 'obj_1': 0.2},
        {'param_0': [1.0, 1.0], 'param_1': [-1.0], 'obj_0': 0.2, 'obj_1': 0.1},
    ]

    gryffin = Gryffin()
    samples = gryffin.recommend(observations=observations)
