"""
A lot of the following lines are directly ported from
AI for Healthcare Bootcamp 2019 Winter - 2019 Spring for EEG

Credits also to authors of CheXpert
"""

import argparse


def str_to_bool(arg):
    """Convert an argument string into its boolean value.

    Args:
        arg (string): String representing a bool.

    Returns:
        (bool) Boolean value for the string.
    """
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
