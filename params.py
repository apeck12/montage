from collections import OrderedDict
import itertools, argparse, os, sys
import numpy as np


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Enumerate all combinations of search parameters.')
    parser.add_argument('-p','--pattern', help='Pattern type: spiral or snowflake', 
                        required=True, type=str)
    parser.add_argument('-s','--savepath', help='Path to which to save file', 
                        required=True, type=str)
    return vars(parser.parse_args())


def set_params(pattern):
    """
    Generate dictionary of grid search parameters for input pattern type.
    """
    params = OrderedDict()
    params['translation_max'] = [1.5,2,3]
    params['xscale'] = [True, False]
    params['start_beam_angle'] = [0,10,20,30,-10,-20,-30]
    params['rotation_step'] = [0,-10,-20,-30]

    if pattern == 'snowflake':
        params['alternate'] = [True, False]

    elif pattern == 'spiral':
        params['alternate'] = [True, False]
        params['n_revolutions'] = [0,2,3,4,5]

    else:
        print("Pattern type must be spiral or snowflake.")
        sys.exit()

    return params


def generate_combinations(params, pattern):
    """
    Enumerate all combinations of values in input dictionary; the value of each key 
    should be a list of parameter values to scan. The elements of each combination 
    will retain the ordering of the params dict. Output is an OrderedDict.
    """
    combinations = OrderedDict()
    for i,cb in enumerate(itertools.product(*params.values())):
        # add value for continuous parameter -- should be opposite of alternate
        if pattern == 'spiral':
            cb = list(cb)
            cont_value = not cb[4]
            cb.insert(5, cont_value)
            cb = tuple(cb)  

        combinations[i] = cb

    return combinations
    

if __name__ == '__main__':
    
    # generate all combinations of parameters
    args = parse_commandline()
    params = set_params(args['pattern'])
    combinations = generate_combinations(params, args['pattern'])

    # write to output file
    with open(args['savepath'], "w") as f:
        for i,v in combinations.items():
            line = str(i) + ' ' + ' '.join(map(str, [*v])) + '\n'
            f.write(line)
    f.close()
