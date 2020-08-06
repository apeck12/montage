from collections import OrderedDict
import itertools, argparse, os, sys
import numpy as np


def parse_commandline():
    """
    Parse commandline input.
    """
    parser = argparse.ArgumentParser(description='Enumerate all combinations of search parameters.')
    parser.add_argument('-p','--pattern', help='Pattern type: spiral, snowflake, or sunflower', 
                        required=True, type=str)
    parser.add_argument('-s','--savepath', help='Path to which to save file', 
                        required=True, type=str)
    return vars(parser.parse_args())


def set_params(pattern):
    """
    Generate dictionary of grid search parameters for input pattern type.
    """    
    params = OrderedDict()

    if pattern == 'sunflower':
        params['translation_max'] = np.arange(0,2.2,0.2)
    else:
        params['translation_max'] = [1,1.5,2]

    params['xscale'] = [True, False]
    params['start_beam_angle'] = [0,10,20,30,-10,-20,-30]
    params['rotation_step'] = [0,-10,-20,-30]
    params['alternate'] = [True, False]

    if pattern == 'snowflake':
        params['n_steps'] = [0,1,2,3,4]

    if pattern == 'spiral':
        params['n_revolutions'] = [0,2,3,4,5]
        
    params['shift_sigma'] = [0]
        
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
        if (pattern == 'spiral') or (pattern == 'sunflower'):
            cb = list(cb)
            cont_value = not cb[4]
            cb.insert(5, cont_value)
            cb = tuple(cb)  

        combinations[i] = cb

    return combinations
    

if __name__ == '__main__':
    
    # generate all combinations of parameters
    args = parse_commandline()
    if args['pattern'] not in ['spiral','snowflake','sunflower']:
        print("Pattern must be spiral, snowflake, or sunflower")
        sys.exit()

    params = set_params(args['pattern'])
    combinations = generate_combinations(params, args['pattern'])

    # write to output file
    with open(args['savepath'], "w") as f:
        for i,v in combinations.items():
            line = str(i) + ' ' + ' '.join(map(str, [*v])) + '\n'
            f.write(line)
    f.close()
