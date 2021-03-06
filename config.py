import imp
import numpy as np
import pandas as pd

from MCMC import priors

def read_config(configfile):
    """
    Initialises a sampler object using parameters in config.

    :param string config: string indicating path to configuration file.
    """

    # Import configuration file as module
    c = imp.load_source('c', configfile)

    # Make copy of all relavant dictionaries
    rundict, input_dict, datadict = map(dict.copy, c.configdicts)

    # Create prior instances
    priordict = priors.prior_constructor(input_dict, {})
    
    # Build list of parameter names
    # parnames, fixparnames = get_parnames(input_dict)
    
    # Read data from file(s)
    read_data(c.datadict)

    # Initial values
    initial_values = draw_initial_values(input_dict, priordict,
                                         rundict['nwalkers'])

    # Fixed parameters
    fixedpardict = get_fixedparvalues(input_dict)

    """
    # Initialise sampler
    if rundict['sampler'] == 'emcee':
        import emcee
        sam = emcee.EnsembleSampler(rundict['nwalkers'], len(priordict),
                                    lambda x: 1, args=[])
    elif rundict['sampler'] == 'cobmcmc':
        import cobmcmc
        sam = cobmcmc.ChangeofBasisSampler(len(priordict), lambda x: 1,
                                           [], {},
                                           startpca=rundict['startpca'],
                                           npca=rundict['npca'],
                                           nupdatepca=rundict['nupdatepca'])

    # Another hacky way to pass desired number of steps
    sam.nsteps = rundict.pop('nsteps', None)
    """    
    return rundict, initial_values, datadict, priordict, fixedpardict

def get_parnames(input_dict):
    parnames = []
    fixparnames = []
    for obj in input_dict:
        for par in input_dict[obj]:
            if input_dict[obj][par][1] > 0:
                parnames.append(obj+'_'+par)
            elif input_dict[obj][par][1] == 0:
                fixparnames.append(obj+'_'+par)
    return parnames, fixparnames

def get_fixedparvalues(input_dict):
    fpdict = {}
    for obj in input_dict:
        for par in input_dict[obj]:
            if input_dict[obj][par][1] == 0:
                fpdict[obj+'_'+par] = input_dict[obj][par][0]
    return fpdict

def draw_initial_values(input_dict, priordict, nwalkers=1):
    """
    Prepare initial values for MCMC algorithm with different mechanisms.
    If flag is 1, draw randomly from prior.
    If flag is 2, start at a given point, but add some "noise" related to
    the size of the prior.
    """
    # Create dictionary of initial values
    initial_values = dict.fromkeys(priordict)

    for fullpar in initial_values:
        obj, par = fullpar.split('_')
        if input_dict[obj][par][1] == 1:
            p0 = priordict[obj+'_'+par].rvs(size=nwalkers)
        elif input_dict[obj][par][1] == 2:
            # For emcee cannot start all walkers exactly at the same place
            # or that parameter will never evolve. Add "noise".
            parlist = input_dict[obj][par]
            
            try:
                scale = parlist[3]
            except IndexError:
                if parlist[0] != 0:
                    scale = parlist[0] * 0.05
                elif parlist[2][0] == 'Uniform':
                    scale = (parlist[2][2] - parlist[2][1]) * 0.05
                elif parlist[2][0] == 'Normal':
                    scale = parlist[2][1] * 0.05
                else:
                    print(parlist)
            p0  = (np.full(nwalkers, parlist[0]) +
                   np.random.randn(nwalkers) * scale)
            del(scale)
        else:
            continue
            # raise ValueError('Ilegal flag for parameter.')
        initial_values[fullpar] = p0
    return initial_values
    
def read_data(datadict):
    for inst in datadict:
        # Try to get custom separator
        try:
            sep = datadict[inst]['sep']
        except KeyError:
            sep = '\t' 

        # Read rdb file
        data = pd.read_csv(datadict[inst]['datafile'], sep=sep,
                           comment='#', skiprows=[1,])
        datadict[inst]['data'] = data
    return
    