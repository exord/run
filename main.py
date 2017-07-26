import os
import numpy as np
import importlib
import datetime
import pickle
# import imp
from .config import read_config

from mcmc_general import lnprob


def runmcmc(configfile, nsteps=None, **kwargs):

    initfromsampler = kwargs.pop('initsampler', None)
    uselaststep = kwargs.pop('uselaststep', False)

    # Read dictionaries from configuration file
    rundict, initdict, datadict, priordict, fixeddict = read_config(
        configfile)

    # Read lnlike and lnprior functions from specific target module.
    modulename = 'model_{target}_{runid}'.format(**rundict)
    mod = importlib.import_module(modulename)
    reload(mod)

    # If initfromsampler given, use it to create starting point for
    # chain. Overrides machinery in config module
    if initfromsampler is not None:
        isampler = pickle.load(open(initfromsampler))

        if uselaststep:
            # Pick last element from chain
            pn = isampler.chain[:, -1, :]
        else:
            # Pick nwalker random samples from chain
            ind = np.random.choice(np.arange(0, isampler.chain.shape[1]),
                                   size=rundict['nwalkers'], replace=False)
            indc = np.random.randint(isampler.chain.shape[0], 
                                     size=rundict['nwalkers']) 
            pn = isampler.chain[indc, ind, :]

        # Overwrite values from initdict for those parameters in common.
        for par in isampler.args[0]:
            if par in initdict:
                initdict[par] = pn[:, isampler.args[0].index(par)]
    
    lnprobargs = [initdict.keys(), mod.lnlike, mod.lnprior]
    lnprobkwargs = {'lnlikeargs': [fixeddict, datadict],
                    'lnpriorargs': [priordict, ],
                    'lnlikekwargs': {}}

    if rundict['sampler'] == 'emcee':
        import emcee
        a = rundict.pop('a', 2.0)
        sampler = emcee.EnsembleSampler(rundict['nwalkers'], len(priordict),
                                        lnprob, args=lnprobargs,
                                        kwargs=lnprobkwargs, a=a)
    elif rundict['sampler'] == 'cobmcmc':
        import cobmcmc
        sampler = cobmcmc.ChangeofBasisSampler(len(priordict), lnprob,
                                               lnprobargs, lnprobkwargs,
                                               startpca=rundict['startpca'],
                                               npca=rundict['npca'],
                                               nupdatepca=rundict['nupdatepca'])

    else:
        raise NameError('Unknown sampler: {}'.format(rundict['sampler']))

    # Number of steps.
    if nsteps is None and 'nsteps' not in rundict:
        raise TypeError('Number of steps must be given in configuration '
                        'file or as argument to runmcmc function.')
    elif nsteps is None:
        nsteps = rundict['nsteps']
            
    # Starting point.
    p0 = np.array(initdict.values()).T

    print('Doing {} steps of {} MCMC sampler, '
          'using {} walkers in {}-dimensional parameter space'
          .format(nsteps, rundict['sampler'], p0.shape[0], p0.shape[1]))

    # ## MAIN MCMC RUN ##
    sampler.run_mcmc(p0, nsteps)

    sampler.runid = rundict['runid']
    sampler.target = rundict['target']
    sampler.comment = rundict.get('comment', '')
    sampler.part = 1

    if sampler.comment != '':
        sampler.comment = '_'+sampler.comment

    # Pickle sampler to file
    dump2pickle(sampler, rundict.get('sampler', None))
    return sampler


def continuemcmc(samplerfile, nsteps, newsampler=False):
    f = open(samplerfile)
    sampler = pickle.load(f)
    f.close()

    import emcee
    import cobmcmc
    if isinstance(sampler, emcee.EnsembleSampler):
        # Produce starting point from last point in chain.
        p0 = sampler.chain[:, -1, :]

        if newsampler:
            sampler.reset()
            try:
                sampler.part += 1
            except AttributeError:
                sampler.part = 2

        sampler.run_mcmc(p0, nsteps)
        sampleralgo = 'emcee'
    elif isinstance(sampler, cobmcmc.ChangeofBasisSampler):        
        sampler.run_mcmc(nsteps)
        sampleralgo = 'cobmcmc'
    
    # Pickle sampler to file
    dump2pickle(sampler, sampleralgo)
    return sampler


def dump2pickle(sampler, sampleralgo='emcee'):
    
    if sampleralgo is None:
        sampleralgo = ''

    pickledict = {'target': sampler.target,
                  'runid': sampler.runid,
                  'comm': sampler.comment,
                  'nwalk': sampler.k,
                  'nstep': sampler.iterations,
                  'sampler': sampleralgo,
                  'date': datetime.datetime.today().isoformat()}

    pickledir = os.path.join(os.getenv('HOME'), 'ExP',
                             pickledict['target'], 'samplers')

    # Check if path exists; create if not
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)
    
    f = open(os.path.join(pickledir,
                          '{target}_{runid}{comm}_{nwalk}walkers_'
                          '{nstep}steps_{sampler}_{date}.dat'.format(
                              **pickledict)), 'wb')
    pickle.dump(sampler, f)
    f.close()
    return
