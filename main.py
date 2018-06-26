import os
import sys
import numpy as np
import importlib
import datetime
import pickle
import time

import emcee
import cobmcmc

try:
    import PyPolyChord as polychord
    import PyPolyChord.settings as polysettings
except ModuleNotFoundError:
    pass

from mcmc_general import lnprob
from emcee.ptsampler import default_beta_ladder

from .config import read_config

HOME = os.getenv('HOME')

def runmcmc(configfile, nsteps=None, modelargs={}, **kwargs):

    initfromsampler = kwargs.pop('initsampler', None)
    uselaststep = kwargs.pop('uselaststep', False)
    counttime = kwargs.pop('time', True)

    # Read dictionaries from configuration file
    rundict, initdict, datadict, priordict, fixeddict = read_config(
        configfile)

    # Read lnlike and lnprior functions from specific target module.
    modulename = 'model_{target}_{runid}'.format(**rundict)
    mod = importlib.import_module(modulename)

    if sys.version_info[0] == 2:
        reload(mod)
    elif sys.version_info[0] == 3:
        importlib.reload(mod)

    # Instantaniate model class (pass additional arguments)
    mymodel = mod.Model(fixeddict, datadict, priordict, **modelargs)
    
# =============================================================================
#     # Preprocess data if a preprocess function exists in module
#     # Return new arguments for lnpostfn
#     if 'preprocess' in dir(mod) and hasattr(mod.preprocess, '__call__'):
#         newargs = mod.preprocess(rundict, initdict, datadict, priordict,
#                                  fixeddict)
# 
#     else:
#         newargs = []
# =============================================================================

    # If initfromsampler given, use it to create starting point for
    # chain. Overrides machinery in config module
    if initfromsampler is not None:
        isampler = pickle.load(open(initfromsampler))

        if isinstance(isampler, (emcee.Sampler, cobmcmc.Sampler)):
            initchain = isampler.chain
            ipars = isampler.args[0]
        elif isinstance(isampler[0], np.ndarray):
            initchain = isampler[0]
            ipars = isampler[-2][0]
            
        if uselaststep:
            
            if rundict['nwalkers'] > initchain.shape[0]:
                raise ValueError('Cannot use last step. Init sampler has less walkers than current sampler.')
            elif rundict['nwalkers'] == initchain.shape[0]:
                # Pick last element from chain
                pn = initchain[:, -1, :]
            else:
                # Pick last element from chain for a random subset of walkers
                ind = np.random.choice(np.arange(0, initchain.shape[0]),
                                       size=rundict['nwalkers'], replace=False)
                pn = initchain[ind, -1, :]
                
        else:
            # Pick nwalker random samples from chain
            ind = np.random.choice(np.arange(0, initchain.shape[1]),
                                   size=rundict['nwalkers'], replace=True)
            indc = np.random.randint(initchain.shape[0], 
                                     size=rundict['nwalkers']) 
            pn = initchain[indc, ind, :]

        # Overwrite values from initdict for those parameters in common.
        for par in ipars:
            if par in initdict:
                initdict[par] = pn[:, ipars.index(par)]

# =============================================================================
#     # Prepare list of arguments for lnlike and lnprior function
#     lnlikeargs = [fixeddict, datadict]
#     lnpriorargs = [priordict,]
# 
#     # Add new arguments from preprocessing
#     for arglist in [lnlikeargs, lnpriorargs]:
#         arglist.extend(newargs)
# 
#     lnprobargs = [list(initdict.keys()), mod.lnlike, mod.lnprior]
#     lnprobkwargs = {'lnlikeargs': lnlikeargs,
#                     'lnpriorargs': lnpriorargs,
#                     'lnlikekwargs': {}}
# 
# =============================================================================
    
    if rundict['sampler'] == 'emcee':
        a = rundict.pop('a', 2.0)
        ncpus = kwargs.pop('threads', rundict.pop('threads', 1))
        print('Running with {} cores'.format(ncpus))
        sampler = emcee.EnsembleSampler(rundict['nwalkers'], len(priordict),
                                        mymodel.logpdf, a=a, threads=ncpus)
        sampler.model = mymodel

    elif rundict['sampler'] == 'PTSampler':
        a = rundict.pop('a', 2.0)
        ntemps = rundict.pop('ntemps', None)
        tmax = rundict.pop('Tmax', None)
        
# =============================================================================
#         # Construct argument lists
#         # for log likelihood
#         loglargs = [list(initdict.keys()), ]
#         loglargs.extend(lnprobkwargs['lnlikeargs'])
#         # and for logprior
#         logpargs = [list(initdict.keys()), ]
#         logpargs.extend(lnprobkwargs['lnpriorargs'])
#         
# =============================================================================
        sampler = emcee.PTSampler(ntemps, rundict['nwalkers'], len(priordict),
                                  logl=mymodel.lnlike, logp=mymodel.lnprior,
                                  Tmax=tmax)
        
    elif rundict['sampler'] == 'cobmcmc':
        sampler = cobmcmc.ChangeofBasisSampler(len(priordict), mymodel.logpdf,
                                               [], {},
                                               startpca=rundict['startpca'],
                                               npca=rundict['npca'],
                                               nupdatepca=rundict['nupdatepca']
                                               )

    else:
        raise NameError('Unknown sampler: {}'.format(rundict['sampler']))

    # Number of steps.
    if nsteps is None and 'nsteps' not in rundict:
        raise ValueError('Number of steps must be given in configuration '
                         'file or as argument to runmcmc function.')
    elif nsteps is None:
        nsteps = rundict['nsteps']

    # Starting point.
    p0 = np.array(list(initdict.values())).T

    print('Doing {} steps of {} MCMC sampler, '
          'using {} walkers in {}-dimensional parameter space'
          .format(nsteps, rundict['sampler'], p0.shape[0], p0.shape[1]))

    # Special treatment for PTSampler
    if rundict['sampler'] == 'PTSampler':
        p0 = np.repeat(p0[np.newaxis, :, :], len(sampler.betas), axis=0)
        sampler.k = sampler.nwalkers
        sampler.iterations = nsteps

    if counttime:
        ti = time.clock()
        tw = time.time()
    # ## MAIN MCMC RUN ##
    sampler.run_mcmc(p0, nsteps)

    # Add times to sampler
    sampler.runtime = time.clock() - ti
    sampler.walltime = time.time() - tw
    
    sampler.runid = rundict['runid']
    sampler.target = rundict['target']
    sampler.comment = rundict.get('comment', '')
    sampler.part = 1

    if sampler.comment != '':
        sampler.comment = '_'+sampler.comment

    # Pickle sampler to file
    dump2pickle(sampler, rundict.get('sampler', None), multi=ncpus)
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
    dump2pickle(sampler, sampleralgo, multi=sampler.threads)
    return sampler


def dump2pickle(sampler, sampleralgo='emcee', multi=1, savedir=None):

    if sampleralgo is None:
        sampleralgo = ''

    if isinstance(sampler, emcee.PTSampler):
        nwalk = sampler.nwalkers
    elif isinstance(sampler, emcee.EnsembleSampler):
        nwalk = sampler.k
    elif isinstance(sampler, cobmcmc.ChangeofBasisSampler):
        nwalk = 1
        
    pickledict = {'target': sampler.target,
                  'runid': sampler.runid,
                  'comm': sampler.comment,
                  'nwalk': nwalk,
                  'nstep': sampler.iterations,
                  'sampler': sampleralgo,
                  'date': datetime.datetime.today().isoformat()}

    if savedir is None:
        pickledir = os.path.join(os.getenv('HOME'), 'ExP',
                                pickledict['target'], 'samplers')
    else:
        pickledir = savedir

    # Check if path exists; create if not
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)

    f = open(os.path.join(pickledir,
                          '{target}_{runid}{comm}_{nwalk}walkers_'
                          '{nstep}steps_{sampler}_{date}.dat'.format(
                              **pickledict)), 'wb')

    if multi>1:
        pickle.dump([sampler.chain, sampler.lnprobability,
                     sampler.acceptance_fraction, 
                     list(sampler.model.priordict.keys()), sampler.model], f)
    else:
        pickle.dump(sampler, f)
    f.close()
    return


def runpoly(configfile, nlive=None, modelargs={}, **kwargs):
    
    # Read dictionaries from configuration file
    rundict, initdict, datadict, priordict, fixeddict = read_config(
        configfile)
    parnames = list(priordict.keys())

    # Import model module
    modulename = 'model_{target}_{runid}'.format(**rundict)
    mod = importlib.import_module(modulename)

    # Instantaniate model class (pass additional arguments)
    mymodel = mod.Model(fixeddict, datadict, priordict, **modelargs)

    # Function to convert from hypercube to physical parameter space
    def prior(hypercube):
        """ Priors for each parameter. """
        theta = []
        for i, x in enumerate(hypercube):
            
            theta.append(priordict[parnames[i]].ppf(x))
        return theta

    def loglike(x):
        return (mymodel.lnlike(x), [])

    # Prepare run
    nderived = 0
    ndim = len(initdict)

    # Fix starting time to identify chain.
    isodate = datetime.datetime.today().isoformat()
    
    # Define PolyChord settings
    settings = polysettings.PolyChordSettings(ndim, nderived, )
    settings.do_clustering = True
    if nlive is None:
        settings.nlive = 20*ndim
    else:
        settings.nlive = nlive
        
    fileroot = rundict['target']+'_'+rundict['runid']
    if rundict['comment'] != '':
        fileroot += '_'+rundict['comment']
        
    # add date
    fileroot += isodate
    
    settings.file_root = fileroot
    settings.read_resume = False
    settings.num_repeats = ndim * 3
    settings.feedback = 1
    settings.precision_criterion = 0.01
    # base directory
    base_dir = os.path.join(HOME, 'ExP', rundict['target'], 'polychains')
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    settings.base_dir = os.path.join(base_dir)

    # Initialise clocks
    ti = time.clock()
    tw = time.time()
    
    # Run PolyChord
    output = polychord.run_polychord(loglike, ndim, nderived, settings, prior)

    output.runtime = time.clock() - ti
    output.walltime = time.time() - tw

    output.target = rundict['target']
    output.runid = rundict['runid']
    output.comment = rundict.get('comment', '')

    if output.comment != '':
        output.comment = '_'+output.comment
    
    print(f'\nTotal run time was: {datetime.timedelta(seconds=int(output.runtime))}')
    print(f'Total wall time was: {datetime.timedelta(seconds=int(output.walltime))}')
    print(f'\nlog10(Z) = {output.logZ*0.43429} \n') # Log10 of the evidence

    dump2pickle_poly(output)    
    
    return output

def dump2pickle_poly(output, savedir=None):

    pickledict = {'target': output.target,
                  'runid': output.runid,
                  'comm': output.comment,
                  'nlive': output.nlive, 
                  'sampler': 'polychord',
                  'date': datetime.datetime.today().isoformat()}

    if savedir is None:
        pickledir = os.path.join(os.getenv('HOME'), 'ExP',
                                output.target, 'samplers')
    else:
        pickledir = savedir

    # Check if path exists; create if not
    if not os.path.isdir(pickledir):
        os.makedirs(pickledir)

    f = open(os.path.join(pickledir,
                          '{target}_{runid}{comm}_{nlive}live_'
                          '{sampler}_{date}.dat'.format(**pickledict)), 'wb')
    
    pickle.dump(output, f)
    f.close()
    return
    
