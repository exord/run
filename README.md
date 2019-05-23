# run

### New for emcee3 sampler.

In the config file, a new key in ```rundict``` called ```moves``` is used to define the parameters of the ```emcee.Moves.StretchMoves``` instances to be used. For the time being, only a mixture of this type of moves is easily available from the config file. In the future, new moves shall be implemented.

The format is a list of lists. Each sublists contains two elements: the move scale (a in the [Goodman & Weare](https://projecteuclid.org/euclid.camcos/1513731992) sense, eq. 9), and the weight of this move.

