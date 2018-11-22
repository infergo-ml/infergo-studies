# Gaussian mixture model

Inferring gaussian mixture components; both component parameters
and probabilities of data points to belong to each component are
inferred. First, a MLE estimate of the parameters is computed,
then, a Hamiltonian Monte Carlo variant is used to infer the
posterior, and empirical means of the full posterior are shown.

The default prior on component membership is improper uniform,
and HMC or NUTS are unlikely to converge with these settings.
Change to '-alpha 0.1 -tau 1' to see the inference converging.

## Installation

* [Install Go](https://golang.org/doc/install), version 1.11 or
newer is required because the case study uses modules. 

* Clone the repository.

```
git clone https://git@bitbucket.org/dtolpin/infergo-studies
```

* Change the current directory to `infergo-studies/gmm`.

## Running the inference

* Run `make`. Make will build the executable and run the
  inference on a small embedded data set, for self-check.
* Run `./gmm -niter 1000 -ncomp data-3.csv` for a bigger data
  set.
