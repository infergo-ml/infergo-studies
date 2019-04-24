# Stan LDA example

The Latent Dirichlet allocation is borrowed from Stan example
models
(ttps://mc-stan.org/docs/stan-users-guide/latent-dirichlet-allocation.html)
and adapted to Infergo. Build using `make` and run `./lda -help`
for the full list of options. The file `data.json` contains the
same example data as Stan's example-models repository.

## A sample run:

```
$ ./lda
MLE (after 78 iterations):
* Log-likelihood: -106.753 => -73.151
Document topics
 1: 0.88 0.12
 2: 0.09 0.91
 3: 0.92 0.08
 4: 0.08 0.92
 5: 0.87 0.13
Topic words
 1: 0.46 0.46 0.04 0.03
 2: 0.03 0.03 0.50 0.43
   Burning:    10
   Burning:    20
...
   Burning:  1000
Collecting:    10
Collecting:    20
...
Collecting:   100
Topic words
 1: 0.50 0.45 0.03 0.01
 2: 0.03 0.04 0.34 0.58
Collecting:   110
Collecting:   120
Collecting:   130
...
Collecting:  1000
Topic words
 1: 0.59 0.31 0.08 0.02
 2: 0.09 0.02 0.53 0.36
Document topics
 1: 0.65 0.35
 2: 0.08 0.92
 3: 0.92 0.08
 4: 0.06 0.94
 5: 0.75 0.25
* HMC:
	accepted: 1172
	rejected: 829
	rate: 0.5857
```
