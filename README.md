# Examples and experiments with infergo

[`infergo`](http://infergo.org/) is a probabilistic programming facility for
the Go language.

* [funnel](funnel) — Neal's funnel, a model that demonstrates
  influence of reparameterization.
* [gmm](gmm) — Gaussian Mixture Model with inference of
  probabilities of each data point to belong to each of the
  components.
* [wasm](wasm) — the [probabilistic 'hello
  world'](http://bitbucket.org/dtolpin/infergo/src/master/examples/hello)
  example running in a web browser or
  [Wasm](http://webassembly.org/) engine.
* [lr](lr) — a probabilistic model of linear regression. 
* [lr-gonum](lr-gonum) — [Gonum](http://gonum.org/) integration.
  The same linear regression as in
  [lr](lr), however the model is re-formulated as a Gonum
  [minimization problem](https://godoc.org/gonum.org/v1/gonum/optimize#Problem) and solved using a Gonum optimization
  algorithm. Gonum provides a number of optimization algorithms,  BFGS and L-BFGS among them.
