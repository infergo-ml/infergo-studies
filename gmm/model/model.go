// Gaussian mixture
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
)

// data are the observations
type Model struct {
	Data  []float64 // samples
	NComp int       // number of components
	Alpha float64   // Dirichlet diffusion
	Tau   float64   // precision of prior on odds
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.0

	// Fetch component parameters
	mu := make([]float64, m.NComp)
	sigma := make([]float64, m.NComp)
	ix := 0
	for j := range mu {
		mu[j] = x[ix]
		ix++
		sigma[j] = math.Exp(x[ix])
		ix++
	}

	// Create an instance of Dirichlet distribution
	// for inferring component labels.
	alpha := make([]float64, m.NComp)
	for j := range alpha {
		alpha[j] = m.Alpha
	}

	// Observe observation odds from the Normal as a prior.
	// Tau=0 means improper uniform prior.
	if m.Tau > 0 {
		ll += Normal.Logps(0., 1/m.Tau, x[ix:]...)
	}

	// Fetch observation probabilities.
	theta := make([][]float64, len(m.Data))
	for i := range m.Data {
		theta[i] = make([]float64, m.NComp)
		D.SoftMax(x[ix:ix+m.NComp], theta[i])
		// Observe them from the Dirichlet to adjust the
		// contrast.
		ll += Dir.Logp(alpha, theta[i])
		ix += m.NComp
	}

	// Compute log likelihood of the mixture given the data.
	for i := range m.Data {
		var l float64
		for j := 0; j != m.NComp; j++ {
			lj := Normal.Logp(mu[j], sigma[j], m.Data[i]) +
				math.Log(theta[i][j])
			if j == 0 {
				l = lj
			} else {
				l = mathx.LogSumExp(l, lj)
			}
		}
		ll += l
	}
	return ll
}
