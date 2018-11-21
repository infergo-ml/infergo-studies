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

	dir := Dirichlet{N: m.NComp}
	alpha := make([]float64, dir.N)
	for j := range alpha {
		alpha[j] = m.Alpha
	}

	// Fetch observation probabilities
	p := make([][]float64, len(m.Data))
	for i := range m.Data {
		p[i] = make([]float64, m.NComp)

		dir.SoftMax(x[ix:ix+m.NComp], p[i])

		ix += m.NComp
	}

	// Compute log likelihood of mixture
	// given the data
	for i := range m.Data {
		var l float64
		for j := 0; j != m.NComp; j++ {
			lj := Normal.Logp(mu[j], sigma[j], m.Data[i]) +
				p[i][j]
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
