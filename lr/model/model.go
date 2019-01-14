// Gaussian mixture
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

// data are the observations
type Model struct {
	Data  [][]float64 // samples
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.

	alpha := x[0]
	beta := x[1]
	sigma := math.Exp(x[2])

	ll += Normal.Logp(0, 1, alpha)
	ll += Normal.Logp(0, 1, beta)

	for i := range m.Data {
		ll += Normal.Logp(
			m.Simulate(m.Data[i][0], alpha, beta),
			sigma, m.Data[i][1])
	}

	return ll
}

func (m *Model) Simulate(x, alpha, beta float64) (y float64) {
	y = alpha +beta*x
	return y
}
