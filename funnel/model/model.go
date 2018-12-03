// Neal's funnel
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

// Naive model
type NaiveModel struct{}

func (m NaiveModel) Observe(parameters []float64) float64 {
	ll := 0.0

	y := parameters[0]
	x := parameters[1:]

	ll += Normal.Logp(0, 3, y)
	ll += Normal.Logps(0, math.Exp(0.5*y), x...)

	return ll
}

// Reparameterized model
type ReparModel struct{}

func (m ReparModel) Observe(parameters []float64) float64 {
	ll := 0.0

	y := parameters[0]
	x := parameters[1:]

	ll += Normal.Logp(0, 1, y)
	ll += Normal.Logps(0, 1, x...)

	return ll
}
