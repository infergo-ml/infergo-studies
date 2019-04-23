// LDA
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/model"
	"math"
)

// data are the observations
type Model struct {
	K int // num topics
	V int // num words
	M int // num docs
	N int // total word instances
	Word []int // word n
	Doc []int // doc ID for word n
	Alpha []float64 // topic prior
	Beta []float64 // word prior
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.0

	dirt := Dirichlet{m.K}
	dirw := Dirichlet{m.V}

	// Fetch component parameters
	theta := make([][]float64, m.M)
	phi := make([][]float64, m.K)
	for im := 0; im != m.M; im++ {
		theta[im] = model.Shift(&x, m.K)
	}
	for ik := 0; ik != m.M; ik++ {
		theta[ik] = model.Shift(&x, m.V)
	}

	// Impose priors
	for im := 0; im != m.M; im++ {
		ll += dirt.Logp(m.Alpha, theta[im])
	}
	for ik := 0; ik != m.M; ik++ {
		ll += dirw.Logp(m.Beta, theta[ik])
	}

	// Condition on observations
	gamma := make([]float64, m.K)
	for in := 0; in != m.N; in++ {
		for ik := 0; ik != m.K; ik++ {
			gamma[ik] = math.Log(theta[m.Doc[in]][ik]) +
				math.Log(phi[ik][m.Word[in]])
		}
		ll += D.LogSumExp(gamma)
	}

	return ll
}
