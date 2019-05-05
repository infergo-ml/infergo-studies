// LDA
package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/model"
	"math"
)

// data are the observations
type Model struct {
	K     int       // num topics
	V     int       // num words
	M     int       // num docs
	N     int       // total word instances
	Word  []int     // word n
	Doc   []int     // doc ID for word n
	Alpha []float64 // topic prior
	Beta  []float64 // word prior
}

func (m *Model) Observe(x []float64) float64 {
	ll := 0.0

	// Regularize the parameter vector
	Normal.Logps(0, 1, x...)

	// Destructure parameters
	theta := make([][]float64, m.M)
	m.FetchSimplices(&x, m.K, theta)
	phi := make([][]float64, m.K)
	m.FetchSimplices(&x, m.V, phi)

	// Impose priors
	ll += Dir.Logps(m.Alpha, theta...)
	ll += Dir.Logps(m.Beta, phi...)

	// Conditioning on observations
	gamma := make([]float64, m.K)
	for in := 0; in != m.N; in++ {
		for ik := 0; ik != m.K; ik++ {
			gamma[ik] = math.Log(theta[m.Doc[in]-1][ik]) +
				math.Log(phi[ik][m.Word[in]-1])
		}
		ll += D.LogSumExp(gamma)
	}

	return ll
}

// FetchmSimplices fetches simplices from the parameter slice.
// The parameter slice is advanced in place, hence a shallow
// copy of the slice must be passed.
func (m *Model) FetchSimplices(
	px *[]float64,
	k int,
	simplices [][]float64,
) {
	for i := range simplices {
		simplices[i] = make([]float64, k)
		D.SoftMax(model.Shift(px, k), simplices[i])
	}
}
