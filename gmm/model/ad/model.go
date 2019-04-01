package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"bitbucket.org/dtolpin/infergo/mathx"
	"math"
)

type Model struct {
	Data  []float64
	NComp int
	Alpha float64
	Tau   float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	var mu []float64

	mu = make([]float64, m.NComp)
	var sigma []float64

	sigma = make([]float64, m.NComp)
	var ix int

	ix = 0
	for j := range mu {
		ad.Assignment(&mu[j], &x[ix])
		ix = ix + 1
		ad.Assignment(&sigma[j], ad.Elemental(math.Exp, &x[ix]))
		ix = ix + 1
	}
	var dir Dirichlet

	dir = Dirichlet{N: m.NComp}
	var alpha []float64

	alpha = make([]float64, dir.N)
	for j := range alpha {
		ad.Assignment(&alpha[j], &m.Alpha)
	}

	if m.Tau > 0 {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
			Normal.Logps(0, 0, x[ix:]...)
		}, 2, ad.Value(0.), ad.Arithmetic(ad.OpDiv, ad.Value(1), &m.Tau))))
	}
	var p [][]float64

	p = make([][]float64, len(m.Data))
	var theta [][]float64

	theta = make([][]float64, len(m.Data))
	for i := range m.Data {
		p[i] = make([]float64, m.NComp)
		theta[i] = make([]float64, m.NComp)

		theta[i] = x[ix : ix+m.NComp]
		ad.Call(func(_ []float64) {
			dir.SoftMax(x[ix:ix+m.NComp], p[i])
		}, 0)
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
			dir.Logp(alpha, p[i])
		}, 0)))
		ix = ix + m.NComp
	}

	for i := range m.Data {
		var l float64
		for j := 0; j != m.NComp; j = j + 1 {
			var lj float64
			ad.Assignment(&lj, ad.Arithmetic(ad.OpAdd, ad.Call(func(_ []float64) {
				Normal.Logp(0, 0, 0)
			}, 3, &mu[j], &sigma[j], &m.Data[i]), &theta[i][j]))

			if j == 0 {
				ad.Assignment(&l, &lj)
			} else {
				ad.Assignment(&l, ad.Elemental(mathx.LogSumExp, &l, &lj))
			}
		}
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, &l))
	}
	return ad.Return(&ll)
}
