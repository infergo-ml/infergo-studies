package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

type Model struct {
	Data [][]float64
}

func (m *Model) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.))
	var alpha float64
	ad.Assignment(&alpha, &x[0])
	var beta float64
	ad.Assignment(&beta, &x[1])
	var sigma float64
	ad.Assignment(&sigma, ad.Elemental(math.Exp, &x[2]))

	for i := range m.Data {
		ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
			Normal.Logp(0, 0, 0)
		}, 3, ad.Call(func(_ []float64) {
			m.Simulate(0, 0, 0)
		}, 3, &m.Data[i][0], &alpha, &beta), &sigma, &m.Data[i][1])))
	}

	return ad.Return(&ll)
}

func (m *Model) Simulate(x, alpha, beta float64) (y float64) {
	if ad.Called() {
		ad.Enter(&x, &alpha, &beta)
	} else {
		panic("Simulate called outside Observe.")
	}
	ad.Assignment(&y, ad.Arithmetic(ad.OpAdd, &alpha, ad.Arithmetic(ad.OpMul, &beta, &x)))
	return ad.Return(&y)
}
