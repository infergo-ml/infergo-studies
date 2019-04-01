package model

import (
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/dist/ad"
	"math"
)

type NaiveModel struct{}

func (m NaiveModel) Observe(parameters []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(parameters)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	var y float64
	ad.Assignment(&y, &parameters[0])
	var x []float64

	x = parameters[1:]
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
		Normal.Logp(0, 0, 0)
	}, 3, ad.Value(0), ad.Value(3), &y)))
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
		Normal.Logps(0, 0, x...)
	}, 2, ad.Value(0), ad.Elemental(math.Exp, ad.Arithmetic(ad.OpMul, ad.Value(0.5), &y)))))

	return ad.Return(&ll)
}

type ReparModel struct{}

func (m ReparModel) Observe(parameters []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(parameters)
	}
	var ll float64
	ad.Assignment(&ll, ad.Value(0.0))
	var y float64
	ad.Assignment(&y, &parameters[0])
	var x []float64

	x = parameters[1:]
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
		Normal.Logp(0, 0, 0)
	}, 3, ad.Value(0), ad.Value(1), &y)))
	ad.Assignment(&ll, ad.Arithmetic(ad.OpAdd, &ll, ad.Call(func(_ []float64) {
		Normal.Logps(0, 0, x...)
	}, 2, ad.Value(0), ad.Value(1))))

	return ad.Return(&ll)
}
