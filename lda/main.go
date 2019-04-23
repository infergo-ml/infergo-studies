package main // import "bitbucket.org/dtolpin/infergo-studies/lda"

import (
	. "bitbucket.org/dtolpin/infergo-studies/lda/model/ad"
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"
)

// Command line arguments

var (
	NCOMP = 2

	// Improper uniform prior on component membership by default.
	ALPHA = 1.
	TAU   = 0.

	// Inference algorithm parameters
	MCMC     = "HMC"
	RATE     = 0.1
	NITER    = 1000
	NBURN    = 0
	NADPT    = 10
	EPS      = 1E-4
	STEP     = 0.1
	DEPTH    = 5.
	MAXDEPTH = 0
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.Usage = func() {
		fmt.Printf(`Gaussian mixture model: lda [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.IntVar(&NCOMP, "ncomp", NCOMP, "number of components")
	flag.Float64Var(&ALPHA, "alpha", ALPHA, "Dirichlet diffusion")
	flag.Float64Var(&TAU, "tau", TAU, "precision of on odds")
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.StringVar(&MCMC, "mcmc", MCMC, "MCMC algorithm")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.IntVar(&NBURN, "nburn", NBURN, "number of burned iterations")
	flag.IntVar(&NADPT, "nadpt", NADPT, "number of steps per adaptation")
	flag.Float64Var(&EPS, "eps", EPS, "optimization precision")
	flag.Float64Var(&STEP, "step", STEP, "HMC step")
	flag.Float64Var(&DEPTH, "depth", DEPTH, "HMC or target NUTS depth")
	flag.IntVar(&MAXDEPTH, "maxdepth", MAXDEPTH, "maximum NUTS depth")
}

func main() {
	flag.Parse()
	if NBURN == 0 {
		NBURN = NITER
	}

	if flag.NArg() > 1 {
		fmt.Fprintf(os.Stderr,
			"unexpected positional arguments: %v\n",
			flag.Args()[1:])
		os.Exit(1)
	}

	// Get the data
	m := Model{}
	var (
		data   []float64
		labels []int
	)

	// Initialize the parameters
	m := &Model{
		Data:  data,
		NComp: NCOMP,
		Alpha: ALPHA,
		Tau:   TAU,
	}
	x := make([]float64, 2*m.NComp+len(m.Data)*m.NComp)

	// Set a starting  point
	if m.NComp == 1 {
		x[0] = 0.
		x[1] = 1. // stddev = exp(1)
	} else {
		// Spread the initial components wide and thin
		for j := 0; j != m.NComp; j++ {
			x[2*j] = -1. + 2./float64(m.NComp-1)*float64(j)
			x[2*j+1] = 1. // stddev = exp(1)
		}
	}

	// Run the optimizer
	opt := &infer.Adam{Rate: RATE}
	ll0, _ := opt.Step(m, x)
	ll, llprev := ll0, ll0
	iter := 0
	for ; iter != NITER; iter++ {
		ll, _ = opt.Step(m, x)
		if math.Abs(ll-llprev)/math.Abs(ll+llprev) < EPS {
			break
		}
		llprev = ll
	}

	// Target for SoftMax
	p := make([]float64, NCOMP)

	// Print the results.
	fmt.Printf("MLE (after %d iterations):\n", iter)
	fmt.Printf("* Log-likelihood: %7.3f => %7.3f\n", ll0, ll)
	fmt.Printf("* Components:\n")
	ix := 0
	for j := 0; j != m.NComp; j++ {
		fmt.Printf("\t%d: mean=%.3f, stddev=%.3f\n",
			j, x[2*j], math.Exp(x[2*j+1]))
		ix += 2
	}

	fmt.Printf("* Observations:\n")
	// Header
	fmt.Print("    #\t  value\t label")
	for j := 0; j != NCOMP; j++ {
		fmt.Printf("\t   p%d", j)
	}
	fmt.Println()
	// Values
	for i := range data {
		fmt.Printf("%5d\t%7.3f\t%4d", i, data[i], labels[i])
		D.SoftMax(x[ix:ix+m.NComp], p)
		ix += m.NComp
		for j := 0; j != m.NComp; j++ {
			fmt.Printf("\t%7.3f", p[j])
		}
		fmt.Println()
	}

	// Now let's infer the posterior with HMC.
	var mcmc infer.MCMC
	switch strings.ToUpper(MCMC) {
	case "HMC":
		mcmc = &infer.HMC{
			L:   int(math.Round(DEPTH)),
			Eps: STEP,
		}
	case "NUTS":
		mcmc = &infer.NUTS{
			Eps:      STEP,
			MaxDepth: MAXDEPTH,
		}
	default:
		fmt.Fprintf(os.Stderr, "invalid MCMC: %v\n", MCMC)
		os.Exit(1)
	}

	samples := make(chan []float64)
	mcmc.Sample(m, x, samples)

	// Print progress for the impatient
	progress := func(stage string, i int) {
		if (i+1)%10 == 0 {
			fmt.Fprintf(os.Stderr, "%10s: %5d\r", stage, i+1)
		}
	}

	switch mcmc := mcmc.(type) {
	case *infer.NUTS:
		// Adapt toward optimum tree depth.
		da := &infer.DepthAdapter{
			DualAveraging: infer.DualAveraging{Rate: RATE},
			Depth:         DEPTH,
			NAdpt:         NADPT,
		}
		da.Adapt(mcmc, samples, NBURN)
	default:
		// Burn
		for i := 0; i != NBURN; i++ {
			progress("Burning", i)
			if len(<-samples) == 0 {
				break
			}
		}
	}

	// Collect after burn-in
	y := make([]float64, len(x))
	n := 0.
	means := make([]float64, m.NComp)
	meanidx := make([]int, m.NComp)
	for i := 0; i != NITER; i++ {
		progress("Collecting", i)
		x := <-samples
		if len(x) == 0 {
			break
		}

		// Sort the components to take care of label switching
		for j := 0; j != m.NComp; j++ {
			means[j] = x[2*j]
			meanidx[j] = j
		}
		sortMeans(means, meanidx)

		// Means and standard deviations.
		iy := 0
		for j := 0; j != m.NComp; j++ {
			k := meanidx[j]
			y[iy] += x[2*k]
			iy++
			y[iy] += math.Exp(x[2*k+1])
			iy++
		}

		// We compute empirical means of component probabilities
		// because odds can shift.
		for range data {
			D.SoftMax(x[iy:iy+m.NComp], p)
			for j := 0; j != m.NComp; j++ {
				k := meanidx[j]
				y[iy] += p[k]
				iy++
			}
		}

		n++
	}
	for j := range y {
		y[j] /= n
	}
	mcmc.Stop()

	fmt.Printf("%32s\n", "")
	fmt.Printf("Posterior means:\n")

	switch mcmc := mcmc.(type) {
	case *infer.HMC:
		fmt.Printf(`* %s:
		accepted: %d
		rejected: %d
		rate: %.4g
	`,
			MCMC,
			mcmc.NAcc, mcmc.NRej,
			float64(mcmc.NAcc)/float64(mcmc.NAcc+mcmc.NRej))
	case *infer.NUTS:
		fmt.Printf(`* %s:
		accepted: %d
		rejected: %d
		rate: %.4g
		mean depth: %.4g
	`,
			MCMC,
			mcmc.NAcc, mcmc.NRej,
			float64(mcmc.NAcc)/float64(mcmc.NAcc+mcmc.NRej),
			mcmc.MeanDepth())
	default:
		panic(fmt.Errorf("invalid mcmc: %T", mcmc))
	}

	fmt.Printf("* Components:\n")
	iy := 0
	for j := 0; j != m.NComp; j++ {
		fmt.Printf("\t%d: mean=%.3f, stddev=%.3f\n",
			j, y[2*j], y[2*j+1])
		iy += 2
	}

	fmt.Printf("* Observations:\n")
	// Header
	fmt.Print("    #\t  value\t label")
	for j := 0; j != NCOMP; j++ {
		fmt.Printf("\t    p%d", j)
	}
	fmt.Println()
	// Values
	for i := range data {
		fmt.Printf("%5d\t%7.3f\t%4d", i, data[i], labels[i])
		for j := 0; j != m.NComp; j++ {
			fmt.Printf("\t%7.3f", y[iy])
			iy++
		}
		fmt.Println()
	}
}

func sortMeans(x []float64, idx []int) {
	for {
		swapped := false
		for i := 1; i != len(x); i++ {
			if x[i-1] > x[i] {
				x[i-1], x[i] = x[i], x[i-1]
				idx[i-1], idx[i] = idx[i], idx[i-1]
				swapped = true
			}
		}
		if !swapped {
			break
		}
	}
}
