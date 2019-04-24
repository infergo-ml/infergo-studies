package main // import "bitbucket.org/dtolpin/infergo-studies/lda"

import (
	. "bitbucket.org/dtolpin/infergo-studies/lda/model"
	ad "bitbucket.org/dtolpin/infergo-studies/lda/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
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
	EPS      = 1E-6
	STEP     = 0.5
	DEPTH    = 5.
	MAXDEPTH = 0
	GAMMA    = 0.
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
	flag.Float64Var(&GAMMA, "gamma", GAMMA, "log(sigma) for N(0, sigma) prior")
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
	var m Model

	if flag.NArg() == 1 {
		// Gamma is set to the value of the command-line flag,
		// but will be overridden if specified in the data file.
		m = Model{Gamma: GAMMA}
		// read the data
		fname := flag.Arg(0)
		file, err := os.Open(fname)
		if err != nil {
			fmt.Fprintf(os.Stderr,
				"Cannot open data file %q: %v\n", fname, err)
			os.Exit(1)
		}
		rdr := json.NewDecoder(file)
		err = rdr.Decode(&m)
		if err != nil {
			fmt.Fprintf(os.Stderr,
				"Error parsing data file %q: %v\n", fname, err)
			os.Exit(1)
		}
	} else {
		// use built-in data for self-testing
		m = Model{
			K: 2,
			V: 4,
			M: 5,
			N: 50,
			Word: []int{
				1, 2, 1, 2, 1, 2, 3, 2, 1, 2,
				3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
				1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
				3, 4, 3, 4, 3, 4, 3, 4, 3, 4,
				1, 2, 1, 3, 1, 2, 1, 2, 1, 2,
			},
			Doc: []int{
				1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
				2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
				3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
				4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
				5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
			},
			Alpha: []float64{0.5, 0.5},
			Beta:  []float64{0.25, 0.25, 0.25, 0.25},
			Gamma: GAMMA,
		}
	}

	// Create the model and initialize the parameters
	adm := (*ad.Model)(&m)
	x := make([]float64, m.M*m.K+m.K*m.V)
	for i := range x {
		x[i] = rand.NormFloat64()
	}

	// Run the optimizer
	opt := &infer.Adam{Rate: RATE}
	ll0, _ := opt.Step(adm, x)
	ll, llprev := ll0, ll0
	iter := 0
	for ; iter != NITER; iter++ {
		ll, _ = opt.Step(adm, x)
		if math.Abs(ll-llprev)/math.Abs(ll+llprev) < EPS {
			break
		}
		llprev = ll
	}

	// Print the results.
	fmt.Printf("MLE (after %d iterations):\n", iter)
	fmt.Printf("* Log-likelihood: %7.3f => %7.3f\n", ll0, ll)
	// Print topics
	{
		x := x

		theta := make([][]float64, m.M)
		m.FetchSimplices(&x, m.K, theta)
		printSimplices("Document topics", theta)

		phi := make([][]float64, m.K)
		m.FetchSimplices(&x, m.V, phi)
		printSimplices("Topic words", phi)
	}

	// Infer the posterior with HMC.
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
	mcmc.Sample(adm, x, samples)

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
			printProgress("Burning", i)
			if len(<-samples) == 0 {
				break
			}
		}
	}

	// Collect after burn-in
	for i := 0; i != NITER; i++ {
		printProgress("Collecting", i)
		x := <-samples
		if len(x) == 0 {
			break
		}
		if (i+1)%(NITER/10) == 0 {
			theta := make([][]float64, m.M)
			m.FetchSimplices(&x, m.K, theta)

			phi := make([][]float64, m.K)
			m.FetchSimplices(&x, m.V, phi)
			printSimplices("Topic words", phi)
		}
	}
	mcmc.Stop()
	fmt.Printf("                                      \r")

	// Print one sample of document topics
	theta := make([][]float64, m.M)
	m.FetchSimplices(&x, m.K, theta)
	printSimplices("Document topics", theta)

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

}

// Print progress for the impatient
func printProgress(stage string, i int) {
	if (i+1)%10 == 0 {
		fmt.Fprintf(os.Stderr, "%10s: %5d\r", stage, i+1)
	}
}

// Print simplices
func printSimplices(title string, simplices [][]float64) {
	fmt.Println(title)
	for i := range simplices {
		fmt.Printf("%2d:", i+1)
		for j := range simplices[i] {
			fmt.Printf(" %.2f", simplices[i][j])
		}
		fmt.Println()
	}
}
