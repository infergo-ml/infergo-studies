package main // import "bitbucket.org/dtolpin/infergo-studies/gmm"

import (
	. "bitbucket.org/dtolpin/infergo-studies/gmm/model/ad"
	. "bitbucket.org/dtolpin/infergo/dist"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

// Command line arguments

var (
	NCOMP = 2
	ALPHA = 1.
	TAU = 0.
	RATE  = 0.1
	NITER = 100
	EPS = 0.001
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.Usage = func() {
		fmt.Printf(`Gaussian mixture model: gmm [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.IntVar(&NCOMP, "ncomp", NCOMP, "number of components")
	flag.Float64Var(&ALPHA, "alpha", ALPHA, "Dirichlet diffusion")
	flag.Float64Var(&TAU, "tau", TAU, "precision of on odds")
	flag.Float64Var(&RATE, "rate", RATE, "learning rate")
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.Float64Var(&EPS, "eps", EPS, "optimization precision")
}

func main() {
	flag.Parse()

	if flag.NArg() > 1 {
		fmt.Fprintf(os.Stderr,
			"unexpected positional arguments: %v\n",
			flag.Args()[1:])
		os.Exit(1)
	}

	// Get the data
	var (
		data   []float64
		labels []int
	)
	if flag.NArg() == 1 {
		// Read the CSV
		fname := flag.Arg(0)
		file, err := os.Open(fname)
		if err != nil {
			fmt.Fprintf(os.Stderr,
				"Cannot open data file %q: %v\n", fname, err)
			os.Exit(1)
		}
		rdr := csv.NewReader(file)
		for {
			record, err := rdr.Read()
			if err == io.EOF {
				break
			}
			value, err := strconv.ParseFloat(record[0], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid data: %v\n", err)
				os.Exit(1)
			}
			label, err := strconv.Atoi(record[1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid label: %v\n", err)
				os.Exit(1)
			}
			data = append(data, value)
			labels = append(labels, label)
		}
		file.Close()
	} else {
		// Use an embedded data set, for self-check
		data = []float64{
			1.899, -1.11, -0.9068, 1.291, -0.755,
			-0.4422, -0.144, 1.214, -0.8183, -0.3386,
			0.3863, -1.036, -0.6248, 1.014, 1.336,
			-1.487, 0.8223, -0.4268, 0.6754, 0.6206,
		}
		labels = []int{
			1, 0, 0, 1, 0,
			0, 0, 1, 0, 0,
			1, 0, 0, 1, 1,
			0, 1, 0, 1, 1,
		}
	}

	// Define the problem
	m := &Model{
		Data:  data,
		NComp: NCOMP,
		Alpha: ALPHA,
		Tau: TAU,
	}
	x := make([]float64, 2*m.NComp+len(m.Data)*m.NComp)

	// Set a starting  point
	if m.NComp == 1 {
		x[0] = 0.
		x[1] = 1.
	} else {
		// Spread the initial components wide and thin
		for j := 0; j != m.NComp; j++ {
			x[2*j] = -1. + 2./float64(m.NComp-1)*float64(j)
			x[2*j+1] = 1.
		}
	}

	// Run the optimizer
	opt := &infer.Adam{Rate: RATE}
	ll0, _ := opt.Step(m, x)
	ll, llprev := ll0, ll0
	iter := 0
	for ; iter != NITER; iter++ {
		ll, _ = opt.Step(m, x)
		if math.Abs(ll - llprev)/math.Abs(ll + llprev) < EPS {
			break
		}
		llprev = ll
	}

	// Print the result.
	fmt.Printf("MLE (after %d iterations):\n", iter)
	fmt.Printf("* Log-likelihood: %7.3f => %7.3f\n", ll0, ll)
	fmt.Printf("* Components:\n")
	for j := 0; j != m.NComp; j++ {
		fmt.Printf("\t%d: mean=%.3f, stddev=%.3f\n",
			j, x[2*j], math.Exp(0.5*x[2*j+1]))
	}

	fmt.Printf("* Observations:\n")
	// Header
	fmt.Print("\t  value\t label")
	for j := 0; j != NCOMP; j++ {
		fmt.Printf("\t   p%d", j)
	}
	fmt.Println()
	// Values
	p := make([]float64, NCOMP)
	ix := 2*m.NComp
	for i := range data {
		fmt.Printf("\t%7.3f\t%4d", data[i], labels[i])
		SoftMax(x[ix:ix+m.NComp], p)
		ix += m.NComp
		for j := 0; j != m.NComp; j++ {
			fmt.Printf("\t%7.3f", p[j])
		}
		fmt.Println()
	}
}
