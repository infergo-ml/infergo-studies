package main // import "bitbucket.org/dtolpin/infergo-studies/lr"

import (
	. "bitbucket.org/dtolpin/infergo-studies/lr/model/ad"
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
	// Inference algorithm parameters
	RATE  = 0.1
	NITER = 1000
	EPS   = 1e-6
)

func init() {
	rand.Seed(time.Now().UnixNano())
	flag.Usage = func() {
		fmt.Printf(`Linear regression: lr [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
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
		data [][]float64
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
			x, err := strconv.ParseFloat(record[0], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid data: %v\n", err)
				os.Exit(1)
			}
			y, err := strconv.ParseFloat(record[1], 64)
			if err != nil {
				fmt.Fprintf(os.Stderr, "invalid label: %v\n", err)
				os.Exit(1)
			}
			data = append(data, []float64{x, y})
		}
		file.Close()
	} else {
		// Use an embedded data set, for self-check
		data = [][]float64{
			{0., 0.9},
			{1., 2.1},
			{2., 2.9},
			{3., 4.05},
			{4., 5.1},
			{5., 5.0},
		}
	}

	// Define the problem
	m := &Model{
		Data: data,
	}
	x := []float64{0, 0, 0}

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

	// Print the results.
	fmt.Printf("MLE (after %d iterations):\n", iter)
	fmt.Printf("* Log-likelihood: %7.3f => %7.3f\n", ll0, ll)
	fmt.Printf("* alpha: %.3f\n", x[0])
	fmt.Printf("* beta: %.3f\n", x[1])
	fmt.Printf("* sigma: %.3f\n", math.Exp(x[2]))
}
