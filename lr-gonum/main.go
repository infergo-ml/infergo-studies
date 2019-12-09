package main // import "bitbucket.org/dtolpin/infergo-studies/lr-gonum"

import (
	. "bitbucket.org/dtolpin/infergo-studies/lr-gonum/model/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"encoding/csv"
	"flag"
	"fmt"
	"gonum.org/v1/gonum/optimize"
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
	NITER  = 100
	EPS    = 1e-6
	MTSAFE = false
	NTASKS = 1
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.Usage = func() {
		fmt.Printf(`Linear regression: lr [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
	flag.IntVar(&NITER, "niter", NITER, "number of iterations")
	flag.Float64Var(&EPS, "eps", EPS, "optimization precision")
	flag.BoolVar(&MTSAFE, "mtsafe", MTSAFE, "multithread-safe tape")
	flag.IntVar(&NTASKS, "ntasks", NTASKS, "number of concurrent tasks")
}

func main() {
	flag.Parse()

	if flag.NArg() > 1 {
		fmt.Fprintf(os.Stderr,
			"unexpected positional arguments: %v\n",
			flag.Args()[1:])
		os.Exit(1)
	}

	if MTSAFE {
		if !ad.MTSafeOn() {
			fmt.Println("Multithreading is not supported.")
		}
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

	// Wrap into Gonum optimization problem
	Func, Grad := infer.FuncGrad(m)
	p := optimize.Problem{Func: Func, Grad: Grad}

	// Initial log likelihood
	ll0 := m.Observe(x)

	result, err := optimize.Minimize(p, x, &optimize.Settings{
		MajorIterations:   NITER,
		GradientThreshold: EPS,
		Concurrent:        NTASKS,
	}, nil)
	if err != nil {
		panic(err)
	}
	x = result.X

	// Final log likelihood
	ll := m.Observe(x)

	ad.DropAllTapes()

	// Print the results.
	fmt.Printf("MLE (after %d iterations):\n", result.Stats.MajorIterations)
	fmt.Printf("* Log-likelihood: %7.3f => %7.3f\n", ll0, ll)
	fmt.Printf("* alpha: %.3f\n", x[0])
	fmt.Printf("* beta: %.3f\n", x[1])
	fmt.Printf("* sigma: %.3f\n", math.Exp(x[2]))
}
