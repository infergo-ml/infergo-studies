package main // import "bitbucket.org/dtolpin/infergo-studies/funnel"

import (
	. "bitbucket.org/dtolpin/infergo-studies/funnel/model/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/model"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

var (
	NITER = 10000
	L     = 5
	EPS   = 0.1
	MODEL = "naive"
	NDIM  = 9
	PLOT  = "funnel-naive.png"
)

func init() {
	rand.Seed(time.Now().UTC().UnixNano())
	flag.StringVar(&MODEL, "model", MODEL, "model (naive or repar)")
	flag.IntVar(&NITER, "niter", NITER, "number of HMC iterations")
	flag.IntVar(&L, "L", L, "number of HMC leapfrog steps")
	flag.Float64Var(&EPS, "eps", EPS, "leapfrog step")
	flag.IntVar(&NDIM, "ndim", NDIM, "number of funnel dimensions")
	flag.StringVar(&PLOT, "plot", PLOT, "number of funnel dimensions")
	flag.Usage = func() {
		fmt.Printf(`Neal's funnel: funnel [OPTIONS]` + "\n")
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	PLOT = strings.Replace(PLOT, "naive", MODEL, 1)
	if flag.NArg() > 0 {
		fmt.Fprintf(os.Stderr,
			"unexpected positional arguments: %v\n",
			flag.Args()[0:])
		os.Exit(1)
	}

	// Define the problem
	var m model.Model
	MODEL = strings.ToLower(MODEL)
	switch MODEL {
	case "naive":
		m = NaiveModel{}
	case "repar":
		m = ReparModel{}
	default:
		fmt.Fprintf(os.Stderr, "unknown model: %q", MODEL)
		os.Exit(1)
	}
	x := make([]float64, NDIM + 1)

	//Infer the posterior with HMC.
	var mcmc infer.MCMC
	mcmc = &infer.HMC{
		L:	 L,
		Eps: EPS,
	}

	samples := make(chan []float64)
	mcmc.Sample(m, x, samples)

	// Print progress for the impatient
	progress := func(stage string, i int) {
		if (i+1)%100 == 0 {
			fmt.Fprintf(os.Stderr, "%10s: %5d\r", stage, i+1)
		}
	}

	// Burn
	for i := 0; i != NITER; i++ {
		progress("Burning", i)
		if len(<-samples) == 0 {
			break
		}
	}

	// Collect after burn-in
	pts := make(plotter.XYs, NITER)
	for i := 0; i != NITER; i++ {
		progress("Collecting", i)
		x := <-samples
		if len(x) == 0 {
			break
		}
		switch MODEL {
		case "naive":
			pts[i].X = x[1]
			pts[i].Y = x[0]
		case "repar":
			pts[i].Y = 3.0 * x[1]
			pts[i].X = math.Exp(0.5*pts[i].Y) * x[0]
		default:
			panic(MODEL)
		}
	}
	mcmc.Stop()

	p, err := plot.New()
	if err != nil {
		panic(err)
	}

	p.Title.Text = "Neal's funnel"
	p.X.Label.Text = "x[0]"
	p.Y.Label.Text = "x[1]"

	err = plotutil.AddScatters(p, pts)
	if err != nil {
		panic(err)
	}

	if err := p.Save(5*vg.Inch, 5*vg.Inch, PLOT); err != nil {
		panic(err)
	}
}
