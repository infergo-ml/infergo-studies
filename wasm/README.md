# Probabilistic 'hello world' in the browser

A version of the probabilistic 'hello world' runing in the
browser. Basic usage:

	goexec 'http.ListenAndServe(":8080", http.FileServer(http.Dir(".")))'

And then access `http://localhost:8080` in the browser. See the
[`infergo`](http://infergo.org/)
[hello](`http://bitbucket.org/dtolpin/infergo/src/master/examples/hello`)
example for details.
