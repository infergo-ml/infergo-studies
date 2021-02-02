all: gmm generate

GO=go

gmm: model/ad/model.go main.go
	$(GO) build .
	./gmm
	./gmm -alpha 0.1 -tau 1.

generate: gen/main.go
	$(GO) build -o $@ ./gen

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./gmm ./generate model/ad/*.go
