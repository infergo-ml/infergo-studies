all: lr-gonum

lr-gonum: model/ad/model.go main.go
	go build .
	./lr-gonum

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lr-gonum model/ad/*.go
