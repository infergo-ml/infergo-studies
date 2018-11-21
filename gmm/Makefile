all: gmm

gmm: model/ad/model.go main.go
	go build .
	./gmm

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./gmm model/ad/*.go
