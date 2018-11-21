all: gmm generate

gmm: model/ad/model.go main.go
	go build .
	./gmm

generate: gen/main.go
	go build -o $@ ./gen

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./gmm ./generate model/ad/*.go
