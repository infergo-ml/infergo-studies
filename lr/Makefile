all: lr

lr: model/ad/model.go main.go
	go build .
	./lr

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lr model/ad/*.go
