all: funnel

funnel: model/ad/model.go main.go
	go build .
	./funnel

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./funnel model/ad/*.go
