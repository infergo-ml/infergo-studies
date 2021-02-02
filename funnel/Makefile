all: funnel

GO=go

funnel: model/ad/model.go main.go
	$(GO) build .
	./funnel

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./funnel model/ad/*.go
