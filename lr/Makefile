all: lr

GO=go

lr: model/ad/model.go main.go
	$(GO) build .
	./lr

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lr model/ad/*.go
