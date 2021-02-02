all: lda

GO=go

lda: model/ad/model.go main.go
	$(GO) build .
	./lda

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lda model/ad/*.go
