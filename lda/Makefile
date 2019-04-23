all: lda

lda: model/ad/model.go main.go
	go build .
	./lda

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lda model/ad/*.go
