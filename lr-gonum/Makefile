all: lr-gonum

GO=go

lr-gonum: model/ad/model.go main.go
	$(GO) build .
	./lr-gonum

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./lr-gonum model/ad/*.go

push:
	for repo in origin ssh://git@github.com/infergo-ml/infergo-studies; do git push $$repo; git push --tags $$repo; done

