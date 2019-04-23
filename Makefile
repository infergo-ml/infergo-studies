STUDIES=funnel gmm lda lr lr-gonum wasm 

all: $(STUDIES)

push:
	for repo in origin ssh://git@github.com/infergo-ml/infergo-studies; do git push $$repo; git push --tags $$repo; done


clean:
	for x in $(STUDIES); do (cd $$x && make clean); done

# Studies
#
# Neal's funnel
.PHONY: funnel
funnel:
	(cd funnel && make)

# Gaussian mixture model
.PHONY: gmm
gmm:
	(cd gmm && make)

# Latent dirichlet allocation
.PHONY: lda
lda:
	(cd lda && make)

# Linear regression
.PHONY: lr
lr:
	(cd lr && make)

#  Linear regression via Gonum's BFGS
.PHONY: lr-gonum
lr-gonum:
	(cd lr-gonum && make)

#  WebAssembly
.PHONY: wasm
wasm:
	(cd wasm && make)
