STUDIES=funnel gmm lda lr lr-gonum wasm 

GO=go

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
	(cd funnel && make GO=$(GO))

# Gaussian mixture model
.PHONY: gmm
gmm:
	(cd gmm && make GO=$(GO))

# Latent dirichlet allocation
.PHONY: lda
lda:
	(cd lda && make GO=$(GO))

# Linear regression
.PHONY: lr
lr:
	(cd lr && make GO=$(GO))

#  Linear regression via Gonum's BFGS
.PHONY: lr-gonum
lr-gonum:
	(cd lr-gonum && make GO=$(GO))

#  WebAssembly
.PHONY: wasm
wasm:
	(cd wasm && make GO=$(GO))
