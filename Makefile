push:
	for repo in origin ssh://git@github.com/dtolpin/infergo-studies ssh://git@github.com/infergo-ml/infergo-studies; do git push $$repo; git push --tags $$repo; done
