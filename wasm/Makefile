all: main.wasm wasm_exec.js

GO=go

PORT=8080
# works with both GNU and BSD make
WASM_EXEC_JS=`echo $$($(GO) env GOROOT)/misc/wasm/wasm_exec.js`
GO_JS_WASM_EXEC=`echo $$($(GO) env GOROOT)/misc/wasm/go_js_wasm_exec`

main.wasm: model/ad/model.go main.go
	GOOS=js GOARCH=wasm $(GO) build -o main.wasm

# the server target runs a local http server serving the page
# with go code
server: main.wasm
	goexec 'http.ListenAndServe(":$(PORT)", http.FileServer(http.Dir(".")))'

# the node target runs the program with Node.js.
node: main.wasm
	 GOOS=js GOARCH=wasm $(GO) run -exec="$(GO_JS_WASM_EXEC)" .

wasm_exec.js:
	cp $(WASM_EXEC_JS) .

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./main.wasm model/ad/*.go wasm_exec.js
