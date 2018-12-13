all: main.wasm wasm_exec.js

# works with both GNU and BSD make
WASM_EXEC_JS=`echo $$(go env GOROOT)/misc/wasm/wasm_exec.js`

main.wasm: model/ad/model.go main.go
	GOOS=js GOARCH=wasm go build -o main.wasm

wasm_exec.js:
	cp $(WASM_EXEC_JS) .

model/ad/model.go: model/model.go
	deriv model

clean:
	rm -f ./main.wasm model/ad/*.go wasm_exec.js
