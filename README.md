# gemini-things-go

A helper library for generating things with [Gemini API](https://github.com/googleapis/go-genai).

## Usage

See the code examples in the [generation_test.go](https://github.com/meinside/gemini-things-go/blob/master/generation_test.go) file.

```go
package main

import (
	"context"
	"fmt"

	gt "github.com/meinside/gemini-things-go"
)

const (
	apiKey = `AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00`
	model  = "gemini-2.0-flash"
)

func main() {
	if client, err := gt.NewClient(apiKey, model); err == nil {
		// do something with `client`
		if res, err := client.Generate(
			context.TODO(),
			[]gt.Prompt{
				gt.PromptFromText(`What is the answer to life, the universe, and everything?`),
			},
		); err == nil {
			fmt.Printf("response: %+v\n", res)
		}
	} else {
		panic(err)
	}
}
```

## Test

```bash
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test

# for verbose output:
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" VERBOSE=true go test
```

