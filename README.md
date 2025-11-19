# gemini-things-go

A helper library for generating things with [Gemini API](https://github.com/googleapis/go-genai).

[![Go Reference](https://pkg.go.dev/badge/github.com/meinside/gemini-things-go.svg)](https://pkg.go.dev/github.com/meinside/gemini-things-go)

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
	apiKey = `AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00` // Your API key here
	model  = "gemini-2.5-flash"
)

func main() {
	if client, err := gt.NewClient(
		apiKey,
		gt.WithModel(model),       // Specify the model
		gt.WithTimeoutSeconds(60), // Set a 60-second timeout for operations
		gt.WithMaxRetryCount(5),   // Configure a maximum of 5 retries on 5xx server errors
	); err == nil {
		// convert prompts and histories to contents for generation
		if contents, err := client.PromptsToContents(
			context.TODO(),
			[]gt.Prompt{
				gt.PromptFromText(`What is the answer to life, the universe, and everything?`),
			},
			nil,
		); err == nil {
			if res, err := client.Generate(
				context.TODO(),
				contents,
			); err == nil {
				// do something with `client`
				fmt.Printf("response: %+v\n", res)
			} else {
				panic(fmt.Errorf("failed to generate: %w", err))
			}
		} else {
			panic(fmt.Errorf("failed to convert prompts: %w", err))
		}
	} else {
		panic(fmt.Errorf("failed to create client: %w", err))
	}
}
```

## Test

```bash
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test

# for verbose output:
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" VERBOSE=true go test
```

