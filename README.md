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
	"time"

	gt "github.com/meinside/gemini-things-go"
)

const (
	apiKey = `AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00` // Your API key here
	model  = "gemini-2.5-flash"

	contentsBuildupTimeoutSeconds = 60
	generationTimeoutSeconds      = 30
)

func main() {
	if client, err := gt.NewClient(
		apiKey,
		gt.WithModel(model),     // Specify the model
		gt.WithMaxRetryCount(5), // Configure a maximum of 5 retries on 5xx server errors
	); err == nil {
		ctxContents, cancelContents := context.WithTimeout(context.TODO(), contentsBuildupTimeoutSeconds*time.Second)
		defer cancelContents()

		// convert prompts and histories to contents for generation
		if contents, err := client.PromptsToContents(
			ctxContents,
			[]gt.Prompt{
				gt.PromptFromText(`What is the answer to life, the universe, and everything?`),
			},
			nil,
		); err == nil {
			ctxGenerate, cancelGenerate := context.WithTimeout(context.TODO(), generationTimeoutSeconds*time.Second)
			defer cancelGenerate()

			// generate with contents
			if res, err := client.Generate(
				ctxGenerate,
				contents,
			); err == nil {
				// do something with the generated response
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

Test with Gemini API key:

```bash
$ GEMINI_API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test -timeout 0

# for verbose output:
$ GEMINI_API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" VERBOSE=true go test -timeout 0

# for testing free APIs only
$ GEMINI_API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test -run="Free"

# for testing paid APIs only
$ GEMINI_API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test -run="Paid"
```

or with Google Cloud credential file (for Vertex AI):

```bash
$ GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json" PROJECT_ID="my-google-project-id" go test -timeout 0
```

