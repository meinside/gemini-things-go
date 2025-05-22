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
	apiKey = `YOUR_API_KEY` // Replace with your actual API key
	model  = "gemini-1.5-flash-latest" // Or any other model you intend to use
)

func main() {
	// Basic client initialization
	if client, err := gt.NewClient(apiKey, gt.WithModel(model)); err == nil {
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

### Configuring Timeout and Retries

You can configure the client with custom timeouts for API operations and set a maximum retry count for retriable server-side errors (e.g., 5xx errors).

```go
package main

import (
	"context"
	"fmt"
	"log"

	gt "github.com/meinside/gemini-things-go"
)

const (
	apiKey = `YOUR_API_KEY` // Replace with your actual API key
	model  = "gemini-1.5-flash-latest" // Or any other model you intend to use
)

func main() {
	// Example of configuring client with custom timeout and retries
	client, err := gt.NewClient(
		apiKey,
		gt.WithModel(model),             // Specify the model
		gt.WithTimeoutSeconds(60),       // Set a 60-second timeout for operations
		gt.WithMaxRetryCount(5),         // Configure a maximum of 5 retries on 5xx server errors
	)
	if err != nil {
		log.Fatalf("Failed to create client: %v", err)
	}
	defer client.Close() // Ensure resources are cleaned up

	// Example usage:
	promptText := "What is the answer to life, the universe, and everything?"
	fmt.Printf("Generating content for prompt: '%s'\n", promptText)

	res, err := client.Generate(
		context.TODO(),
		[]gt.Prompt{
			gt.PromptFromText(promptText),
		},
	)
	if err != nil {
		log.Fatalf("Failed to generate content: %v", err)
	}

	// Assuming the response has at least one candidate and one part.
	// In a real application, you should check these conditions.
	if len(res.Candidates) > 0 && len(res.Candidates[0].Content.Parts) > 0 {
		if textPart, ok := res.Candidates[0].Content.Parts[0].(genai.Text); ok {
			fmt.Printf("Response: %s\n", string(textPart))
		} else {
			fmt.Println("Response part is not text.")
		}
	} else {
		fmt.Println("No content received.")
	}
}

```

## Test

```bash
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" go test

# for verbose output:
$ API_KEY="AIabcdefghijklmnopqrstuvwxyz_ABCDEFG-00000000-00" VERBOSE=true go test
```

