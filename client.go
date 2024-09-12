package gt

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
)

const (
	defaultTimeoutSeconds    = 30
	defaultSystemInstruction = `You are chat bot for helping the user.

Respond to the user according to the following principles:
- Do not repeat the user's request.
- Be as accurate as possible.
- Be as truthful as possible.
- Be as comprehensive and informative as possible.
`
)

// Client struct
type Client struct {
	model  string
	apiKey string

	systemInstructionFunc FnSystemInstruction

	timeoutSeconds int

	Verbose bool
}

// StreamCallbackData struct contains the data for stream callback function.
type StreamCallbackData struct {
	// when there is a text delta,
	TextDelta *string

	// when there is a function call,
	FunctionCall *genai.FunctionCall

	// when the number of tokens are calculated,
	NumTokens *NumTokens

	// when there is a finish reason,
	FinishReason *genai.FinishReason

	// when there is an error,
	Error error

	// NOTE: TODO: add more data here
}

// NumTokens struct for input/output token numbers
type NumTokens struct {
	Input  int32
	Output int32
}

// function definitions
type FnSystemInstruction func() string
type FnStreamCallback func(callbackData StreamCallbackData)

// NewClient returns a new client with given values.
func NewClient(model, apiKey string) *Client {
	return &Client{
		model:  model,
		apiKey: apiKey,

		systemInstructionFunc: func() string {
			return defaultSystemInstruction
		},

		timeoutSeconds: defaultTimeoutSeconds,

		Verbose: false,
	}
}

// SetSystemInstructionFunc sets the system instruction function.
func (c *Client) SetSystemInstructionFunc(fn FnSystemInstruction) {
	c.systemInstructionFunc = fn
}

// SetTimeout sets the timeout in seconds.
func (c *Client) SetTimeout(seconds int) {
	c.timeoutSeconds = seconds
}

// GenerationOptions struct for function Generate.
type GenerationOptions struct {
	// generation config
	Config *genai.GenerationConfig

	// tool config
	Tools      []*genai.Tool
	ToolConfig *genai.ToolConfig
}

// GenerateStreamed generates with given values synchronously.
func (c *Client) GenerateStreamed(
	ctx context.Context,
	promptText string,
	promptFiles []io.Reader,
	fnStreamCallback FnStreamCallback,
	options ...*GenerationOptions,
) error {
	// set timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
	defer cancel()

	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	// generate client and model
	client, model, err := c.getClientAndModel(ctx, opts)
	if err != nil {
		return fmt.Errorf("failed to get client and model: %s", err)
	}
	defer client.Close()

	// generate parts for prompting
	var prompts []genai.Part
	prompts, err = c.buildPromptParts(ctx, client, promptText, promptFiles)
	if err != nil {
		return fmt.Errorf("failed to build prompts: %s", err)
	}

	// number of tokens
	var numTokensInput int32 = 0
	var numTokensOutput int32 = 0

	// check callback function
	if fnStreamCallback == nil {
		fnStreamCallback = func(callbackData StreamCallbackData) {
			log.Printf("> stream callback data: %s", prettify(callbackData))
		}
	}

	// generate and stream response
	iter := model.GenerateContentStream(ctx, prompts...)
	for {
		if it, err := iter.Next(); err == nil {
			var candidate *genai.Candidate
			var content *genai.Content
			var parts []genai.Part

			if len(it.Candidates) > 0 {
				// update number of tokens
				numTokensInput = it.UsageMetadata.PromptTokenCount
				numTokensOutput = it.UsageMetadata.TotalTokenCount - it.UsageMetadata.PromptTokenCount

				candidate = it.Candidates[0]
				content = candidate.Content

				if content != nil && len(content.Parts) > 0 {
					parts = content.Parts
				} else if candidate.FinishReason > 0 {
					fnStreamCallback(StreamCallbackData{
						FinishReason: &candidate.FinishReason,
					})
				} else {
					fnStreamCallback(StreamCallbackData{
						Error: fmt.Errorf("no content in candidate: %s", prettify(candidate)),
					})
				}
			}

			for _, part := range parts {
				if text, ok := part.(genai.Text); ok { // (text)
					fnStreamCallback(StreamCallbackData{
						TextDelta: genai.Ptr(string(text)),
					})
				} else if fc, ok := part.(genai.FunctionCall); ok { // (function call)
					fnStreamCallback(StreamCallbackData{
						FunctionCall: &fc,
					})
				} else {
					fnStreamCallback(StreamCallbackData{
						Error: fmt.Errorf("unsupported type of part for streaming: %s", prettify(part)),
					})
				}
			}
		} else {
			if err != iterator.Done {
				fnStreamCallback(StreamCallbackData{
					Error: fmt.Errorf("failed to iterate stream: %s", errorString(err)),
				})
			}
			break
		}
	}

	// pass the number of tokens
	fnStreamCallback(StreamCallbackData{
		NumTokens: &NumTokens{
			Input:  numTokensInput,
			Output: numTokensOutput,
		},
	})

	return nil
}

// Generate generates with given values synchronously.
func (c *Client) Generate(
	ctx context.Context,
	promptText string,
	promptFiles []io.Reader,
	options ...*GenerationOptions,
) (res *genai.GenerateContentResponse, err error) {
	// set timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
	defer cancel()

	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	// generate client and model
	client, model, err := c.getClientAndModel(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to get client and model: %s", err)
	}
	defer client.Close()

	// generate parts for prompting
	var prompts []genai.Part
	prompts, err = c.buildPromptParts(ctx, client, promptText, promptFiles)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompts: %s", err)
	}

	// generate and return the response
	return model.GenerateContent(ctx, prompts...)
}
