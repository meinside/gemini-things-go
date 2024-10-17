package gt

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
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
	client *genai.Client

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

	// when there is a code execution result,
	ExecutableCode      *genai.ExecutableCode
	CodeExecutionResult *genai.CodeExecutionResult

	// when the number of tokens are calculated, (after the stream is finished)
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
	Cached int32
}

// function definitions
type FnSystemInstruction func() string
type FnStreamCallback func(callbackData StreamCallbackData)

// NewClient returns a new client with given values.
func NewClient(model, apiKey string) (*Client, error) {
	// genai client
	var client *genai.Client
	var err error
	client, err = genai.NewClient(context.TODO(), option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %s", err)
	}

	return &Client{
		model:  model,
		apiKey: apiKey,
		client: client,

		systemInstructionFunc: func() string {
			return defaultSystemInstruction
		},

		timeoutSeconds: defaultTimeoutSeconds,

		Verbose: false,
	}, nil
}

// Close closes the client.
func (c *Client) Close() error {
	return c.client.Close()
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

	// safety settings: harm block threshold
	HarmBlockThreshold *genai.HarmBlockThreshold

	// cached context
	CachedContextName *string

	// history (for session)
	History []*genai.Content
}

// CacheContext caches the context with given values and return the name of the cached context.
func (c *Client) CacheContext(ctx context.Context, systemInstruction, promptText *string, promptFiles []io.Reader, tools []*genai.Tool, toolConfig *genai.ToolConfig) (cachedContextName string, err error) {
	if c.Verbose {
		log.Printf("> caching context with system prompt: %s, prompt: %s, %d files, tools: %s, and tool config: %s", prettify(systemInstruction), prettify(promptText), len(promptFiles), prettify(tools), prettify(toolConfig))
	}

	// generate parts for context
	var prompts []genai.Part
	prompts, err = c.buildPromptParts(ctx, promptText, promptFiles)
	if err != nil {
		return "", fmt.Errorf("failed to build prompts for caching context: %s", err)
	}

	argcc := &genai.CachedContent{
		Model: c.model,
	}

	// system instruction
	if systemInstruction != nil {
		argcc.SystemInstruction = genai.NewUserContent(genai.Text(*systemInstruction))
	}

	// prompts
	if len(prompts) > 0 {
		argcc.Contents = []*genai.Content{genai.NewUserContent(prompts...)}
	}

	// tools and tool config
	if tools != nil {
		argcc.Tools = tools
	}
	if toolConfig != nil {
		argcc.ToolConfig = toolConfig
	}

	// create cached context
	var cc *genai.CachedContent
	if cc, err = c.client.CreateCachedContent(ctx, argcc); err != nil {
		return "", fmt.Errorf("failed to cache context: %s", err)
	}

	return cc.Name, nil
}

// generate stream iterator with given values
func (c *Client) generateStream(
	ctx context.Context,
	promptText string,
	promptFiles []io.Reader,
	options ...*GenerationOptions,
) (iterator *genai.GenerateContentResponseIterator, err error) {
	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	if c.Verbose {
		log.Printf("> generating streamed content with prompt '%s' and %d files (options: %s)", promptText, len(promptFiles), prettify(opts))
	}

	// generate parts for prompting
	var prompts []genai.Part
	prompts, err = c.buildPromptParts(ctx, &promptText, promptFiles)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompts: %s", err)
	}

	// generate model
	model, err := c.getModel(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to get model for streamed generation: %s", err)
	}

	// generate and stream response
	session := model.StartChat()
	if opts != nil && len(opts.History) > 0 {
		session.History = opts.History
	}

	return session.SendMessageStream(ctx, prompts...), nil
}

// GenerateStreamIterated generates stream iterator with given values.
//
// It does not timeout itself, so set timeout with `ctx` when needed.
func (c *Client) GenerateStreamIterated(
	ctx context.Context,
	promptText string,
	promptFiles []io.Reader,
	options ...*GenerationOptions,
) (iterator *genai.GenerateContentResponseIterator, err error) {
	return c.generateStream(ctx, promptText, promptFiles, options...)
}

// GenerateStreamed generates with given values synchronously and calls back `fnStreamCallback`.
//
// It times out in `timeoutSeconds` seconds.
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

	if iter, err := c.generateStream(ctx, promptText, promptFiles, options...); err == nil {
		// number of tokens
		var numTokensInput int32 = 0
		var numTokensOutput int32 = 0
		var numTokensCached int32 = 0

		for {
			if it, err := iter.Next(); err == nil {
				if c.Verbose {
					log.Printf("> iterating stream response: %s", prettify(it))
				}

				var candidate *genai.Candidate
				var content *genai.Content
				var parts []genai.Part

				if len(it.Candidates) > 0 {
					// update number of tokens
					if numTokensInput < it.UsageMetadata.PromptTokenCount {
						numTokensInput = it.UsageMetadata.PromptTokenCount
					}
					if numTokensOutput < it.UsageMetadata.TotalTokenCount-it.UsageMetadata.PromptTokenCount {
						numTokensOutput = it.UsageMetadata.TotalTokenCount - it.UsageMetadata.PromptTokenCount
					}
					if numTokensCached < it.UsageMetadata.CachedContentTokenCount {
						numTokensCached = it.UsageMetadata.CachedContentTokenCount
					}

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
					} else if code, ok := part.(genai.ExecutableCode); ok { // (code execution: executable code)
						fnStreamCallback(StreamCallbackData{
							ExecutableCode: &code,
						})
					} else if result, ok := part.(genai.CodeExecutionResult); ok { // (code execution: result)
						fnStreamCallback(StreamCallbackData{
							CodeExecutionResult: &result,
						})
					} else { // NOTE: TODO: add more conditions here
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
				Cached: numTokensCached,
			},
		})
	} else {
		fnStreamCallback(StreamCallbackData{
			Error: fmt.Errorf("failed to generate stream: %s", errorString(err)),
		})

		return fmt.Errorf("failed to generate stream: %s", err)
	}

	return nil
}

// Generate generates with given values synchronously.
//
// It times out in `timeoutSeconds` seconds.
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

	if c.Verbose {
		log.Printf("> generating content with prompt '%s' and %d files (options: %s)", promptText, len(promptFiles), prettify(opts))
	}

	// generate parts for prompting
	var prompts []genai.Part
	prompts, err = c.buildPromptParts(ctx, &promptText, promptFiles)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompts: %s", err)
	}

	// generate model
	model, err := c.getModel(ctx, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to get model for generation: %s", err)
	}

	// return the generated response
	session := model.StartChat()
	if opts != nil && len(opts.History) > 0 {
		session.History = opts.History
	}

	return session.SendMessage(ctx, prompts...)
}

// SetCachedContextExpireTime sets the expiration time of a cached context.
//
// (default: 1 hour later)
func (c *Client) SetCachedContextExpireTime(ctx context.Context, cachedContextName string, expireTime time.Time) (err error) {
	var cc *genai.CachedContent
	if cc, err = c.client.GetCachedContent(ctx, cachedContextName); err == nil {
		_, err = c.client.UpdateCachedContent(ctx, cc, &genai.CachedContentToUpdate{
			Expiration: &genai.ExpireTimeOrTTL{
				ExpireTime: expireTime,
			},
		})
	}
	return err
}

// SetCachedContextTTL sets the TTL of a cached context.
//
// (default: 1 hour)
func (c *Client) SetCachedContextTTL(ctx context.Context, cachedContextName string, ttl time.Duration) (err error) {
	var cc *genai.CachedContent
	if cc, err = c.client.GetCachedContent(ctx, cachedContextName); err == nil {
		_, err = c.client.UpdateCachedContent(ctx, cc, &genai.CachedContentToUpdate{
			Expiration: &genai.ExpireTimeOrTTL{
				TTL: ttl,
			},
		})
	}
	return err
}

// DeleteAllCachedContexts deletes all cached contexts.
func (c *Client) DeleteAllCachedContexts(ctx context.Context) (err error) {
	if c.Verbose {
		log.Printf("> deleting all cached contexts...")
	}

	iter := c.client.ListCachedContents(ctx)
	for {
		cachedContext, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to iterate cached contexts while deleting: %s", err)
		}

		if c.Verbose {
			fmt.Printf(".")
		}

		err = c.client.DeleteCachedContent(ctx, cachedContext.Name)
		if err != nil {
			return fmt.Errorf("failed to delete cached context: %s", err)
		}
	}

	return nil
}

// DeleteAllFiles deletes all uploaded files.
func (c *Client) DeleteAllFiles(ctx context.Context) (err error) {
	if c.Verbose {
		log.Printf("> deleting all uploaded files...")
	}

	iter := c.client.ListFiles(ctx)
	for {
		file, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to iterate files while deleting: %s", err)
		}

		if c.Verbose {
			fmt.Printf(".")
		}

		err = c.client.DeleteFile(ctx, file.Name)
		if err != nil {
			return fmt.Errorf("failed to delete file: %s", err)
		}
	}

	return nil
}
