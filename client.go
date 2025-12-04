// client.go

// Package gt provides a client for interacting with the Google Generative AI API.
// It encapsulates a genai.Client and adds higher-level functionalities.
package gt

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"log"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"google.golang.org/genai"
)

const (
	// default system instruction
	defaultSystemInstruction = `You are a chat bot for helping the user.

Respond to the user according to the following principles:
- Do not repeat the user's request.
- Be as accurate as possible.
- Be as truthful as possible.
- Be as comprehensive and informative as possible.
`

	// default thinking budget
	defaultThingkingBudget int32 = 1024

	// default maximum retry count
	defaultMaxRetryCount uint = 3 // NOTE: will retry only on `5xx` errors
)

// Client provides methods for interacting with the Google Generative AI API.
// It encapsulates a genai.Client and adds higher-level functionalities.
type Client struct {
	apiKey string        // API key for authentication.
	client *genai.Client // Underlying Google Generative AI client.

	model                 string                    // model to be used for generation.
	systemInstructionFunc FnSystemInstruction       // Function that returns the system instruction string.
	fileConvertFuncs      map[string]FnConvertBytes // Map of MIME types to custom file conversion functions.
	timeoutSeconds        int                       // timeout in seconds for API calls like Generate, GenerateStreamed.
	maxRetryCount         uint                      // maximum retry count for retriable API errors (e.g., 5xx).

	DeleteFilesOnClose  bool // If true, automatically deletes all uploaded files when Close is called.
	DeleteCachesOnClose bool // If true, automatically deletes all cached contexts when Close is called.
	Verbose             bool // If true, enables verbose logging for debugging.
}

// ClientOption is a function type used to configure a new Client.
// It follows the functional options pattern.
type ClientOption func(*Client)

// WithModel is a ClientOption that sets the default model for the client.
// This model will be used for generation tasks unless overridden in specific function calls.
func WithModel(model string) ClientOption {
	return func(c *Client) {
		c.model = model
	}
}

// WithTimeoutSeconds is a ClientOption that sets the default timeout in seconds
// for API calls that support it (e.g., Generate, GenerateStreamed, GenerateImages).
func WithTimeoutSeconds(seconds int) ClientOption {
	return func(c *Client) {
		c.timeoutSeconds = seconds
	}
}

// WithMaxRetryCount is a ClientOption that sets the default maximum retry count
// for retriable API errors (typically 5xx server errors).
func WithMaxRetryCount(count uint) ClientOption {
	return func(c *Client) {
		c.maxRetryCount = count
	}
}

// NewClient creates and returns a new Client instance.
// It requires an API key and accepts optional ClientOption functions to customize the client.
//
// Example:
//
//	client, err := gt.NewClient("YOUR_API_KEY",
//	    gt.WithModel("gemini-2.0-flash"),
//	    gt.WithTimeoutSeconds(60),
//	    gt.WithMaxRetryCount(5),
//	)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer client.Close()
func NewClient(apiKey string, opts ...ClientOption) (*Client, error) {
	var err error

	// genai client
	var client *genai.Client
	client, err = genai.NewClient(context.TODO(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create genai client: %w", err)
	}

	c := &Client{
		apiKey: apiKey,
		client: client,
		systemInstructionFunc: func() string {
			return defaultSystemInstruction
		},
		fileConvertFuncs:    make(map[string]FnConvertBytes),
		maxRetryCount:       defaultMaxRetryCount,
		DeleteFilesOnClose:  false,
		DeleteCachesOnClose: false,
		Verbose:             false,
	}

	for _, opt := range opts {
		opt(c)
	}

	return c, nil
}

// Close releases resources associated with the client.
// It handles the deletion of uploaded files and cached contexts if configured to do so
// via `deleteFilesOnClose` and `deleteCachesOnClose` respectively.
// Any errors encountered during cleanup are collected and returned as a single joined error.
// An optional context can be provided for the cleanup operations.
func (c *Client) Close(ctx ...context.Context) error {
	errs := []error{}

	var ctxx context.Context
	if len(ctx) > 0 {
		ctxx = ctx[0]
	} else {
		ctxx = context.TODO()
	}

	// delete all files before close
	if c.DeleteFilesOnClose {
		if c.Verbose {
			log.Printf("> deleting all files before close...")
		}

		if err := c.DeleteAllFiles(ctxx); err != nil {
			errs = append(errs, err)
		}
	}

	// delete all caches before close
	if c.DeleteCachesOnClose {
		if c.Verbose {
			log.Printf("> deleting all caches before close...")
		}

		if err := c.DeleteAllCachedContexts(ctxx); err != nil {
			errs = append(errs, err)
		}
	}

	return errors.Join(errs...)
}

// SetSystemInstructionFunc sets a custom function that provides the system instruction string.
// This instruction is used by the model to guide its behavior.
// If a model or specific generation task does not support system instructions,
// or to remove a previously set instruction, pass `nil`.
func (c *Client) SetSystemInstructionFunc(fn FnSystemInstruction) {
	c.systemInstructionFunc = fn
}

// SetFileConverter registers a custom function to convert files of a specific MIME type
// before they are uploaded. This is useful for unsupported MIME types or for pre-processing.
// The provided function `fn` will be called if the MIME type matches `mimeType` and
// is not natively supported (see `SupportedMimeType`).
func (c *Client) SetFileConverter(mimeType string, fn FnConvertBytes) {
	c.fileConvertFuncs[mimeType] = fn
}

// SetTimeoutSeconds sets the default timeout in seconds for API calls like Generate,
// GenerateStreamed, and GenerateImages. This timeout is applied to the context
// used for these operations.
func (c *Client) SetTimeoutSeconds(seconds int) {
	c.timeoutSeconds = seconds
}

// SetMaxRetryCount sets the default maximum number of retries for retriable API errors
// (typically 5xx server errors) encountered during operations like Generate.
func (c *Client) SetMaxRetryCount(count uint) {
	c.maxRetryCount = count
}

// generateStream is an internal helper to create a stream iterator for content generation.
func (c *Client) generateStream(
	ctx context.Context,
	contents []*genai.Content,
	options ...*GenerationOptions,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating streamed with contents: %v (options: %s)",
			contents,
			prettify(opts),
		)
	}

	// stream
	return c.client.Models.GenerateContentStream(
		ctx,
		c.model,
		contents,
		c.generateContentConfig(opts),
	)
}

// GenerateStreamIterated returns an iterator (iter.Seq2) for streaming generated content.
// This method allows for processing parts of the response as they arrive.
//
// A `model` must be set in the Client (e.g., via WithModel or by setting c.model directly)
// before calling this function, otherwise an error will be yielded by the iterator.
//
// This function itself does not implement a timeout. If a timeout is required,
// it should be managed by passing a context with a deadline (`context.WithTimeout`).
//
// Example:
//
//	if contents, err := client.PromptsToContents(ctx, []gt.Prompt{gt.PromptFromText("Tell me a story.")}, nil); err == nil {
//		iter := client.GenerateStreamIterated(ctx, []gt.Prompt{gt.PromptFromText("Tell me a story.")})
//		for resp, err := range iter {
//			if err != nil { /* handle error */
//			}
//			// process response part
//		}
//	}
func (c *Client) GenerateStreamIterated(
	ctx context.Context,
	contents []*genai.Content,
	options ...*GenerationOptions,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	// check if model is set
	if c.model == "" {
		return yieldErrorAndEndIterator(fmt.Errorf("model is not set for generating iterated stream"))
	}

	return c.generateStream(ctx, contents, options...)
}

// GenerateStreamed generates content and streams the response through a callback function.
// It is a synchronous convenience wrapper around GenerateStreamIterated.
//
// A `model` must be set in the Client before calling.
//
// The function processes the stream and invokes `fnStreamCallback` for each piece of data received.
// It typically processes only the first candidate from the response.
//
// This method includes a timeout mechanism based on `c.timeoutSeconds`.
// (configurable via `WithTimeoutSeconds“ or `SetTimeoutSeconds`).
// If `c.timeoutSeconds` is bigger than 0, timeout will be applied to the passed context `ctx`.
//
// Note: For more granular control or to avoid potential hangs with malformed server responses,
// using GenerateStreamIterated directly is recommended.
func (c *Client) GenerateStreamed(
	ctx context.Context,
	contents []*genai.Content,
	fnStreamCallback FnStreamCallback, // fnStreamCallback is the function to call with each part of the streamed response.
	options ...*GenerationOptions,
) (err error) {
	// check if model is set
	if c.model == "" {
		return fmt.Errorf("model is not set for generating stream")
	}

	// set timeout if configured
	var cancel context.CancelFunc
	if c.timeoutSeconds > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
		defer cancel()
	}

	// number of tokens
	var numTokensCached int32 = 0
	var numTokensOutput int32 = 0
	var numTokensInput int32 = 0
	var numTokensThoughts int32 = 0
	var numTokensToolUse int32 = 0
	var numTokensTotal int32 = 0

	var it *genai.GenerateContentResponse
	for it, err = range c.generateStream(ctx, contents, options...) {
		if err != nil {
			err = fmt.Errorf("failed to iterate stream: %w", err)
			break
		}

		if c.Verbose {
			log.Printf("> iterating stream response: %s", prettify(it))
		}

		// update number of tokens
		if it.UsageMetadata != nil {
			if it.UsageMetadata.CachedContentTokenCount != 0 && numTokensCached < it.UsageMetadata.CachedContentTokenCount {
				numTokensCached = it.UsageMetadata.CachedContentTokenCount
			}
			if it.UsageMetadata.CandidatesTokenCount != 0 && numTokensOutput < it.UsageMetadata.CandidatesTokenCount {
				numTokensOutput = it.UsageMetadata.CandidatesTokenCount
			}
			if it.UsageMetadata.PromptTokenCount != 0 && numTokensInput < it.UsageMetadata.PromptTokenCount {
				numTokensInput = it.UsageMetadata.PromptTokenCount
			}
			if it.UsageMetadata.ThoughtsTokenCount != 0 && numTokensThoughts < it.UsageMetadata.ThoughtsTokenCount {
				numTokensThoughts = it.UsageMetadata.ThoughtsTokenCount
			}
			if it.UsageMetadata.ToolUsePromptTokenCount != 0 && numTokensToolUse < it.UsageMetadata.ToolUsePromptTokenCount {
				numTokensToolUse = it.UsageMetadata.ToolUsePromptTokenCount
			}
			if it.UsageMetadata.TotalTokenCount != 0 && numTokensTotal < it.UsageMetadata.TotalTokenCount {
				numTokensTotal = it.UsageMetadata.TotalTokenCount
			}
		}

		var candidate *genai.Candidate
		var content *genai.Content
		var parts []*genai.Part

		// FIXME: (is it OK?) take the first candidate,
		if len(it.Candidates) > 0 {
			candidate = it.Candidates[0]
			content = candidate.Content

			if content != nil && len(content.Parts) > 0 {
				parts = content.Parts
			} else if candidate.FinishReason == "" {
				fnStreamCallback(StreamCallbackData{
					Error: fmt.Errorf("no content in candidate: %s", prettify(candidate)),
				})
			}

			// if there is a finish reason,
			if candidate.FinishReason != "" {
				fnStreamCallback(StreamCallbackData{
					FinishReason: &candidate.FinishReason,
				})
			}
		}

		// iterate parts,
		for _, part := range parts {
			if len(part.Text) > 0 { // (text)
				fnStreamCallback(StreamCallbackData{
					TextDelta: genai.Ptr(part.Text),
				})
			} else if part.InlineData != nil { // (file: image, ...)
				fnStreamCallback(StreamCallbackData{
					InlineData: part.InlineData,
				})
			} else if part.FileData != nil { // URI based data
				fnStreamCallback(StreamCallbackData{
					FileData: part.FileData,
				})
			} else if part.Thought {
				fnStreamCallback(StreamCallbackData{
					Thought: part.Thought,
				})
			} else if part.VideoMetadata != nil { // (video metadata)
				fnStreamCallback(StreamCallbackData{
					VideoMetadata: part.VideoMetadata,
				})
			} else if part.FunctionCall != nil { // (function call)
				fnStreamCallback(StreamCallbackData{
					FunctionCall: part.FunctionCall,
				})
			} else if part.FunctionResponse != nil {
				fnStreamCallback(StreamCallbackData{
					FunctionResponse: part.FunctionResponse,
				})
			} else if part.ExecutableCode != nil { // (code execution: executable code)
				fnStreamCallback(StreamCallbackData{
					ExecutableCode: part.ExecutableCode,
				})
			} else if part.CodeExecutionResult != nil { // (code execution: result)
				fnStreamCallback(StreamCallbackData{
					CodeExecutionResult: part.CodeExecutionResult,
				})
			} else { // NOTE: TODO: add more conditions here
				// NOTE: unsupported types will reach here

				if (len(options) > 0 && options[0].IgnoreUnsupportedType) || candidate.FinishReason != "" {
					// ignore unsupported type
				} else {
					fnStreamCallback(StreamCallbackData{
						Error: fmt.Errorf("unsupported type of part for streaming: %s", prettify(part)),
					})
				}
			}
		}

		// pass the number of tokens
		if numTokensCached > 0 ||
			numTokensOutput > 0 ||
			numTokensInput > 0 ||
			numTokensThoughts > 0 ||
			numTokensToolUse > 0 ||
			numTokensTotal > 0 {
			fnStreamCallback(StreamCallbackData{
				NumTokens: &NumTokens{
					Cached:   numTokensCached,
					Output:   numTokensOutput,
					Input:    numTokensInput,
					Thoughts: numTokensThoughts,
					ToolUse:  numTokensToolUse,
					Total:    numTokensTotal,
				},
			})
		}
	}

	return err
}

// Generate performs a synchronous content generation request.
// It builds the prompt from the provided `prompts`, sends it to the API, and returns the full response.
//
// A `model` must be set in the Client before calling.
//
// The function includes a timeout mechanism based on `c.timeoutSeconds`
// (configurable via `WithTimeoutSeconds“ or `SetTimeoutSeconds`).
// If `c.timeoutSeconds` is bigger than 0, timeout will be applied to the passed context `ctx`.
//
// It also implements a retry mechanism for 5xx server errors, configured by `c.maxRetryCount`
// (configurable via `WithMaxRetryCount` or `SetMaxRetryCount`).
func (c *Client) Generate(
	ctx context.Context,
	contents []*genai.Content,
	options ...*GenerationOptions, // Optional GenerationOptions to customize the request.
) (res *genai.GenerateContentResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generation")
	}

	// set timeout if configured
	var cancel context.CancelFunc
	if c.timeoutSeconds > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
		defer cancel()
	}

	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating with contents: %v (options: %s)",
			contents,
			prettify(opts),
		)
	}

	return c.generate(ctx, contents, c.maxRetryCount, opts)
}

// FunctionCallHandler is a function type for handling function calls
type FunctionCallHandler func(args map[string]any) (string, error)

// GenerateWithRecursiveToolCalls generates recursively with resulting tool calls.
func (c *Client) GenerateWithRecursiveToolCalls(
	ctx context.Context,
	fnCallHandlers map[string]FunctionCallHandler,
	contents []*genai.Content,
	options ...*GenerationOptions, // Optional GenerationOptions to customize the request.
) (res *genai.GenerateContentResponse, err error) {
	res, err = c.Generate(ctx, contents, options...)
	if err == nil {
		// check if `res` has a function call,
		if fnCall, exists := hasFunctionCall(res.Candidates); exists {
			// if so, call the corresponding function,
			if handler, exists := fnCallHandlers[fnCall.Name]; exists {
				if handled, err := handler(fnCall.Args); err == nil {
					// append the result to `contents`
					contents = append(contents, &genai.Content{
						Role: genai.RoleUser,
						Parts: []*genai.Part{
							{
								FunctionResponse: &genai.FunctionResponse{
									Name: fnCall.Name,
									Response: map[string]any{
										"output": handled,
									},
								},
							},
						},
					})

					// and recurse
					return c.GenerateWithRecursiveToolCalls(
						ctx,
						fnCallHandlers,
						contents,
						options...,
					)
				} else {
					return nil, fmt.Errorf("failed to handle function call: %w", err)
				}
			} else {
				return nil, fmt.Errorf("no handler found for function call: %s", fnCall.Name)
			}
		}
	}
	return res, err
}

// ImageGenerationOptions defines parameters for image generation requests.
// These options correspond to the fields in `genai.GenerateImagesConfig`.
type ImageGenerationOptions struct {
	NegativePrompt           string                    `json:"negativePrompt,omitempty"`           // Specifies what not to include in the generated images.
	NumberOfImages           int32                     `json:"numberOfImages,omitempty"`           // The number of images to generate.
	AspectRatio              string                    `json:"aspectRatio,omitempty"`              // The desired aspect ratio for the generated images (e.g., "16:9", "1:1").
	GuidanceScale            *float32                  `json:"guidanceScale,omitempty"`            // Controls how closely the image generation follows the prompt.
	Seed                     *int32                    `json:"seed,omitempty"`                     // A seed for deterministic image generation.
	SafetyFilterLevel        genai.SafetyFilterLevel   `json:"safetyFilterLevel,omitempty"`        // The safety filtering level to apply.
	PersonGeneration         genai.PersonGeneration    `json:"personGeneration,omitempty"`         // Controls settings related to person generation.
	IncludeSafetyAttributes  bool                      `json:"includeSafetyAttributes,omitempty"`  // Whether to include safety attributes in the response.
	IncludeRAIReason         bool                      `json:"includeRaiReason,omitempty"`         // Whether to include RAI (Responsible AI) reasons in the response.
	Language                 genai.ImagePromptLanguage `json:"language,omitempty"`                 // The language of the prompt.
	OutputMIMEType           string                    `json:"outputMimeType,omitempty"`           // The desired MIME type for the output images.
	OutputCompressionQuality *int32                    `json:"outputCompressionQuality,omitempty"` // The compression quality for the output images.
	AddWatermark             bool                      `json:"addWatermark,omitempty"`             // Whether to add a watermark to the generated images.
	EnhancePrompt            bool                      `json:"enhancePrompt,omitempty"`            // Whether to enhance the prompt for better image generation.
}

// GenerateImages generates images based on the provided text prompt and options.
//
// A `model` (specifically an image generation model) must be set in the Client before calling.
//
// This method includes a timeout mechanism based on `c.timeoutSeconds`.
// (configurable via `WithTimeoutSeconds“ or `SetTimeoutSeconds`).
// If `c.timeoutSeconds` is bigger than 0, timeout will be applied to the passed context `ctx`.
func (c *Client) GenerateImages(
	ctx context.Context,
	prompt string, // The text prompt describing the images to generate.
	options ...*ImageGenerationOptions, // Optional ImageGenerationOptions to customize the request.
) (res *genai.GenerateImagesResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generating images")
	}

	// set timeout if configured
	var cancel context.CancelFunc
	if c.timeoutSeconds > 0 {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
		defer cancel()
	}

	// generation options
	var config *genai.GenerateImagesConfig = nil
	var opts *ImageGenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
		config = &genai.GenerateImagesConfig{
			NegativePrompt:           opts.NegativePrompt,
			NumberOfImages:           opts.NumberOfImages,
			AspectRatio:              opts.AspectRatio,
			GuidanceScale:            opts.GuidanceScale,
			Seed:                     opts.Seed,
			SafetyFilterLevel:        opts.SafetyFilterLevel,
			PersonGeneration:         opts.PersonGeneration,
			IncludeSafetyAttributes:  opts.IncludeRAIReason,
			IncludeRAIReason:         opts.IncludeRAIReason,
			Language:                 opts.Language,
			OutputMIMEType:           opts.OutputMIMEType,
			OutputCompressionQuality: opts.OutputCompressionQuality,
			AddWatermark:             opts.AddWatermark,
			EnhancePrompt:            opts.EnhancePrompt,
		}
	}

	if c.Verbose {
		log.Printf(
			"> generating images with prompt: '%s' (options: %s)",
			prompt,
			prettify(opts),
		)
	}

	res, err = c.client.Models.GenerateImages(
		ctx,
		c.model,
		prompt,
		config,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate images: %w", err)
	}
	return res, nil
}

// generate with retry count
func (c *Client) generate(
	ctx context.Context,
	parts []*genai.Content,
	retryBudget uint,
	options ...*GenerationOptions,
) (res *genai.GenerateContentResponse, err error) {
	if c.Verbose && retryBudget < c.maxRetryCount { // Compare with the original maxRetryCount from client config
		log.Printf(
			"> retrying generation with remaining retry budget: %d (initial: %d)",
			retryBudget,
			c.maxRetryCount,
		)
	}

	// generation options
	var opts *GenerationOptions = nil
	if len(options) > 0 {
		opts = options[0]
	}

	res, err = c.client.Models.GenerateContent(
		ctx,
		c.model,
		parts,
		c.generateContentConfig(opts),
	)
	if err != nil {
		retriable := false

		// retry on server errors (5xx)
		var se genai.APIError
		if errors.As(err, &se) && se.Code >= 500 ||
			regexpHTTP5xx.MatchString(err.Error()) {
			retriable = true
		}

		if retriable {
			if retryBudget > 0 { // retriable,
				// then retry with decremented budget
				return c.generate(ctx, parts, retryBudget-1)
			} else { // not retriable (all retries have failed),
				return nil, fmt.Errorf(
					"all %d retries of generation failed with the latest error: %w",
					c.maxRetryCount,
					err,
				)
			}
		}

		// Wrap non-retried errors
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	return res, nil
}

// generate config for content generation
func (c *Client) generateContentConfig(opts *GenerationOptions) (generated *genai.GenerateContentConfig) {
	generated = &genai.GenerateContentConfig{}

	if c.systemInstructionFunc != nil {
		generated.SystemInstruction = &genai.Content{
			Role: string(RoleModel),
			Parts: []*genai.Part{
				{
					Text: c.systemInstructionFunc(),
				},
			},
		}
	}

	if opts != nil {
		if opts.Config != nil {
			generated.Temperature = opts.Config.Temperature
			generated.TopP = opts.Config.TopP
			generated.TopK = opts.Config.TopK
			generated.CandidateCount = opts.Config.CandidateCount
			generated.MaxOutputTokens = opts.Config.MaxOutputTokens
			generated.StopSequences = opts.Config.StopSequences
			generated.ResponseLogprobs = opts.Config.ResponseLogprobs
			generated.Logprobs = opts.Config.Logprobs
			generated.PresencePenalty = opts.Config.PresencePenalty
			generated.FrequencyPenalty = opts.Config.FrequencyPenalty
			generated.Seed = opts.Config.Seed
			generated.ResponseMIMEType = opts.Config.ResponseMIMEType
			generated.ResponseSchema = opts.Config.ResponseSchema
			generated.RoutingConfig = opts.Config.RoutingConfig
		}
		generated.SafetySettings = safetySettings(opts.HarmBlockThreshold)
		generated.Tools = opts.Tools
		generated.ToolConfig = opts.ToolConfig
		generated.CachedContent = opts.CachedContent
		if opts.ResponseModalities != nil {
			generated.ResponseModalities = []string{}
			for _, modality := range opts.ResponseModalities {
				generated.ResponseModalities = append(generated.ResponseModalities, string(modality))
			}
		}
		generated.MediaResolution = opts.MediaResolution
		generated.SpeechConfig = opts.SpeechConfig
		if opts.ThinkingOn {
			thinkingBudget := defaultThingkingBudget
			if opts.ThinkingBudget > 0 {
				thinkingBudget = opts.ThinkingBudget
			}

			generated.ThinkingConfig = &genai.ThinkingConfig{
				IncludeThoughts: true,
				ThinkingBudget:  &thinkingBudget,
			}
		}
	}

	return generated
}

// CacheContext creates a cached content entry on the server.
// This can be used to reduce latency and token count for frequently used context.
//
// A `model` must be set in the Client before calling.
// The `systemInstruction`, `prompts`, `tools`, `toolConfig`, and `cachedContextDisplayName`
// are used to configure the cached content.
//
// Returns the server-assigned name of the cached content and any error encountered.
func (c *Client) CacheContext(
	ctx context.Context,
	systemInstruction *string, // Optional system instruction for the cached context.
	prompts []Prompt, // Prompts to include in the cached context.
	tools []*genai.Tool, // Optional tools to be available with the cached context.
	toolConfig *genai.ToolConfig, // Optional configuration for the tools.
	cachedContextDisplayName *string, // Optional display name for the cached content.
) (cachedContextName string, err error) {
	// check if model is set
	if c.model == "" {
		return "", fmt.Errorf("model is not set for caching context")
	}

	if c.Verbose {
		log.Printf(
			"> caching context with system prompt: %s, prompts: %v, tools: %s, and tool config: %s",
			prettify(systemInstruction),
			prompts,
			prettify(tools),
			prettify(toolConfig),
		)
	}

	// context to cache
	argcc := &genai.CreateCachedContentConfig{}
	if cachedContextDisplayName != nil {
		argcc.DisplayName = *cachedContextDisplayName
	}

	// system instruction
	if systemInstruction != nil {
		argcc.SystemInstruction = &genai.Content{
			Role: string(RoleModel),
			Parts: []*genai.Part{
				genai.NewPartFromText(*systemInstruction),
			},
		}
	}

	// prompts
	argcc.Contents, err = c.PromptsToContents(ctx, prompts, nil)
	if err != nil {
		return "", fmt.Errorf("failed to build prompts for caching context: %w", err)
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
	if cc, err = c.client.Caches.Create(ctx, c.model, argcc); err != nil {
		return "", fmt.Errorf("failed to cache context: %w", err)
	}

	return cc.Name, nil
}

// SetCachedContextExpireTime updates the expiration time of an existing cached content.
// The `model` does not need to be set in the client for this operation, as it uses the `cachedContextName`.
// The `expireTime` specifies the new absolute time at which the cache should expire.
func (c *Client) SetCachedContextExpireTime(
	ctx context.Context,
	cachedContextName string, // The name of the cached content to update.
	expireTime time.Time, // The new expiration time.
) (err error) {
	var cc *genai.CachedContent
	cc, err = c.client.Caches.Get(ctx, cachedContextName, &genai.GetCachedContentConfig{})
	if err != nil {
		return fmt.Errorf(
			"failed to get cached context %s: %w",
			cachedContextName,
			err,
		)
	}

	_, err = c.client.Caches.Update(ctx, cc.Name, &genai.UpdateCachedContentConfig{
		ExpireTime: expireTime,
	})
	if err != nil {
		return fmt.Errorf(
			"failed to update cached context %s: %w",
			cachedContextName,
			err,
		)
	}

	return nil
}

// SetCachedContextTTL updates the Time-To-Live (TTL) of an existing cached content.
// The `model` does not need to be set in the client for this operation.
// The `ttl` specifies the new duration for which the cache should live from the time of the update.
// A common default TTL (e.g., 1 hour) is often applied by the server if not specified.
func (c *Client) SetCachedContextTTL(
	ctx context.Context,
	cachedContextName string, // The name of the cached content to update.
	ttl time.Duration, // The new TTL duration.
) (err error) {
	var cc *genai.CachedContent
	cc, err = c.client.Caches.Get(ctx, cachedContextName, &genai.GetCachedContentConfig{})
	if err != nil {
		return fmt.Errorf(
			"failed to get cached context %s: %w",
			cachedContextName,
			err,
		)
	}

	_, err = c.client.Caches.Update(ctx, cc.Name, &genai.UpdateCachedContentConfig{
		TTL: ttl,
	})
	if err != nil {
		return fmt.Errorf(
			"failed to update cached context %s: %w",
			cachedContextName,
			err,
		)
	}

	return nil
}

// ListAllCachedContexts retrieves a list of all cached content entries available.
// The `model` does not need to be set in the client for this operation.
// It returns a map where keys are cached content names and values are the corresponding `*genai.CachedContent` objects.
func (c *Client) ListAllCachedContexts(ctx context.Context) (listed map[string]*genai.CachedContent, err error) {
	listed = make(map[string]*genai.CachedContent)

	if c.Verbose {
		log.Printf("> listing all cached contexts...")
	}

	for it, err := range c.client.Caches.All(ctx) {
		if err != nil {
			return nil, fmt.Errorf("failed to iterate cached contexts while listing: %w", err)
		}

		if c.Verbose {
			log.Printf("> iterating cached context: %s", prettify(it))
		}

		listed[it.Name] = it
	}

	return listed, nil
}

// DeleteAllCachedContexts iterates through all available cached content and deletes each one.
// The `model` does not need to be set in the client for this operation.
// Errors encountered during deletion of individual caches are aggregated.
// Consider the rate limits if deleting a very large number of caches.
func (c *Client) DeleteAllCachedContexts(ctx context.Context) (err error) {
	if c.Verbose {
		log.Printf("> deleting all cached contexts...")
	}

	for it, err := range c.client.Caches.All(ctx) {
		if err != nil {
			return fmt.Errorf("failed to iterate cached contexts while deleting: %w", err)
		}

		if c.Verbose {
			fmt.Printf(".")
		}

		if err = c.DeleteCachedContext(ctx, it.Name); err != nil {
			return fmt.Errorf(
				"failed to delete cached context %s during DeleteAllCachedContexts: %w",
				it.Name,
				err,
			)
		}
	}

	return nil
}

// DeleteCachedContext deletes a specific cached content entry by its name.
// The `model` does not need to be set in the client for this operation.
func (c *Client) DeleteCachedContext(
	ctx context.Context,
	cachedContextName string, // The name of the cached content to delete.
) (err error) {
	if c.Verbose {
		log.Printf("> deleting cached context: %s...", cachedContextName)
	}

	if _, err = c.client.Caches.Delete(
		ctx,
		cachedContextName,
		&genai.DeleteCachedContentConfig{},
	); err != nil {
		return fmt.Errorf("failed to delete cached context: %w", err)
	}

	return nil
}

// DeleteAllFiles iterates through all uploaded files associated with the API key and deletes them.
// The `model` does not need to be set in the client for this operation.
// Errors during deletion of individual files are aggregated.
// Consider rate limits if deleting a large number of files.
func (c *Client) DeleteAllFiles(ctx context.Context) (err error) {
	if c.Verbose {
		log.Printf("> deleting all uploaded files...")
	}

	for it, err := range c.client.Files.All(ctx) {
		if err != nil {
			return fmt.Errorf("failed to iterate files while deleting: %w", err)
		}

		if c.Verbose {
			fmt.Printf(".")
		}

		if _, err := c.client.Files.Delete(
			ctx,
			it.Name,
			&genai.DeleteFileConfig{},
		); err != nil {
			return fmt.Errorf("failed to delete file %s: %w", it.Name, err)
		}
	}

	return nil
}

// EmbeddingTaskType defines the specific task for which embeddings are being generated.
// This helps the model produce more relevant embeddings.
// See https://ai.google.dev/api/embeddings#v1beta.TaskType for details.
type EmbeddingTaskType string

// EmbeddingTaskType constants represent the various tasks for which embeddings can be optimized.
const (
	EmbeddingTaskUnspecified        EmbeddingTaskType = "TASK_TYPE_UNSPECIFIED" // Default, unspecified task type.
	EmbeddingTaskRetrievalQuery     EmbeddingTaskType = "RETRIEVAL_QUERY"       // Embeddings for a query to be used in retrieval.
	EmbeddingTaskRetrievalDocument  EmbeddingTaskType = "RETRIEVAL_DOCUMENT"    // Embeddings for a document to be indexed for retrieval.
	EmbeddingTaskSemanticSimilarity EmbeddingTaskType = "SEMANTIC_SIMILARITY"   // Embeddings for semantic similarity tasks.
	EmbeddingTaskClassification     EmbeddingTaskType = "CLASSIFICATION"        // Embeddings for classification tasks.
	EmbeddingTaskClustering         EmbeddingTaskType = "CLUSTERING"            // Embeddings for clustering tasks.
	EmbeddingTaskQuestionAnswering  EmbeddingTaskType = "QUESTION_ANSWERING"    // Embeddings for question answering.
	EmbeddingTaskFactVerification   EmbeddingTaskType = "FACT_VERIFICATION"     // Embeddings for fact verification.
	EmbeddingTaskCodeRetrievalQuery EmbeddingTaskType = "CODE_RETRIEVAL_QUERY"  // Embeddings for code retrieval query.
)

// GenerateEmbeddings creates vector embeddings for the given content.
//
// A `model` (specifically an embedding model) must be set in the Client.
// The `title` is optional but recommended if `taskType` is `EmbeddingTaskRetrievalDocument`.
// `contents` are the actual pieces of text/data to embed.
// `taskType` specifies the intended use of the embeddings, allowing the model to optimize them.
//
// Refer to https://ai.google.dev/gemini-api/docs/embeddings for more details.
func (c *Client) GenerateEmbeddings(
	ctx context.Context,
	title string, // Optional title for the content, relevant for RETRIEVAL_DOCUMENT task type.
	contents []*genai.Content, // The content to generate embeddings for.
	taskType *EmbeddingTaskType, // Optional task type to optimize embeddings.
) (vectors [][]float32, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generating embeddings")
	}

	if c.Verbose {
		log.Printf("> generating embeddings......")
	}

	// embeddings configuration
	conf := &genai.EmbedContentConfig{}

	// task type
	var selectedTaskType EmbeddingTaskType
	if taskType != nil {
		selectedTaskType = *taskType
	} /* else {
		// FIXME: 500 error when task is unspecified
		selectedTaskType = EmbeddingTaskUnspecified
	}*/

	// title
	if title != "" {
		conf.Title = title
		if selectedTaskType == "" || selectedTaskType == EmbeddingTaskUnspecified {
			selectedTaskType = EmbeddingTaskRetrievalDocument
		}
		if selectedTaskType != EmbeddingTaskRetrievalDocument {
			return nil, fmt.Errorf(
				"`title` is only applicable when `taskType` is '%s', but '%s' was given",
				EmbeddingTaskRetrievalDocument,
				selectedTaskType,
			)
		}
	}
	conf.TaskType = string(selectedTaskType)

	var res *genai.EmbedContentResponse
	if res, err = c.client.Models.EmbedContent(ctx, c.model, contents, conf); err == nil {
		if res != nil && res.Embeddings != nil {
			vectors = [][]float32{}
			for _, embedding := range res.Embeddings {
				vectors = append(vectors, embedding.Values)
			}
		}
	} else {
		err = fmt.Errorf("failed to generate embeddings: %w", err)
	}
	return vectors, err
}

// CountTokens calculates the number of tokens that the provided content would consume.
// This is useful for understanding and managing token usage.
//
// A `model` must be set in the Client.
//
// See https://ai.google.dev/gemini-api/docs/tokens?lang=go for more information.
func (c *Client) CountTokens(
	ctx context.Context,
	contents []*genai.Content, // The content for which to count tokens.
	config ...*genai.CountTokensConfig, // Optional configuration for token counting.
) (res *genai.CountTokensResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for counting tokens")
	}

	if c.Verbose {
		log.Printf("> counting tokens for contents: %s", prettify(contents))
	}

	var cfg *genai.CountTokensConfig = nil
	if len(config) > 0 {
		cfg = config[0]
	}

	res, err = c.client.Models.CountTokens(ctx, c.model, contents, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to count tokens: %w", err)
	}
	return res, nil
}

// ListModels retrieves a list of all generative models available to the authenticated API key.
// The `model` field in the Client does not need to be set for this operation.
// The function handles pagination automatically and returns a consolidated list of models.
func (c *Client) ListModels(ctx context.Context) (models []*genai.Model, err error) {
	if c.Verbose {
		log.Printf("> listing models...")
	}

	models = []*genai.Model{}

	var page genai.Page[genai.Model]
	pageToken := ""
	for {
		if page, err = c.client.Models.List(ctx, &genai.ListModelsConfig{
			PageToken: pageToken,
		}); err == nil {
			if c.Verbose {
				log.Printf("> fetched %d models", len(page.Items))
			}

			models = append(models, page.Items...)

			if page.NextPageToken == "" {
				break
			}
			pageToken = page.NextPageToken
		} else {
			if err == genai.ErrPageDone {
				err = nil
			} else {
				err = fmt.Errorf("failed to list models: %w", err)
			}
			break
		}
	}

	return models, err
}

// UploadFile uploads a file to the Gemini API.
//
// If `overrideMimeType` is not given, it will be inferred from the file bytes.
func (c *Client) UploadFile(
	ctx context.Context,
	file io.Reader,
	fileDisplayName string,
	overrideMimeType ...string,
) (uploaded *genai.File, err error) {
	config := &genai.UploadFileConfig{
		DisplayName: fileDisplayName,
	}

	var overridden string
	if len(overrideMimeType) > 0 {
		overridden = overrideMimeType[0]
	}
	if len(overridden) > 0 {
		config.MIMEType = overridden
	} else {
		var mime *mimetype.MIME
		if mime, err = mimetype.DetectReader(file); err == nil {
			if matched, supported := checkMimeTypeForFile(mime); supported {
				config.MIMEType = matched
			} else {
				return nil, fmt.Errorf("unsupported mime type for file: %s", mime.String())
			}
		} else {
			return nil, fmt.Errorf("failed to detect mimetype: %w", err)
		}
	}

	return c.client.Files.Upload(
		ctx,
		file,
		config,
	)
}

// CreateFileSearchStore creates a new file search store.
func (c *Client) CreateFileSearchStore(
	ctx context.Context,
	displayName string,
) (store *genai.FileSearchStore, err error) {
	return c.client.FileSearchStores.Create(ctx, &genai.CreateFileSearchStoreConfig{
		DisplayName: displayName,
	})
}

// DeleteFileSearchStore deletes a file search store.
func (c *Client) DeleteFileSearchStore(
	ctx context.Context,
	fileSearchStoreName string,
) (err error) {
	return c.client.FileSearchStores.Delete(ctx, fileSearchStoreName, &genai.DeleteFileSearchStoreConfig{
		Force: ptr(true),
	})
}

// ListFileSearchStores lists all file search stores.
func (c *Client) ListFileSearchStores(
	ctx context.Context,
) (stores iter.Seq2[*genai.FileSearchStore, error]) {
	return c.client.FileSearchStores.All(ctx)
}

// GetFileSearchStore gets a file search store.
func (c *Client) GetFileSearchStore(
	ctx context.Context,
	fileSearchStoreName string,
) (store *genai.FileSearchStore, err error) {
	return c.client.FileSearchStores.Get(ctx, fileSearchStoreName, &genai.GetFileSearchStoreConfig{})
}

// UploadFileForSearch creates a new file in a file search store.
//
// If `overrideMimeType` is not given, it will be inferred from the file bytes.
//
// Supported file formats are: https://ai.google.dev/gemini-api/docs/file-search#supported-files
func (c *Client) UploadFileForSearch(
	ctx context.Context,
	fileSearchStoreName string,
	file io.Reader,
	fileDisplayName string,
	metadata []*genai.CustomMetadata,
	chunkConfig *genai.ChunkingConfig,
	overrideMimeType ...string,
) (operation *genai.UploadToFileSearchStoreOperation, err error) {
	config := &genai.UploadToFileSearchStoreConfig{
		DisplayName: fileDisplayName,

		CustomMetadata: metadata,
		ChunkingConfig: chunkConfig,
	}

	var overridden string
	if len(overrideMimeType) > 0 {
		overridden = overrideMimeType[0]
	}
	if len(overridden) > 0 {
		config.MIMEType = overridden
	} else {
		var mime *mimetype.MIME
		if mime, err = mimetype.DetectReader(file); err == nil {
			if matched, supported := checkMimeTypeForFileSearch(mime); supported {
				config.MIMEType = matched
			} else {
				return nil, fmt.Errorf("unsupported mime type for file search: %s", mime.String())
			}
		} else {
			return nil, fmt.Errorf("failed to detect mimetype: %w", err)
		}
	}

	return c.client.FileSearchStores.UploadToFileSearchStore(
		ctx,
		file,
		fileSearchStoreName,
		config,
	)
}

// ImportFileForSearch imports an alread-uploaded file into a file search store.
//
// Supported file formats are: https://ai.google.dev/gemini-api/docs/file-search#supported-files
func (c *Client) ImportFileForSearch(
	ctx context.Context,
	fileSearchStoreName string,
	fileName string,
	metadata []*genai.CustomMetadata,
	chunkConfig *genai.ChunkingConfig,
) (operation *genai.ImportFileOperation, err error) {
	return c.client.FileSearchStores.ImportFile(
		ctx,
		fileSearchStoreName,
		fileName,
		&genai.ImportFileConfig{
			CustomMetadata: metadata,
			ChunkingConfig: chunkConfig,
		},
	)
}

// ListFilesInFileSearchStore lists all files in a file search store.
func (c *Client) ListFilesInFileSearchStore(
	ctx context.Context,
	fileSearchStoreName string,
) iter.Seq2[*genai.Document, error] {
	return c.client.FileSearchStores.Documents.All(ctx, fileSearchStoreName)
}

// DeleteFileInFileSearchStore deletes a file in a file search store.
func (c *Client) DeleteFileInFileSearchStore(
	ctx context.Context,
	fileName string,
) error {
	return c.client.FileSearchStores.Documents.Delete(
		ctx,
		fileName,
		&genai.DeleteDocumentConfig{
			Force: ptr(true),
		},
	)
}

// RequestBatch creates a batch job for the given job source.
//
// A `model` (specifically a batch model) must be set in the Client.
// `displayName` is an optional name to give the batch job.
func (c *Client) RequestBatch(
	ctx context.Context,
	job *genai.BatchJobSource,
	displayName string,
) (batch *genai.BatchJob, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for batch requests")
	}

	return c.client.Batches.Create(
		ctx,
		c.model,
		job,
		&genai.CreateBatchJobConfig{
			DisplayName: displayName,
		},
	)
}

// RequestBatchEmbeddings creates a batch job for the given embeddings job source.
//
// A `model` (specifically a batch model) must be set in the Client.
// `displayName` is an optional name to give the batch job.
func (c *Client) RequestBatchEmbeddings(
	ctx context.Context,
	job *genai.EmbeddingsBatchJobSource,
	displayName string,
) (batch *genai.BatchJob, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for batch embeddings requests")
	}

	return c.client.Batches.CreateEmbeddings(
		ctx,
		&c.model,
		job,
		&genai.CreateEmbeddingsBatchJobConfig{
			DisplayName: displayName,
		},
	)
}

// Batch returns the batch job with the given name.
func (c *Client) Batch(
	ctx context.Context,
	name string,
) (batch *genai.BatchJob, err error) {
	return c.client.Batches.Get(
		ctx,
		name,
		&genai.GetBatchJobConfig{},
	)
}

// ListBatches returns a list of all batch jobs available to the authenticated API key.
func (c *Client) ListBatches(ctx context.Context) iter.Seq2[*genai.BatchJob, error] {
	return c.client.Batches.All(ctx)
}

// CancelBatch cancels the batch job with the given name.
func (c *Client) CancelBatch(ctx context.Context, name string) (err error) {
	return c.client.Batches.Cancel(
		ctx,
		name, &genai.CancelBatchJobConfig{},
	)
}

// DeleteBatch deletes the batch job with the given name.
func (c *Client) DeleteBatch(ctx context.Context, name string) (err error) {
	var delete *genai.DeleteResourceJob
	delete, err = c.client.Batches.Delete(
		ctx,
		name,
		&genai.DeleteBatchJobConfig{},
	)
	if err != nil {
		return fmt.Errorf("failed to delete batch: %w", err)
	}
	if delete.Error != nil {
		return fmt.Errorf("error in delete resource job: %s", delete.Error.Message)
	}
	if delete.Done {
		return nil
	}
	return fmt.Errorf("failed to delete batch")
}
