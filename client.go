// client.go

package gt

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"log"
	"time"

	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
	"google.golang.org/genai"

	old "github.com/google/generative-ai-go/genai" // FIXME: remove this after all APIs are implemented
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

	maxRetryCount uint = 3 // NOTE: will retry on 5xx errors
)

const (
	RoleUser  string = "user"
	RoleModel string = "model"
)

// Client struct
type Client struct {
	apiKey string
	client *genai.Client

	model                 string
	systemInstructionFunc FnSystemInstruction
	fileConvertFuncs      map[string]FnConvertBytes

	timeoutSeconds int

	DeleteFilesOnClose  bool
	DeleteCachesOnClose bool
	Verbose             bool

	oldClient *old.Client // FIXME: remove this after file APIs are implemented
}

// NewClient returns a new client with given values.
func NewClient(apiKey, model string) (*Client, error) {
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

	// FIXME: remove these lines after file APIs are implemented
	var oldClient *old.Client
	oldClient, err = old.NewClient(context.TODO(), option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to create old genai client: %w", err)
	}

	return &Client{
		apiKey: apiKey,
		client: client,

		model: model,
		systemInstructionFunc: func() string {
			return defaultSystemInstruction
		},
		fileConvertFuncs: make(map[string]FnConvertBytes),

		timeoutSeconds: defaultTimeoutSeconds,

		DeleteFilesOnClose:  false,
		DeleteCachesOnClose: false,
		Verbose:             false,

		oldClient: oldClient, // FIXME: remove this line after file APIs are implemented
	}, nil
}

// Close closes the client.
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
			if c.Verbose {
				log.Printf("> failed to delete all files before close: %s", err)
			}

			errs = append(errs, err)
		}
	}

	// delete all caches before close
	if c.DeleteCachesOnClose {
		if c.Verbose {
			log.Printf("> deleting all caches before close...")
		}

		if err := c.DeleteAllCachedContexts(ctxx); err != nil {
			if c.Verbose {
				log.Printf("> failed to delete all caches before close: %s", err)
			}

			errs = append(errs, err)
		}
	}

	// FIXME: remove these lines after file APIs are implemented
	if err := c.oldClient.Close(); err != nil {
		if c.Verbose {
			log.Printf("> failed to close old client: %s", err)
		}

		errs = append(errs, err)
	}

	return errors.Join(errs...)
}

// SetSystemInstructionFunc sets the system instruction function.
//
// NOTE: if a model or function does not support system instruction, set `nil` with this function.
func (c *Client) SetSystemInstructionFunc(fn FnSystemInstruction) {
	c.systemInstructionFunc = fn
}

// SetFileConverter sets the file converter function which converts given bytes to another kind of bytes before uploading.
//
// NOTE: given function will be called only when the mime type is not supported defaultly (see function: `SupportedMimeType`)
func (c *Client) SetFileConverter(mimeType string, fn FnConvertBytes) {
	c.fileConvertFuncs[mimeType] = fn
}

// SetTimeout sets the timeout in seconds.
func (c *Client) SetTimeout(seconds int) {
	c.timeoutSeconds = seconds
}

// ResponseModality constants
const (
	ResponseModalityText  = "Text"
	ResponseModalityImage = "Image"
)

// generate stream iterator with given values
func (c *Client) generateStream(
	ctx context.Context,
	prompts []Prompt,
	options ...*GenerationOptions,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	// generation options
	var opts *GenerationOptions = nil
	var history []genai.Content = nil
	if len(options) > 0 {
		opts = options[0]
		history = opts.History
	}

	if c.Verbose {
		log.Printf("> generating streamed content with prompts: %v (options: %s)", prompts, prettify(opts))
	}

	var err error
	yieldErrorAndEndIterator := func(err error) iter.Seq2[*genai.GenerateContentResponse, error] {
		return func(yield func(*genai.GenerateContentResponse, error) bool) {
			if !yield(nil, err) {
				return
			}
		}
	}

	// generate parts for prompting
	var contents []*genai.Content
	contents, err = c.buildPromptContents(ctx, prompts, history)
	if err != nil {
		return yieldErrorAndEndIterator(fmt.Errorf("failed to build prompts: %w", err))
	}

	// stream
	return c.client.Models.GenerateContentStream(
		ctx,
		c.model,
		contents,
		c.generateContentConfig(opts),
	)
}

// GenerateStreamIterated generates stream iterator with given values.
//
// It does not timeout itself, so set timeout with `ctx` when needed.
func (c *Client) GenerateStreamIterated(
	ctx context.Context,
	prompts []Prompt,
	options ...*GenerationOptions,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	return c.generateStream(ctx, prompts, options...)
}

// GenerateStreamed generates with given values synchronously and calls back `fnStreamCallback`.
//
// It times out in `timeoutSeconds` seconds.
func (c *Client) GenerateStreamed(
	ctx context.Context,
	prompts []Prompt,
	fnStreamCallback FnStreamCallback,
	options ...*GenerationOptions,
) (err error) {
	// set timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
	defer cancel()

	// number of tokens
	var numTokensInput int32 = 0
	var numTokensOutput int32 = 0
	var numTokensCached int32 = 0

	var it *genai.GenerateContentResponse
	for it, err = range c.generateStream(ctx, prompts, options...) {
		if err != nil {
			err = fmt.Errorf("failed to iterate stream: %w", err)
			break
		}

		if c.Verbose {
			log.Printf("> iterating stream response: %s", prettify(it))
		}

		var candidate *genai.Candidate
		var content *genai.Content
		var parts []*genai.Part

		if len(it.Candidates) > 0 {
			// update number of tokens
			if it.UsageMetadata != nil {
				if it.UsageMetadata.PromptTokenCount != nil && numTokensInput < *it.UsageMetadata.PromptTokenCount {
					numTokensInput = *it.UsageMetadata.PromptTokenCount
				}
				if it.UsageMetadata.CandidatesTokenCount != nil && numTokensOutput < *it.UsageMetadata.CandidatesTokenCount {
					numTokensOutput = *it.UsageMetadata.CandidatesTokenCount
				}
				if it.UsageMetadata.CachedContentTokenCount != nil && numTokensCached < *it.UsageMetadata.CachedContentTokenCount {
					numTokensCached = *it.UsageMetadata.CachedContentTokenCount
				}
			}

			candidate = it.Candidates[0]
			content = candidate.Content

			if content != nil && len(content.Parts) > 0 {
				parts = content.Parts
			} else if len(candidate.FinishReason) > 0 {
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
			if len(part.Text) > 0 { // (text)
				fnStreamCallback(StreamCallbackData{
					TextDelta: genai.Ptr(part.Text),
					Thought:   part.Thought,
				})
			} else if part.InlineData != nil { // (file: image, ...)
				fnStreamCallback(StreamCallbackData{
					InlineData: part.InlineData,
					Thought:    part.Thought,
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
				fnStreamCallback(StreamCallbackData{
					Error: fmt.Errorf("unsupported type of part for streaming: %s", prettify(part)),
				})
			}
		}

		// pass the number of tokens
		if numTokensInput > 0 {
			fnStreamCallback(StreamCallbackData{
				NumTokens: &NumTokens{
					Input:  numTokensInput,
					Output: numTokensOutput,
					Cached: numTokensCached,
				},
			})
		}
	}

	return err
}

// Generate generates with given values synchronously.
//
// It times out in `timeoutSeconds` seconds.
//
// It retries on 5xx errors for `maxRetryCount` times.
func (c *Client) Generate(
	ctx context.Context,
	prompts []Prompt,
	options ...*GenerationOptions,
) (res *genai.GenerateContentResponse, err error) {
	// set timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
	defer cancel()

	// generation options
	var opts *GenerationOptions = nil
	var history []genai.Content = nil
	if len(options) > 0 {
		opts = options[0]
		history = opts.History
	}

	if c.Verbose {
		log.Printf("> generating content with prompts: %v (options: %s)", prompts, prettify(opts))
	}

	// generate parts for prompting
	var contents []*genai.Content
	contents, err = c.buildPromptContents(ctx, prompts, history)
	if err != nil {
		return nil, fmt.Errorf("failed to build prompts: %w", err)
	}

	return c.generate(ctx, contents, maxRetryCount, opts)
}

// ImageGenerationOptions struct for generating images
type ImageGenerationOptions struct {
	NegativePrompt           string                    `json:"negativePrompt,omitempty"`
	NumberOfImages           int32                     `json:"numberOfImages,omitempty"`
	AspectRatio              string                    `json:"aspectRatio,omitempty"`
	GuidanceScale            *float32                  `json:"guidanceScale,omitempty"`
	Seed                     *int32                    `json:"seed,omitempty"`
	SafetyFilterLevel        genai.SafetyFilterLevel   `json:"safetyFilterLevel,omitempty"`
	PersonGeneration         genai.PersonGeneration    `json:"personGeneration,omitempty"`
	IncludeSafetyAttributes  bool                      `json:"includeSafetyAttributes,omitempty"`
	IncludeRAIReason         bool                      `json:"includeRaiReason,omitempty"`
	Language                 genai.ImagePromptLanguage `json:"language,omitempty"`
	OutputMIMEType           string                    `json:"outputMimeType,omitempty"`
	OutputCompressionQuality *int32                    `json:"outputCompressionQuality,omitempty"`
	AddWatermark             bool                      `json:"addWatermark,omitempty"`
	EnhancePrompt            bool                      `json:"enhancePrompt,omitempty"`
}

// GenerateImages generates images with given prompt.
func (c *Client) GenerateImages(
	ctx context.Context,
	prompt string,
	options ...*ImageGenerationOptions,
) (res *genai.GenerateImagesResponse, err error) {
	// set timeout
	ctx, cancel := context.WithTimeout(ctx, time.Duration(c.timeoutSeconds)*time.Second)
	defer cancel()

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
		log.Printf("> generating images with prompt: '%s' (options: %s)", prompt, prettify(opts))
	}

	return c.client.Models.GenerateImages(
		ctx,
		c.model,
		prompt,
		config,
	)
}

// generate with retry count
func (c *Client) generate(
	ctx context.Context,
	parts []*genai.Content,
	remainingRetryCount uint,
	options ...*GenerationOptions,
) (res *genai.GenerateContentResponse, err error) {
	if c.Verbose && remainingRetryCount < maxRetryCount {
		log.Printf("> retrying generation with remaining retry count: %d", remainingRetryCount)
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
		var se *genai.ServerError
		if errors.As(err, &se) && se.Code >= 500 { // retry on server errors (5xx)
			if remainingRetryCount > 0 { // retriable,
				// then retry
				return c.generate(ctx, parts, remainingRetryCount-1)
			} else { // all retries failed,
				return nil, fmt.Errorf("all %d retries of generation failed with the latest error: %w", maxRetryCount, err)
			}
		}
	}
	return res, err
}

// generate config for content generation
func (c *Client) generateContentConfig(opts *GenerationOptions) (generated *genai.GenerateContentConfig) {
	generated = &genai.GenerateContentConfig{}

	if c.systemInstructionFunc != nil {
		generated.SystemInstruction = &genai.Content{
			Role: RoleModel,
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
		generated.ResponseModalities = opts.ResponseModalities
		generated.MediaResolution = opts.MediaResolution
		generated.SpeechConfig = opts.SpeechConfig
	}

	return generated
}

// CacheContext caches the context with given values and returns the name of the cached context.
//
// `tools`, `toolConfig`, and `cachedContextDisplayName` are optional.
func (c *Client) CacheContext(
	ctx context.Context,
	systemInstruction *string,
	prompts []Prompt,
	tools []*genai.Tool,
	toolConfig *genai.ToolConfig,
	cachedContextDisplayName *string,
) (cachedContextName string, err error) {
	if c.Verbose {
		log.Printf("> caching context with system prompt: %s, prompts: %v, tools: %s, and tool config: %s", prettify(systemInstruction), prompts, prettify(tools), prettify(toolConfig))
	}

	// context to cache
	argcc := &genai.CreateCachedContentConfig{}
	if cachedContextDisplayName != nil {
		argcc.DisplayName = *cachedContextDisplayName
	}

	// system instruction
	if systemInstruction != nil {
		argcc.SystemInstruction = &genai.Content{
			Role: RoleModel,
			Parts: []*genai.Part{
				genai.NewPartFromText(*systemInstruction),
			},
		}
	}

	// prompts
	argcc.Contents, err = c.buildPromptContents(ctx, prompts, nil)
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

// SetCachedContextExpireTime sets the expiration time of a cached context.
//
// (default: 1 hour later)
func (c *Client) SetCachedContextExpireTime(
	ctx context.Context,
	cachedContextName string,
	expireTime time.Time,
) (err error) {
	var cc *genai.CachedContent
	if cc, err = c.client.Caches.Get(ctx, cachedContextName, &genai.GetCachedContentConfig{}); err == nil {
		_, err = c.client.Caches.Update(ctx, cc.Name, &genai.UpdateCachedContentConfig{
			ExpireTime: expireTime,
		})
	}
	return err
}

// SetCachedContextTTL sets the TTL of a cached context.
//
// (default: 1 hour)
func (c *Client) SetCachedContextTTL(
	ctx context.Context,
	cachedContextName string,
	ttl string,
) (err error) {
	var cc *genai.CachedContent
	if cc, err = c.client.Caches.Get(ctx, cachedContextName, &genai.GetCachedContentConfig{}); err == nil {
		_, err = c.client.Caches.Update(ctx, cc.Name, &genai.UpdateCachedContentConfig{
			TTL: ttl,
		})
	}
	return err
}

// ListAllCachedContexts lists all cached contexts.
//
// `listed` is a map of cached context name and cached context.
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

// DeleteAllCachedContexts deletes all cached contexts.
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
			return err
		}
	}

	return nil
}

// DeleteCachedContext deletes a cached context.
func (c *Client) DeleteCachedContext(
	ctx context.Context,
	cachedContextName string,
) (err error) {
	if c.Verbose {
		log.Printf("> deleting cached context: %s...", cachedContextName)
	}

	if _, err = c.client.Caches.Delete(ctx, cachedContextName, &genai.DeleteCachedContentConfig{}); err != nil {
		return fmt.Errorf("failed to delete cached context: %w", err)
	}

	return nil
}

// DeleteAllFiles deletes all uploaded files.
//
// FIXME: fix this function after file APIs are implemented
func (c *Client) DeleteAllFiles(ctx context.Context) (err error) {
	if c.Verbose {
		log.Printf("> deleting all uploaded files...")
	}

	// FIXME: fix this line after file APIs are implemented
	iter := c.oldClient.ListFiles(ctx)
	for {
		file, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return fmt.Errorf("failed to iterate files while deleting: %w", err)
		}

		if c.Verbose {
			fmt.Printf(".")
		}

		// FIXME: fix this line after file APIs are implemented
		err = c.oldClient.DeleteFile(ctx, file.Name)
		if err != nil {
			return fmt.Errorf("failed to delete file: %w", err)
		}
	}

	return nil
}

// GenerateEmbeddings generates embeddings with given values.
//
// `title` can be empty.
//
// https://ai.google.dev/gemini-api/docs/embeddings
func (c *Client) GenerateEmbeddings(ctx context.Context, title string, contents []*genai.Content) (vectors [][]float32, err error) {
	if c.Verbose {
		log.Printf("> generating embeddings......")
	}

	conf := &genai.EmbedContentConfig{}
	if title != "" {
		conf.TaskType = "RETRIEVAL_DOCUMENT"
		conf.Title = title
	}

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
	return
}
