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

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
	"cloud.google.com/go/storage"
	"github.com/gabriel-vasile/mimetype"
	"github.com/google/uuid"
	"github.com/googleapis/gax-go/v2/apierror"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
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

	// default maximum retry count
	defaultMaxRetryCount uint = 3 // NOTE: will retry only on `5xx` errors
)

const (
	defaultBucketName                    = `gemini-things`
	defaultNumDaysUploadedFilesTTL int64 = 1
)

// Client provides methods for interacting with the Google Generative AI API.
// It encapsulates a genai.Client and adds higher-level functionalities.
type Client struct {
	client *genai.Client // Underlying Google Generative AI client.
	Type   genai.Backend

	// for Vertex AI
	projectID               string          // Google Cloud Project ID
	storage                 *storage.Client // Google Cloud Storage client
	bucketName              string          // Google Cloud Storage bucket name
	numDaysUploadedFilesTTL int64           // Google Cloud Storage objects' TTL (in days)

	model                 string                    // model to be used for generation.
	systemInstructionFunc FnSystemInstruction       // Function that returns the system instruction string.
	fileConvertFuncs      map[string]FnConvertBytes // Map of MIME types to custom file conversion functions.
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

// WithMaxRetryCount is a ClientOption that sets the default maximum retry count
// for retriable API errors (typically 5xx server errors).
func WithMaxRetryCount(count uint) ClientOption {
	return func(c *Client) {
		c.maxRetryCount = count
	}
}

// NewClient creates and returns a new Client instance which uses Gemini API.
//
// It requires an API key and accepts optional ClientOption functions to customize the client.
//
// Example:
//
//	client, err := gt.NewClient(
//		"YOUR_API_KEY",
//		gt.WithModel("gemini-2.5-flash"),
//		gt.WithMaxRetryCount(3),
//	)
//	if err != nil {
//		log.Fatal(err)
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
		client: client,
		Type:   genai.BackendGeminiAPI,
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

// NewVertexClient creates and returns a new Client instance which uses Vertex AI API.
//
// It requires a project ID, location, and credentials and accepts optional ClientOption functions to customize the client.
func NewVertexClient(
	ctx context.Context,
	credentialsJSON []byte,
	location string,
	opts ...ClientOption,
) (*Client, error) {
	var err error

	// credentials
	var creds *auth.Credentials
	if creds, err = credentials.NewCredentialsFromJSON(
		credentials.ServiceAccount,
		credentialsJSON,
		&credentials.DetectOptions{
			Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
		},
	); err == nil {
		var projectID string
		projectID, err = creds.ProjectID(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to get project ID: %w", err)
		}

		// genai client
		var client *genai.Client
		client, err = genai.NewClient(
			ctx,
			&genai.ClientConfig{
				Backend:     genai.BackendVertexAI,
				Project:     projectID,
				Location:    location,
				Credentials: creds,
			},
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create genai client: %w", err)
		}

		// google cloud storage client
		var sclient *storage.Client
		if sclient, err = storage.NewClient(
			ctx,
			option.WithAuthCredentials(creds),
		); err != nil {
			return nil, fmt.Errorf("failed to create google cloud storage client: %w", err)
		}

		c := &Client{
			client:                  client,
			Type:                    genai.BackendVertexAI,
			projectID:               projectID,
			storage:                 sclient,
			bucketName:              defaultBucketName,
			numDaysUploadedFilesTTL: defaultNumDaysUploadedFilesTTL,
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

	return nil, fmt.Errorf("failed to read credentials from JSON: %w", err)
}

// SetBucketName sets the name of the Google Cloud Storage bucket to use for
// (temporary) file uploads.
func (c *Client) SetBucketName(name string) {
	c.bucketName = name
}

// SetNumDaysUploadedFilesTTL sets the number of days for which uploaded files
// should be retained in the Google Cloud Storage bucket.
func (c *Client) SetNumDaysUploadedFilesTTL(days int64) {
	c.numDaysUploadedFilesTTL = days
}

// Storage returns the Google Cloud Storage client used by the client.
func (c *Client) Storage() *storage.Client {
	return c.storage
}

// CreateBucketForFileUploads creates a Google Cloud Storage bucket for file uploads.
func (c *Client) CreateBucketForFileUploads(ctx context.Context) error {
	if c.Type != genai.BackendVertexAI {
		return fmt.Errorf("`CreateBucketForFileUploads` is only for Vertex AI")
	}

	if err := c.storage.Bucket(c.bucketName).Create(ctx, c.projectID, &storage.BucketAttrs{
		PublicAccessPrevention: storage.PublicAccessPreventionEnforced,
		Lifecycle: storage.Lifecycle{
			Rules: []storage.LifecycleRule{
				{
					Action: storage.LifecycleAction{
						Type: storage.DeleteAction,
					},
					Condition: storage.LifecycleCondition{
						AgeInDays: c.numDaysUploadedFilesTTL,
					},
				},
			},
		},
	}); err != nil {
		var ae *apierror.APIError
		if ok := errors.As(err, &ae); ok {
			if ae.GRPCStatus().Code() != codes.AlreadyExists {
				return fmt.Errorf("failed to create bucket: %w", err)
			}
		}
		var e *googleapi.Error
		if ok := errors.As(err, &e); ok {
			if e.Code != 409 {
				return fmt.Errorf("failed to create bucket: %w", err)
			}
		}
	}

	return nil
}

// DeleteBucketForFileUploads deletes a Google Cloud Storage bucket which was created for file uploads.
func (c *Client) DeleteBucketForFileUploads(ctx context.Context) error {
	if c.Type != genai.BackendVertexAI {
		return fmt.Errorf("`DeleteBucketForFileUploads` is only for Vertex AI")
	}

	return c.storage.Bucket(c.bucketName).Delete(ctx)
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

// SetMaxRetryCount sets the default maximum number of retries for retriable API errors
// (typically 5xx server errors) encountered during operations like Generate.
func (c *Client) SetMaxRetryCount(count uint) {
	c.maxRetryCount = count
}

// generateStream is an internal helper to create a stream iterator for content generation.
func (c *Client) generateStream(
	ctx context.Context,
	contents []*genai.Content,
	options ...*genai.GenerateContentConfig,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	// generation options
	var opts *genai.GenerateContentConfig = nil
	if len(options) > 0 {
		opts = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating streamed with contents: %s (options: %s)",
			prettify(contents, true),
			prettify(opts, true),
		)
	}

	// stream
	return c.client.Models.GenerateContentStream(
		ctx,
		c.model,
		contents,
		c.alterGenerateContentConfig(opts),
	)
}

// GenerateStreamIterated returns an iterator (iter.Seq2) for streaming generated content.
// This method allows for processing parts of the response as they arrive.
//
// A `model` must be set in the Client (e.g., via WithModel or by setting c.model directly)
// before calling this function, otherwise an error will be yielded by the iterator.
//
// Example:
//
//	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
//	defer cancel()
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
	options ...*genai.GenerateContentConfig,
) iter.Seq2[*genai.GenerateContentResponse, error] {
	// check if model is set
	if c.model == "" {
		return yieldErrorAndEndIterator[genai.GenerateContentResponse](fmt.Errorf("model is not set for generating iterated stream"))
	}

	return c.generateStream(ctx, contents, options...)
}

// Generate performs a synchronous content generation request.
// It builds the prompt from the provided `prompts`, sends it to the API, and returns the full response.
//
// A `model` must be set in the Client before calling.
//
// It also implements a retry mechanism for 5xx server errors, configured by `c.maxRetryCount`
// (configurable via `WithMaxRetryCount` or `SetMaxRetryCount`).
func (c *Client) Generate(
	ctx context.Context,
	contents []*genai.Content,
	options ...*genai.GenerateContentConfig, // Optional genai.GenerateContentConfig to customize the request.
) (res *genai.GenerateContentResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generation")
	}

	// generation options
	var opts *genai.GenerateContentConfig = nil
	if len(options) > 0 {
		opts = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating with contents: %s (options: %s)",
			prettify(contents, true),
			prettify(opts, true),
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
	options ...*genai.GenerateContentConfig, // Optional genai.GenerateContentConfig to customize the request.
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

// GenerateImages generates images based on the provided text prompt and options.
//
// A `model` (specifically an image generation model) must be set in the Client before calling.
func (c *Client) GenerateImages(
	ctx context.Context,
	prompt string, // The text prompt describing the images to generate.
	options ...*genai.GenerateImagesConfig, // Optional genai.GenerateImagesConfig to customize the request.
) (res *genai.GenerateImagesResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generating images")
	}

	// generation options
	var config *genai.GenerateImagesConfig = nil
	if len(options) > 0 {
		config = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating images with prompt: '%s' (options: %s)",
			prompt,
			prettify(config),
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

// GenerateVideos generates videos based on the provided text prompt and options.
// It will wait for the operation to complete before returning the result.
//
// A `model` (specifically a video generation model) must be set in the Client before calling.
func (c *Client) GenerateVideos(
	ctx context.Context,
	prompt *string, // The optional text prompt describing the videos to generate.
	image *genai.Image, // The optional image prompt for the video.
	video *genai.Video, // The optional video prompt for the video.
	options ...*genai.GenerateVideosConfig, // Optional genai.GenerateVideosConfig to customize the request.
) (res *genai.GenerateVideosResponse, err error) {
	// check if model is set
	if c.model == "" {
		return nil, fmt.Errorf("model is not set for generating videos")
	}

	// check params
	if prompt == nil && image == nil && video == nil {
		return nil, fmt.Errorf("at least one of prompt, image, or video must be provided")
	}

	// generation options
	var config *genai.GenerateVideosConfig = nil
	if len(options) > 0 {
		config = options[0]
	}

	if c.Verbose {
		log.Printf(
			"> generating videos (options: %s)",
			prettify(config),
		)
	}

	source := genai.GenerateVideosSource{}
	if prompt != nil {
		source.Prompt = *prompt
	}
	if image != nil {
		source.Image = image
	}
	if video != nil {
		source.Video = video
	}

	var operation *genai.GenerateVideosOperation
	operation, err = c.client.Models.GenerateVideosFromSource(
		ctx,
		c.model,
		&source,
		config,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to generate videos: %w", err)
	}

	if c.Verbose {
		log.Printf("> waiting for videos generation to complete...")
	}

	var status *genai.GenerateVideosOperation
	for {
		if status, err = c.client.Operations.GetVideosOperation(ctx, operation, &genai.GetOperationConfig{}); err == nil {
			if c.Verbose {
				log.Printf("> videos operation status: %v (%s)", status.Done, prettify(status.Metadata))
			}

			if status.Done {
				return status.Response, nil
			}
			time.Sleep(generatingVideoFileStateCheckIntervalMilliseconds * time.Millisecond)
		} else {
			if c.Verbose {
				log.Printf("> failed to get videos operation status: %s", err)
			}

			return nil, fmt.Errorf("failed to get videos operation status: %w", err)
		}
	}
}

// generate with retry count
func (c *Client) generate(
	ctx context.Context,
	parts []*genai.Content,
	retryBudget uint,
	options ...*genai.GenerateContentConfig,
) (res *genai.GenerateContentResponse, err error) {
	if c.Verbose && retryBudget < c.maxRetryCount { // Compare with the original maxRetryCount from client config
		log.Printf(
			"> retrying generation with remaining retry budget: %d (initial: %d)",
			retryBudget,
			c.maxRetryCount,
		)
	}

	// generation options
	var opts *genai.GenerateContentConfig = nil
	if len(options) > 0 {
		opts = options[0]
	}

	res, err = c.client.Models.GenerateContent(
		ctx,
		c.model,
		parts,
		c.alterGenerateContentConfig(opts),
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

// alter generate content config for content generation
func (c *Client) alterGenerateContentConfig(opts *genai.GenerateContentConfig) (generated *genai.GenerateContentConfig) {
	if opts == nil {
		generated = &genai.GenerateContentConfig{}
	} else {
		generated = opts
	}

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
	if c.Type == genai.BackendVertexAI {
		return fmt.Errorf("`DeleteAllFiles` not implemented yet for Vertex AI")
	}

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
	switch c.Type {
	case genai.BackendGeminiAPI:
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
	case genai.BackendVertexAI:
		if c.storage != nil {
			bucket := c.storage.Bucket(c.bucketName)
			// mimetype
			var mimeType *mimetype.MIME
			var mimeTypeString, overridden string
			if len(overrideMimeType) > 0 {
				overridden = overrideMimeType[0]
			}
			if len(overridden) > 0 {
				mimeTypeString = overridden
			} else {
				if mimeType, file, err = readMimeAndRecycle(file); err != nil {
					return nil, fmt.Errorf("failed to read mimetype: %w", err)
				}
				mimeTypeString = mimeType.String()
			}

			filename := fmt.Sprintf("%s_%s", uuid.New().String(), fileDisplayName)

			// upload
			obj := bucket.Object(filename)
			w := obj.NewWriter(ctx)
			defer func() { _ = w.Close() }()
			if _, err := io.Copy(w, file); err != nil {
				return nil, fmt.Errorf("failed to upload file: %w", err)
			}
			return &genai.File{
				DisplayName: fileDisplayName,
				URI:         fmt.Sprintf("gs://%s/%s", c.bucketName, filename),
				MIMEType:    mimeTypeString,
			}, nil
		} else {
			return nil, fmt.Errorf("storage client is not initialized")
		}
	}

	return nil, fmt.Errorf("unsupported backend type: %s", c.Type)
}

// CreateFileSearchStore creates a new file search store.
func (c *Client) CreateFileSearchStore(
	ctx context.Context,
	displayName string,
) (store *genai.FileSearchStore, err error) {
	if c.Type == genai.BackendVertexAI {
		return nil, fmt.Errorf("`CreateFileSearchStore` is not implemented yet for Vertex AI")
	}

	return c.client.FileSearchStores.Create(ctx, &genai.CreateFileSearchStoreConfig{
		DisplayName: displayName,
	})
}

// DeleteFileSearchStore deletes a file search store.
func (c *Client) DeleteFileSearchStore(
	ctx context.Context,
	fileSearchStoreName string,
) (err error) {
	if c.Type == genai.BackendVertexAI {
		return fmt.Errorf("`DeleteFileSearchStore` is not implemented yet for Vertex AI")
	}

	return c.client.FileSearchStores.Delete(ctx, fileSearchStoreName, &genai.DeleteFileSearchStoreConfig{
		Force: ptr(true),
	})
}

// ListFileSearchStores lists all file search stores.
func (c *Client) ListFileSearchStores(
	ctx context.Context,
) (stores iter.Seq2[*genai.FileSearchStore, error]) {
	if c.Type == genai.BackendVertexAI {
		return yieldErrorAndEndIterator[genai.FileSearchStore](fmt.Errorf("`ListFileSearchStores` is not implemented yet for Vertex AI"))
	}

	return c.client.FileSearchStores.All(ctx)
}

// GetFileSearchStore gets a file search store.
func (c *Client) GetFileSearchStore(
	ctx context.Context,
	fileSearchStoreName string,
) (store *genai.FileSearchStore, err error) {
	if c.Type == genai.BackendVertexAI {
		return nil, fmt.Errorf("`GetFileSearchStore` is not implemented yet for Vertex AI")
	}

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
	if c.Type == genai.BackendVertexAI {
		return nil, fmt.Errorf("`UploadFileForSearch` is not implemented yet for Vertex AI")
	}

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
	if c.Type == genai.BackendVertexAI {
		return nil, fmt.Errorf("`ImportFileForSearch` is not implemented yet for Vertex AI")
	}

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
	if c.Type == genai.BackendVertexAI {
		return yieldErrorAndEndIterator[genai.Document](fmt.Errorf("`ListFilesInFileSearchStore` is not implemented yet for Vertex AI"))
	}

	return c.client.FileSearchStores.Documents.All(ctx, fileSearchStoreName)
}

// DeleteFileInFileSearchStore deletes a file in a file search store.
func (c *Client) DeleteFileInFileSearchStore(
	ctx context.Context,
	fileName string,
) error {
	if c.Type == genai.BackendVertexAI {
		return fmt.Errorf("`DeleteFileInFileSearchStore` is not implemented yet for Vertex AI")
	}

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
	if c.Type == genai.BackendVertexAI {
		return nil, fmt.Errorf("`RequestBatchEmbeddings` is not implemented yet for Vertex AI")
	}

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
