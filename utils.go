package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
)

const (
	uploadedFileStateCheckIntervalMilliseconds = 300 // 300 milliseconds
)

// FuncArg searches for and returns a value with given `key` from the function call arguments `from`.
func FuncArg[T any](from map[string]any, key string) (*T, error) {
	if v, exists := from[key]; exists {
		if cast, ok := v.(T); ok {
			return &cast, nil
		}
		return nil, fmt.Errorf("could not cast %[2]T '%[1]s' (%[2]v) to %[3]T", key, v, *new(T))
	}
	return nil, nil // not found
}

// upload files for prompt and wait for them to be ready
func uploadFilesAndWait(ctx context.Context, client *genai.Client, files []io.Reader) (uploaded []genai.FileData, err error) {
	uploaded = []genai.FileData{}
	fileNames := []string{}

	for i, file := range files {
		if mimeType, recycledInput, err := readMimeAndRecycle(file); err == nil {
			if matchedMimeType, supported := checkMimeType(mimeType); supported {
				if file, err := client.UploadFile(ctx, "", recycledInput, &genai.UploadFileOptions{
					MIMEType: matchedMimeType,
				}); err == nil {
					uploaded = append(uploaded, genai.FileData{
						MIMEType: file.MIMEType,
						URI:      file.URI,
					})

					fileNames = append(fileNames, file.Name)
				} else {
					return nil, fmt.Errorf("failed to upload file[%d] for prompt: %s", i, err)
				}
			} else {
				return nil, fmt.Errorf("MIME type of file[%d] not supported: %s", i, mimeType.String())
			}
		} else {
			return nil, fmt.Errorf("failed to detect MIME type of file[%d]: %s", i, err)
		}
	}

	// NOTE: wait for all the uploaded files to be ready
	waitForFiles(ctx, client, fileNames)

	return uploaded, nil
}

// UploadFilesAndWait uploads files and wait for them to be ready.
func (c *Client) UploadFilesAndWait(ctx context.Context, files []io.Reader) (uploaded []genai.FileData, err error) {
	// generate genai client
	var client *genai.Client
	client, err = genai.NewClient(ctx, option.WithAPIKey(c.apiKey))
	if err != nil {
		return nil, fmt.Errorf("failed to get client for upload: %s", err)
	}
	defer client.Close()

	return uploadFilesAndWait(ctx, client, files)
}

// get generative client and model
func (c *Client) getClientAndModel(ctx context.Context, opts *GenerationOptions) (client *genai.Client, model *genai.GenerativeModel, err error) {
	// client
	client, err = genai.NewClient(ctx, option.WithAPIKey(c.apiKey))
	if err != nil {
		return nil, nil, err
	}

	// model
	model = client.GenerativeModel(c.model)

	// system instruction
	if c.systemInstructionFunc != nil {
		model.SystemInstruction = &genai.Content{
			Role: "model",
			Parts: []genai.Part{
				genai.Text(c.systemInstructionFunc()),
			},
		}
	}

	// tool configs
	if opts != nil {
		if len(opts.Tools) > 0 {
			model.Tools = opts.Tools
		}
		if opts.ToolConfig != nil {
			model.ToolConfig = opts.ToolConfig
		}
	}

	// generation config
	if opts != nil && opts.Config != nil {
		model.GenerationConfig = *opts.Config
	}

	// safety settings for all categories (default: block only high)
	if opts != nil && opts.HarmBlockThreshold != nil {
		model.SafetySettings = safetySettings(*opts.HarmBlockThreshold)
	} else {
		model.SafetySettings = safetySettings(genai.HarmBlockOnlyHigh)
	}

	return
}

// build prompt parts for prompting
func (c *Client) buildPromptParts(ctx context.Context, client *genai.Client, promptText string, promptFiles []io.Reader) (parts []genai.Part, err error) {
	// text prompt
	parts = []genai.Part{
		genai.Text(promptText),
	}

	// files
	if uploaded, err := uploadFilesAndWait(ctx, client, promptFiles); err == nil {
		for _, part := range uploaded {
			parts = append(parts, part)
		}
	} else {
		return nil, fmt.Errorf("failed to upload files for prompt: %s", err)
	}

	return parts, nil
}

// generate safety settings for all supported harm categories
func safetySettings(threshold genai.HarmBlockThreshold) (settings []*genai.SafetySetting) {
	for _, category := range []genai.HarmCategory{
		/*
			// categories for PaLM 2 (Legacy) models
			genai.HarmCategoryUnspecified,
			genai.HarmCategoryDerogatory,
			genai.HarmCategoryToxicity,
			genai.HarmCategoryViolence,
			genai.HarmCategorySexual,
			genai.HarmCategoryMedical,
			genai.HarmCategoryDangerous,
		*/

		// all categories supported by Gemini models
		genai.HarmCategoryHarassment,
		genai.HarmCategoryHateSpeech,
		genai.HarmCategorySexuallyExplicit,
		genai.HarmCategoryDangerousContent,
	} {
		settings = append(settings, &genai.SafetySetting{
			Category:  category,
			Threshold: threshold,
		})
	}

	return settings
}

// check if given file's mime type is matched and supported
//
// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#supported_file_formats
func checkMimeType(mimeType *mimetype.MIME) (matched string, supported bool) {
	return func(mimeType *mimetype.MIME) (matchedMimeType string, supportedMimeType bool) {
		matchedMimeType = mimeType.String() // fallback

		switch {
		case slices.ContainsFunc([]string{
			// images
			//
			// https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-image
			"image/png",
			"image/jpeg",
			"image/webp",
			"image/heic",
			"image/heif",

			// audios
			//
			// https://ai.google.dev/gemini-api/docs/audio?lang=go#supported-formats
			"audio/wav",
			"audio/mp3",
			"audio/aiff",
			"audio/aac",
			"audio/ogg",
			"audio/flac",

			// videos
			//
			// https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-video
			"video/mp4",
			"video/mpeg",
			"video/mov",
			"video/avi",
			"video/x-flv",
			"video/mpg",
			"video/webm",
			"video/wmv",
			"video/3gpp",

			// document formats
			//
			// https://ai.google.dev/gemini-api/docs/document-processing?lang=go#technical-details
			"application/pdf",
			"application/x-javascript", "text/javascript",
			"application/x-python", "text/x-python",
			"text/plain",
			"text/html",
			"text/css",
			"text/md",
			"text/csv",
			"text/xml",
			"text/rtf",
		}, func(element string) bool {
			if mimeType.Is(element) { // supported,
				matchedMimeType = element
				return true
			}
			return false // matched but not supported,
		}): // matched,
			return matchedMimeType, true
		default: // not matched, or not supported
			return matchedMimeType, false
		}
	}(mimeType)
}

// wait for all given uploaded files to be active.
func waitForFiles(ctx context.Context, client *genai.Client, fileNames []string) {
	var wg sync.WaitGroup
	for _, fileName := range fileNames {
		wg.Add(1)

		go func(name string) {
			for {
				if file, err := client.GetFile(ctx, name); err == nil {
					if file.State == genai.FileStateActive {
						wg.Done()
						break
					} else {
						time.Sleep(uploadedFileStateCheckIntervalMilliseconds * time.Millisecond)
					}
				} else {
					time.Sleep(uploadedFileStateCheckIntervalMilliseconds * time.Millisecond)
				}
			}
		}(fileName)
	}
	wg.Wait()
}

// prettify given thing in JSON format
func prettify(v any) string {
	if bytes, err := json.MarshalIndent(v, "", "  "); err == nil {
		return string(bytes)
	}
	return fmt.Sprintf("%+v", v)
}

// convert error (possibly goolge api error) to string
func errorString(err error) (error string) {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		return fmt.Sprintf("googleapi error: %s", gerr.Body)
	} else {
		return err.Error()
	}
}

// read mime type of given input
//
// https://pkg.go.dev/github.com/gabriel-vasile/mimetype#example-package-DetectReader
func readMimeAndRecycle(input io.Reader) (mimeType *mimetype.MIME, recycled io.Reader, err error) {
	// header will store the bytes mimetype uses for detection.
	header := bytes.NewBuffer(nil)

	// After DetectReader, the data read from input is copied into header.
	mtype, err := mimetype.DetectReader(io.TeeReader(input, header))
	if err != nil {
		return
	}

	// Concatenate back the header to the rest of the file.
	// recycled now contains the complete, original data.
	recycled = io.MultiReader(header, input)

	return mtype, recycled, err
}
