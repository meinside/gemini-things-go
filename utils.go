package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"sync"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/googleapi"
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

// UploadFilesAndWait uploads files and wait for them to be ready.
func (c *Client) UploadFilesAndWait(ctx context.Context, files []io.Reader) (uploaded []genai.FileData, err error) {
	uploaded = []genai.FileData{}
	fileNames := []string{}

	for i, file := range files {
		if mimeType, recycledInput, err := readMimeAndRecycle(file); err == nil {
			if matchedMimeType, supported := checkMimeType(mimeType); supported {
				if file, err := c.client.UploadFile(ctx, "", recycledInput, &genai.UploadFileOptions{
					MIMEType: matchedMimeType,
				}); err == nil {
					uploaded = append(uploaded, genai.FileData{
						MIMEType: file.MIMEType,
						URI:      file.URI,
					})

					fileNames = append(fileNames, file.Name)
				} else {
					return nil, fmt.Errorf("failed to upload file[%d] for prompt: %w", i, err)
				}
			} else {
				return nil, fmt.Errorf("MIME type of file[%d] not supported: %s", i, mimeType.String())
			}
		} else {
			return nil, fmt.Errorf("failed to detect MIME type of file[%d]: %w", i, err)
		}
	}

	// NOTE: wait for all the uploaded files to be ready
	waitForFiles(ctx, c.client, fileNames)

	return uploaded, nil
}

// get generative model
func (c *Client) getModel(ctx context.Context, opts *GenerationOptions) (model *genai.GenerativeModel, err error) {
	// model
	if opts == nil || opts.CachedContextName == nil {
		model = c.client.GenerativeModel(c.model)

		// system instruction
		if c.systemInstructionFunc != nil {
			model.SystemInstruction = genai.NewUserContent(genai.Text(c.systemInstructionFunc()))
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
	} else {
		// NOTE: CachedContent can not be used with GenerateContent request setting system_instruction, tools or tool_config.
		if argcc, err := c.client.GetCachedContent(ctx, *opts.CachedContextName); err == nil {
			model = c.client.GenerativeModelFromCachedContent(argcc)
		} else {
			return nil, fmt.Errorf("failed to get cached content while generating model: %w", err)
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
func (c *Client) buildPromptParts(ctx context.Context, promptText *string, promptFiles []io.Reader) (parts []genai.Part, err error) {
	parts = []genai.Part{}

	// text prompt
	if promptText != nil {
		parts = append(parts, genai.Text(*promptText))
	}

	// files
	if len(promptFiles) > 0 {
		if uploaded, err := c.UploadFilesAndWait(ctx, promptFiles); err == nil {
			for _, part := range uploaded {
				parts = append(parts, part)
			}
		} else {
			return nil, fmt.Errorf("failed to upload files for prompt: %w", err)
		}
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

// SupportedMimeType detects and returns the matched mime type of given data and whether it's supported or not.
func SupportedMimeType(data []byte) (matchedMimeType string, supported bool) {
	if mimeType, err := mimetype.DetectReader(bytes.NewReader(data)); err == nil {
		return checkMimeType(mimeType)
	}
	return http.DetectContentType(data), false
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

// ErrToStr converts error (possibly goolge api error) to string.
func ErrToStr(err error) (str string) {
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
