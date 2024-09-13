package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strings"
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

// UploadFilesAndWait uploads files and wait for them to be ready
func UploadFilesAndWait(ctx context.Context, client *genai.Client, files []io.Reader) (uploaded []genai.FileData, err error) {
	uploaded = []genai.FileData{}
	fileNames := []string{}

	for i, file := range files {
		if mime, recycledInput, err := readMimeAndRecycle(file); err == nil {
			mimeType := stripCharsetFromMimeType(mime)

			if supportedFileMimeType(mimeType) {
				if file, err := client.UploadFile(ctx, "", recycledInput, &genai.UploadFileOptions{
					MIMEType: mimeType,
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
				return nil, fmt.Errorf("MIME type of file[%d] not supported: %s", i, mimeType)
			}
		} else {
			return nil, fmt.Errorf("failed to detect MIME type of file[%d]: %s", i, err)
		}
	}

	// NOTE: wait for all the uploaded files to be ready
	waitForFiles(ctx, client, fileNames)

	return uploaded, nil
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

	// safety filters (block only high)
	model.SafetySettings = safetySettings(genai.HarmBlockThreshold(genai.HarmBlockOnlyHigh))

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

	return
}

// build prompt parts for prompting
func (c *Client) buildPromptParts(ctx context.Context, client *genai.Client, promptText string, promptFiles []io.Reader) (parts []genai.Part, err error) {
	// text prompt
	parts = []genai.Part{
		genai.Text(promptText),
	}

	// files
	if uploaded, err := UploadFilesAndWait(ctx, client, promptFiles); err == nil {
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

// strip trailing charset text from given mime type
func stripCharsetFromMimeType(mimeType string) string {
	splitted := strings.Split(mimeType, ";")
	return splitted[0]
}

// check if given file's mime type is supported
//
// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#supported_file_formats
func supportedFileMimeType(mimeType string) bool {
	return func(mimeType string) bool {
		switch {
		// images
		//
		// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#image_formats
		case slices.Contains([]string{
			"image/png",
			"image/jpeg",
			"image/webp",
			"image/heic",
			"image/heif",
		}, mimeType):
			return true
		// audios
		//
		// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#audio_formats
		case slices.Contains([]string{
			"audio/wav",
			"audio/mp3",
			"audio/aiff",
			"audio/aac",
			"audio/ogg",
			"audio/flac",
		}, mimeType):
			return true
		// videos
		//
		// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#video_formats
		case slices.Contains([]string{
			"video/mp4",
			"video/mpeg",
			"video/mov",
			"video/avi",
			"video/x-flv",
			"video/mpg",
			"video/webm",
			"video/wmv",
			"video/3gpp",
		}, mimeType):
			return true
		// plain text formats
		//
		// https://ai.google.dev/gemini-api/docs/prompting_with_media?lang=go#plain_text_formats
		case slices.Contains([]string{
			"text/plain",
			"text/html",
			"text/css",
			"text/javascript",
			"application/x-javascript",
			"text/x-typescript",
			"application/x-typescript",
			"text/csv",
			"text/markdown",
			"text/x-python",
			"application/x-python-code",
			"application/json",
			"text/xml",
			"application/rtf",
			"text/rtf",

			// FIXME: not stated in the document yet
			"application/pdf",
		}, mimeType):
			return true
		default:
			return false
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
func readMimeAndRecycle(input io.Reader) (mimeType string, recycled io.Reader, err error) {
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

	return mtype.String(), recycled, err
}
