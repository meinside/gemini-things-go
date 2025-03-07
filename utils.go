package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
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
//
// `files` is a map of keys: display name, and values: io.Reader.
func (c *Client) UploadFilesAndWait(ctx context.Context, files map[string]io.Reader) (uploaded []genai.FileData, err error) {
	uploaded = []genai.FileData{}
	fileNames := []string{}

	i := 0
	for displayName, file := range files {
		if mimeType, recycledInput, err := readMimeAndRecycle(file); err == nil {
			if matchedMimeType, supported := checkMimeType(mimeType); supported {
				if file, err := c.client.UploadFile(ctx, "", recycledInput, &genai.UploadFileOptions{
					MIMEType:    matchedMimeType,
					DisplayName: displayName,
				}); err == nil {
					uploaded = append(uploaded, genai.FileData{
						MIMEType: file.MIMEType,
						URI:      file.URI,
					})

					fileNames = append(fileNames, file.Name)
				} else {
					return nil, fmt.Errorf("failed to upload file[%d] (%s) for prompt: %w", i, displayName, err)
				}
			} else {
				return nil, fmt.Errorf("MIME type of file[%d] (%s) not supported: %s", i, displayName, mimeType.String())
			}
		} else {
			return nil, fmt.Errorf("failed to detect MIME type of file[%d] (%s): %w", i, displayName, err)
		}

		i++
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
func (c *Client) buildPromptParts(ctx context.Context, promptText *string, promptFiles map[string]io.Reader) (parts []genai.Part, err error) {
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

// SupportedMimeType detects and returns the matched mime type of given bytes data and whether it's supported or not.
func SupportedMimeType(data []byte) (matchedMimeType string, supported bool, err error) {
	var mimeType *mimetype.MIME
	if mimeType, err = mimetype.DetectReader(bytes.NewReader(data)); err == nil {
		matchedMimeType, supported = checkMimeType(mimeType)

		return matchedMimeType, supported, nil
	}

	return http.DetectContentType(data), false, err
}

// SupportedMimeTypePath detects and returns the matched mime type of given path and whether it's supported or not.
func SupportedMimeTypePath(filepath string) (matchedMimeType string, supported bool, err error) {
	var f *os.File
	if f, err = os.Open(filepath); err == nil {
		defer f.Close()

		var mimeType *mimetype.MIME
		if mimeType, err = mimetype.DetectReader(f); err == nil {
			matchedMimeType, supported = checkMimeType(mimeType)

			return matchedMimeType, supported, nil
		}
	}

	return "", false, err
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

// return pointer to the given value
func ptr[T any](v T) *T {
	return &v
}

// ErrToStr converts error (possibly goolge api error) to string.
func ErrToStr(err error) (str string) {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		msg := gerr.Body
		if len(msg) <= 0 {
			msg = gerr.Message
		}
		if len(msg) <= 0 {
			msg = fmt.Sprintf("HTTP %d", gerr.Code)
		}

		return fmt.Sprintf("googleapi error: %s", msg)
	} else {
		return err.Error()
	}
}

// IsQuotaExceeded returns if given error is from execeeded API quota.
func IsQuotaExceeded(err error) bool {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		if gerr.Code == 429 {
			return true
		}
	}
	return false
}

// IsModelOverloaded returns if given error is from overloaded model.
func IsModelOverloaded(err error) bool {
	var gerr *googleapi.Error
	if errors.As(err, &gerr) {
		if gerr.Code == 503 && gerr.Message == `The model is overloaded. Please try again later.` {
			return true
		}
	}
	return false
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

// eg.
//
//	1,048,576 input tokens for gemini-2.0-flash
//	2,048 input tokens for text-embedding-004
const (
	defaultChunkedTextLengthInBytes    uint = 1024 * 1024 * 2
	defaultOverlappedTextLengthInBytes uint = defaultChunkedTextLengthInBytes / 100
)

// TextChunkOption contains options for chunking text.
type TextChunkOption struct {
	ChunkSize                uint
	OverlappedSize           uint
	KeepBrokenUTF8Characters bool
	EllipsesText             string
}

// ChunkedText contains the original text and the chunks.
type ChunkedText struct {
	Original string
	Chunks   []string
}

// ChunkText splits the given text into chunks of the specified size.
func ChunkText(text string, opts ...TextChunkOption) (ChunkedText, error) {
	opt := TextChunkOption{
		ChunkSize:      defaultChunkedTextLengthInBytes,
		OverlappedSize: defaultOverlappedTextLengthInBytes,
	}
	if len(opts) > 0 {
		opt = opts[0]
	}

	chunkSize := opt.ChunkSize
	overlappedSize := opt.OverlappedSize
	keepBrokenUTF8Chars := opt.KeepBrokenUTF8Characters
	ellipses := opt.EllipsesText

	// check `opt`
	if overlappedSize >= chunkSize {
		return ChunkedText{}, fmt.Errorf("overlapped size(= %d) must be less than chunk size(= %d)", overlappedSize, chunkSize)
	}

	var chunk string
	var chunks []string
	for start := 0; start < len(text); start += int(chunkSize) {
		end := start + int(chunkSize)
		if end > len(text) {
			end = len(text)
		}

		// cut text
		offset := start
		if offset > int(overlappedSize) {
			offset -= int(overlappedSize)
		}
		if keepBrokenUTF8Chars {
			chunk = text[offset:end]
		} else {
			chunk = strings.ToValidUTF8(text[offset:end], "")
		}

		// append ellipses
		if start > 0 {
			chunk = ellipses + chunk
		}
		if end < len(text) {
			chunk = chunk + ellipses
		}

		chunks = append(chunks, chunk)
	}

	return ChunkedText{
		Original: text,
		Chunks:   chunks,
	}, nil
}
