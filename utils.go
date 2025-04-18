// utils.go

package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"google.golang.org/genai"
)

const (
	uploadedFileStateCheckIntervalMilliseconds = 300 // 300 milliseconds
)

// wait for all given uploaded files to be active.
func (c *Client) waitForFiles(ctx context.Context, fileNames []string) {
	var wg sync.WaitGroup
	for _, fileName := range fileNames {
		wg.Add(1)

		go func(name string) {
			for {
				if file, err := c.client.Files.Get(ctx, name, &genai.GetFileConfig{}); err == nil {
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

// UploadFilesAndWait uploads files and wait for them to be ready.
func (c *Client) UploadFilesAndWait(ctx context.Context, files []Prompt) (processed []Prompt, err error) {
	processed = []Prompt{}
	fileNames := []string{}

	fileIndex := 0
	for _, f := range files {
		if text, ok := f.(TextPrompt); ok {
			processed = append(processed, text)
		} else if file, ok := f.(FilePrompt); ok {
			// read mime type,
			if mimeType, reader, err := readMimeAndRecycle(file.reader); err == nil {
				// check mime type,
				var matchedMimeType string
				var supported bool
				if matchedMimeType, supported = checkMimeType(mimeType); !supported {
					// if it has a custom converter, convert it
					if fn, exists := c.fileConvertFuncs[matchedMimeType]; !exists {
						return nil, fmt.Errorf("MIME type of file[%d] (%s) not supported: %s", fileIndex, file.filename, mimeType.String())
					} else {
						if bs, err := io.ReadAll(reader); err == nil {
							if c.Verbose {
								log.Printf("> converting reader with custom file converter for %s...", matchedMimeType)
							}

							if converted, convertedMimeType, err := fn(bs); err == nil {
								reader = bytes.NewBuffer(converted)
								matchedMimeType = convertedMimeType
							} else {
								return nil, fmt.Errorf("converting %s failed: %w", matchedMimeType, err)
							}
						} else {
							return nil, fmt.Errorf("read failed while converting %s: %w", matchedMimeType, err)
						}
					}
				}

				// upload
				if uploaded, err := c.client.Files.Upload(
					ctx,
					reader,
					&genai.UploadFileConfig{
						MIMEType:    matchedMimeType,
						DisplayName: file.filename,
					},
				); err == nil {
					processed = append(processed, FilePrompt{
						filename: uploaded.Name,
						data: &genai.FileData{
							FileURI:  uploaded.URI,
							MIMEType: uploaded.MIMEType,
						},
					})

					fileNames = append(fileNames, uploaded.Name)
				} else {
					return nil, fmt.Errorf("failed to upload file[%d] (%s) for prompt: %w", fileIndex, file.filename, err)
				}
			} else {
				return nil, fmt.Errorf("failed to detect MIME type of file[%d] (%s): %w", fileIndex, file.filename, err)
			}

			fileIndex++
		} else if fbytes, ok := f.(BytesPrompt); ok {
			// read mime type,
			mimeType := mimetype.Detect(fbytes.bytes)

			// check mime type,
			var matchedMimeType string
			var supported bool
			if matchedMimeType, supported = checkMimeType(mimeType); !supported {
				// if it has a custom converter, convert it
				if fn, exists := c.fileConvertFuncs[matchedMimeType]; !exists {
					return nil, fmt.Errorf("MIME type of file[%d] (%d bytes) not supported: %s", fileIndex, len(fbytes.bytes), mimeType.String())
				} else {
					if c.Verbose {
						log.Printf("> converting bytes with custom file converter for %s...", matchedMimeType)
					}

					if converted, convertedMimeType, err := fn(fbytes.bytes); err == nil {
						fbytes.bytes = converted
						matchedMimeType = convertedMimeType
					} else {
						return nil, fmt.Errorf("converting %s failed: %w", matchedMimeType, err)
					}
				}
			}

			// upload
			if uploaded, err := c.client.Files.Upload(
				ctx,
				bytes.NewReader(fbytes.bytes),
				&genai.UploadFileConfig{
					MIMEType:    matchedMimeType,
					DisplayName: fmt.Sprintf("%d bytes of file", len(fbytes.bytes)),
				},
			); err == nil {
				processed = append(processed, FilePrompt{
					filename: uploaded.Name,
					data: &genai.FileData{
						FileURI:  uploaded.URI,
						MIMEType: uploaded.MIMEType,
					},
				})

				fileNames = append(fileNames, uploaded.Name)
			} else {
				return nil, fmt.Errorf("failed to upload file[%d] (%d bytes) for prompt: %w", fileIndex, len(fbytes.bytes), err)
			}

			fileIndex++
		} else if uri, ok := f.(URIPrompt); ok {
			processed = append(processed, uri)
		}
	}

	// NOTE: wait for all the uploaded files to be ready
	c.waitForFiles(ctx, fileNames)

	return processed, nil
}

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

// build prompt contents for generation
func (c *Client) buildPromptContents(ctx context.Context, prompts []Prompt, histories []genai.Content) (contents []*genai.Content, err error) {
	var processed []Prompt
	processed, err = c.UploadFilesAndWait(ctx, prompts)
	if err != nil {
		return nil, fmt.Errorf("failed to upload files for prompt: %w", err)
	}

	for _, history := range histories {
		for _, part := range history.Parts {
			contents = append(contents, &genai.Content{
				Role: history.Role,
				Parts: []*genai.Part{
					part,
				},
			})
		}
	}
	for _, prompt := range processed {
		contents = append(contents, &genai.Content{
			Role: string(RoleUser),
			Parts: []*genai.Part{
				ptr(prompt.ToPart()),
			},
		})
	}

	return contents, nil
}

// generate safety settings for all supported harm categories
func safetySettings(threshold *genai.HarmBlockThreshold) (settings []*genai.SafetySetting) {
	if threshold == nil {
		// threshold = ptr(genai.HarmBlockThresholdBlockOnlyHigh)
		threshold = ptr(genai.HarmBlockThresholdOff)
	}

	for _, category := range []genai.HarmCategory{
		// all categories supported by Gemini models
		genai.HarmCategoryHateSpeech,
		genai.HarmCategoryDangerousContent,
		genai.HarmCategoryHarassment,
		genai.HarmCategorySexuallyExplicit,
		genai.HarmCategoryCivicIntegrity,
	} {
		settings = append(settings, &genai.SafetySetting{
			// Method:    genai.HarmBlockMethodSeverity, // FIXME: => error: 'method parameter is not supported in Gemini API'
			Category:  category,
			Threshold: *threshold,
		})
	}

	return settings
}

// check if given file's mime type is matched and supported
//
// https://ai.google.dev/gemini-api/docs/file-prompting-strategies
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
		var mimeType *mimetype.MIME
		if mimeType, err = mimetype.DetectReader(f); err == nil {
			matchedMimeType, supported = checkMimeType(mimeType)

			return matchedMimeType, supported, nil
		}
	}

	return "", false, err
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

// ErrToStr converts error (possibly genai error) to string.
func ErrToStr(err error) (str string) {
	var ae *genai.APIError
	if errors.As(err, &ae) {
		return fmt.Sprintf("genai API error: %s", ae.Error())
	} else {
		return err.Error()
	}
}

// IsQuotaExceeded returns if given error is from execeeded API quota.
func IsQuotaExceeded(err error) bool {
	var ae *genai.APIError
	if errors.As(err, &ae) {
		if ae.Code == 429 {
			return true
		}
	}
	return false
}

// IsModelOverloaded returns if given error is from overloaded model.
func IsModelOverloaded(err error) bool {
	var ae *genai.APIError
	if errors.As(err, &ae) {
		if ae.Code == 503 && ae.Message == `The model is overloaded. Please try again later.` {
			return true
		}
	}
	return false
}

// read mime type of given io.Reader
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
//	1,048,576 input tokens for 'gemini-2.0-flash'
//	    8,192 input tokens for 'gemini-embedding-exp-03-07''
const (
	defaultChunkedTextLengthInBytes    uint = 1024 * 1024 * 2
	defaultOverlappedTextLengthInBytes uint = defaultChunkedTextLengthInBytes / 200
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
		end := min(start+int(chunkSize), len(text))

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
