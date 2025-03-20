// types.go

package gt

import (
	"fmt"
	"io"

	"github.com/gabriel-vasile/mimetype"
	"google.golang.org/genai"

	old "github.com/google/generative-ai-go/genai" // FIXME: remove this after file APIs are implemented
)

// Prompt interface for various types of prompt
type Prompt interface {
	String() string
	ToPart() genai.Part
}

// TextPrompt struct
type TextPrompt struct {
	text string
}

// ToPart converts TextPrompt to genai.Part.
func (p TextPrompt) ToPart() genai.Part {
	return genai.Part{
		Text: p.text,
	}
}

// String returns the text prompt as a string.
func (p TextPrompt) String() string {
	return fmt.Sprintf("text='%s'", p.text)
}

// PromptFromText returns a Prompt with given text.
func PromptFromText(text string) Prompt {
	return TextPrompt{
		text: text,
	}
}

// FilePrompt struct
type FilePrompt struct {
	filename string
	reader   io.Reader

	data *old.FileData // FIXME: change to *genai.FileData when file APIs are implemented in `genai`
}

// ToPart converts FilePrompt to genai.Part.
func (p FilePrompt) ToPart() genai.Part {
	return genai.Part{
		FileData: &genai.FileData{
			FileURI:  p.data.URI,
			MIMEType: p.data.MIMEType,
		},
	}
}

// String returns the file prompt as a string.
func (p FilePrompt) String() string {
	if p.data != nil {
		return fmt.Sprintf("file='%s';uri='%s';mimeType=%s", p.filename, p.data.URI, p.data.MIMEType)
	}
	return fmt.Sprintf("file='%s'", p.filename)
}

// PromptFromFile returns a Prompt with given filename and reader.
func PromptFromFile(filename string, reader io.Reader) Prompt {
	return FilePrompt{
		filename: filename,
		reader:   reader,
	}
}

// URIPrompt struct
type URIPrompt struct {
	uri string
}

// ToPart converts URIPrompt to genai.Part.
func (p URIPrompt) ToPart() genai.Part {
	return genai.Part{
		FileData: &genai.FileData{
			FileURI: p.uri,
		},
	}
}

// String returns the URI prompt as a string.
func (p URIPrompt) String() string {
	return fmt.Sprintf("uri='%s'", p.uri)
}

// PromptFromURI returns a Prompt with given URI.
func PromptFromURI(uri string) Prompt {
	return URIPrompt{
		uri: uri,
	}
}

// BytesPrompt struct
type BytesPrompt struct {
	bytes    []byte
	mimeType string
}

// ToPart converts BytesPrompt to genai.Part.
func (p BytesPrompt) ToPart() genai.Part {
	return genai.Part{
		InlineData: &genai.Blob{
			Data:     p.bytes,
			MIMEType: p.mimeType,
		},
	}
}

// String returns the inline file prompt as a string.
func (p BytesPrompt) String() string {
	return fmt.Sprintf("bytes[%d];mimeType=%s", len(p.bytes), p.mimeType)
}

// PromptFromBytes returns a Prompt with given bytes.
func PromptFromBytes(bytes []byte) Prompt {
	return BytesPrompt{
		bytes:    bytes,
		mimeType: mimetype.Detect(bytes).String(),
	}
}

// GenerationOptions struct for text generations
type GenerationOptions struct {
	// generation config
	Config *genai.GenerationConfig

	// tool config
	Tools      []*genai.Tool
	ToolConfig *genai.ToolConfig

	// safety settings: harm block threshold
	HarmBlockThreshold *genai.HarmBlockThreshold

	// for reusing the cached content
	CachedContent string

	// for multimodal response
	ResponseModalities []string
	MediaResolution    genai.MediaResolution
	SpeechConfig       *genai.SpeechConfig

	// history (for session)
	History []genai.Content
}

// NewGenerationOptions returns a new GenerationOptions with default values.
func NewGenerationOptions() *GenerationOptions {
	return &GenerationOptions{
		HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
	}
}

// StreamCallbackData struct contains the data for stream callback function.
type StreamCallbackData struct {
	// when there is a text delta,
	TextDelta *string

	// when there is a file bytes array,
	InlineData *genai.Blob

	// thinking...?
	Thought bool

	// when there is a function call,
	FunctionCall     *genai.FunctionCall
	FunctionResponse *genai.FunctionResponse

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
type (
	FnSystemInstruction func() string

	// returns converted bytes, converted mime type, and/or error
	FnConvertBytes func(bytes []byte) ([]byte, string, error)

	FnStreamCallback func(callbackData StreamCallbackData)
)
