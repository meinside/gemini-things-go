// types.go

package gt

import (
	"fmt"
	"io"

	"github.com/gabriel-vasile/mimetype"
	"google.golang.org/genai"
)

// role constants for convenience
const (
	RoleUser  genai.Role = genai.RoleUser
	RoleModel genai.Role = genai.RoleModel
)

// Prompt is an interface representing different types of input that can be
// converted into a `genai.Part` for use in generative AI model requests.
// It standardizes how various input forms (text, file, URI, bytes) are processed.
type Prompt interface {
	// String returns a string representation of the prompt, useful for logging or debugging.
	String() string
	// ToPart converts the prompt into a `genai.Part`.
	// The implementation determines how the specific prompt type is transformed
	// (e.g., text to TextPart, file URI to FileDataPart).
	ToPart() genai.Part
}

// TextPrompt represents a simple text-based prompt.
type TextPrompt struct {
	Text string // The actual text content of the prompt.
}

// ToPart converts the TextPrompt into a `genai.Part` containing the text.
func (p TextPrompt) ToPart() genai.Part {
	return genai.Part{
		Text: p.Text,
	}
}

// String returns a string representation of the TextPrompt, showing the text content.
func (p TextPrompt) String() string {
	return fmt.Sprintf("text='%s'", p.Text)
}

// PromptFromText creates a new TextPrompt from the given text string.
// It implements the Prompt interface.
func PromptFromText(text string) Prompt {
	return TextPrompt{
		Text: text,
	}
}

// FilePrompt represents a prompt that involves a file to be uploaded.
// The file content is provided via an io.Reader.
// After processing (e.g., uploading via client.processPromptToPartAndInfo), the `data` field
// will be populated with the URI and MIME type of the uploaded file from the server.
type FilePrompt struct {
	Filename string    // filename is the display name for the file, used during upload.
	Reader   io.Reader // reader provides the content of the file.

	// Data holds the URI and MIME type of the file after it has been uploaded
	// and the server has responded with the file's metadata.
	// This field is populated by processing functions like `client.processPromptToPartAndInfo`.
	// It is nil until the file is successfully processed and its URI obtained.
	Data *genai.FileData

	// When this data needs to be handled as a specific MIME type,
	// set this value then it will be forced to that type.
	ForcedMIMEType string
}

// ToPart converts the FilePrompt into a `genai.Part` using the FileData (URI and MIME type).
// This method relies on the `data` field being populated by a prior file upload step
// (e.g., within `client.processPromptToPartAndInfo`). If `p.data` is nil,
// it will result in a `genai.Part` with empty FileData, which may be invalid for API requests.
func (p FilePrompt) ToPart() genai.Part {
	return genai.Part{
		FileData: &genai.FileData{
			// DisplayName: p.data.DisplayName, // TODO: uncomment this line when Gemini API supports it
			FileURI:  p.Data.FileURI,
			MIMEType: p.Data.MIMEType,
		},
	}
}

// String returns a string representation of the FilePrompt.
// If the file has been uploaded and `data` is populated, it includes the URI and MIME type.
func (p FilePrompt) String() string {
	if p.Data != nil {
		return fmt.Sprintf("file='%s';uri='%s';mimeType=%s", p.Filename, p.Data.FileURI, p.Data.MIMEType)
	}
	return fmt.Sprintf("file='%s'", p.Filename)
}

// PromptFromFile creates a new FilePrompt with the given display filename and an io.Reader for its content.
// It implements the Prompt interface.
func PromptFromFile(
	filename string,
	reader io.Reader,
	forceMimeType ...string,
) Prompt {
	prompt := FilePrompt{
		Filename: filename,
		Reader:   reader,
	}

	if len(forceMimeType) > 0 {
		prompt.ForcedMIMEType = forceMimeType[0]
	}

	return prompt
}

// URIPrompt represents a prompt that uses a URI to point to file data
// (e.g., a gs:// URI for a file in Google Cloud Storage, or a publicly accessible HTTPS URI).
type URIPrompt struct {
	URI      string // The URI of the file.
	MIMEType string
}

// ToPart converts the URIPrompt into a `genai.Part` using the FileData URI.
func (p URIPrompt) ToPart() genai.Part {
	return genai.Part{
		FileData: &genai.FileData{
			FileURI:  p.URI,
			MIMEType: p.MIMEType,
		},
	}
}

// String returns a string representation of the URIPrompt.
func (p URIPrompt) String() string {
	return fmt.Sprintf("uri='%s'", p.URI)
}

// PromptFromURI creates a new URIPrompt from the given URI string.
// It implements the Prompt interface.
func PromptFromURI(uri, mimeType string) Prompt {
	return URIPrompt{
		URI:      uri,
		MIMEType: mimeType,
	}
}

// BytesPrompt represents a prompt where the file data is provided directly as a byte slice.
// This is typically used for smaller files that can be inlined in the request if not uploaded,
// or uploaded if they exceed size limits for inline data or if a file URI is preferred.
// The `filename` field can be used to provide a display name if the bytes are uploaded.
type BytesPrompt struct {
	Filename string // Optional display name for the byte data, used if uploaded.
	Bytes    []byte // The raw byte data of the file.

	// The MIME type of the byte data (e.g., "image/png"), typically auto-detected.
	MIMEType string

	// When this data needs to be handled as a specific MIME type,
	// set this value then it will be forced to that type.
	ForcedMIMEType string
}

// ToPart converts the BytesPrompt into a `genai.Part`.
// If the BytesPrompt was processed by `client.processPromptToPartAndInfo` and uploaded,
// it would have been converted to a FilePrompt, and that FilePrompt's ToPart would be used.
// This ToPart method is for when BytesPrompt is used directly to form an InlineData part.
func (p BytesPrompt) ToPart() genai.Part {
	return genai.Part{
		InlineData: &genai.Blob{
			Data:     p.Bytes,
			MIMEType: p.MIMEType,
		},
	}
}

// String returns a string representation of the BytesPrompt, including its filename (if any), length, and MIME type.
func (p BytesPrompt) String() string {
	if p.Filename != "" {
		return fmt.Sprintf("bytes(file='%s')[%d];mimeType=%s", p.Filename, len(p.Bytes), p.MIMEType)
	}
	return fmt.Sprintf("bytes[%d];mimeType=%s", len(p.Bytes), p.MIMEType)
}

// PromptFromBytes creates a new BytesPrompt from a byte slice.
// The MIME type is automatically detected from the byte content.
// It implements the Prompt interface.
// This version does not include a filename.
func PromptFromBytes(
	bytes []byte,
	forceMimeType ...string,
) Prompt {
	prompt := BytesPrompt{
		Bytes:    bytes,
		MIMEType: mimetype.Detect(bytes).String(),
	}

	if len(forceMimeType) > 0 {
		prompt.ForcedMIMEType = forceMimeType[0]
	}

	return prompt
}

// PromptFromBytesWithName creates a new BytesPrompt from a byte slice with an associated filename.
// The MIME type is automatically detected from the byte content.
// It implements the Prompt interface.
func PromptFromBytesWithName(
	bytes []byte,
	filename string,
	forceMimeType ...string,
) Prompt {
	prompt := BytesPrompt{
		Filename: filename,
		Bytes:    bytes,
		MIMEType: mimetype.Detect(bytes).String(),
	}

	if len(forceMimeType) > 0 {
		prompt.ForcedMIMEType = forceMimeType[0]
	}

	return prompt
}

// NumTokens represents the breakdown of token usage for a generation request.
type NumTokens struct {
	Cached   int32 // Number of tokens from cached content.
	Output   int32 // Number of tokens in the generated output (candidates).
	Input    int32 // Number of tokens in the input prompt.
	Thoughts int32 // Number of tokens used for model's internal "thoughts" (if applicable and requested).
	ToolUse  int32 // Number of tokens used for tool interactions (function calls, etc.).
	Total    int32 // Total number of tokens processed for the request.
}

// function definitions
type (
	// FnSystemInstruction is a function type that returns a string to be used as the system instruction.
	// This allows for dynamic generation of system instructions if needed.
	FnSystemInstruction func() string

	// FnConvertBytes is a function type for converting file bytes into potentially multiple files.
	// It's used when a single unsupported file needs to be converted into one or more supported files
	// (e.g., a multi-page PDF into multiple images).
	FnConvertBytes func(filename string, bytes []byte) ([]ConvertedFile, error)

	// ConvertedFile represents a single converted file from FnConvertBytes.
	ConvertedFile struct {
		Bytes    []byte // The converted file bytes.
		MimeType string // The MIME type of the converted file.
		Filename string // (Optional) The new name of the converted file.
	}
)
