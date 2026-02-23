// utils.go

package gt

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"log"
	"net/http"
	"os"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gabriel-vasile/mimetype"
	"google.golang.org/genai"
)

const (
	PrefixAPIError = "genai API error"
)

const (
	uploadedFileStateCheckIntervalMilliseconds       = 300  // 0.3 second
	uploadedFileSearchStateCheckIntervalMilliseconds = 1000 // 1.0 second

	generatingVideoFileStateCheckIntervalMilliseconds = 5000 // 5.0 seconds
)

// waitForFiles waits for all specified uploaded files
// to reach the "ACTIVE" state.
// It uses a sync.WaitGroup to wait for all concurrent checks to complete.
// This is an internal helper function.
func (c *Client) waitForFiles(
	ctx context.Context,
	fileNames []string,
) {
	var wg sync.WaitGroup
	for _, fileName := range fileNames {
		wg.Go(func() {
			for {
				if file, err := c.client.Files.Get(
					ctx,
					fileName,
					&genai.GetFileConfig{},
				); err == nil {
					if file.State == genai.FileStateActive {
						break
					} else {
						time.Sleep(uploadedFileStateCheckIntervalMilliseconds * time.Millisecond)
					}
				} else {
					time.Sleep(uploadedFileStateCheckIntervalMilliseconds * time.Millisecond)
				}
			}
		})
	}
	wg.Wait()
}

// waitForFilesForSearch waits for all specified uploaded documents for search
// to reach the "ACTIVE" state.
// It uses a sync.WaitGroup to wait for all concurrent checks to complete.
// This is an internal helper function.
//
// FIXME: (not working yet) state is returned empty
func (c *Client) waitForFilesForSearch(
	ctx context.Context,
	fileNames []string,
) {
	var wg sync.WaitGroup
	for _, fileName := range fileNames {
		wg.Go(func() {
			for {
				if document, err := c.client.FileSearchStores.Documents.Get(
					ctx,
					fileName,
					&genai.GetDocumentConfig{},
				); err == nil {
					if document.State == genai.DocumentStateActive {
						break
					} else {
						time.Sleep(uploadedFileSearchStateCheckIntervalMilliseconds * time.Millisecond)
					}
				} else {
					time.Sleep(uploadedFileSearchStateCheckIntervalMilliseconds * time.Millisecond)
				}
			}
		})
	}
	wg.Wait()
}

// processPromptToPartAndInfo is an internal helper function that processes a single Prompt.
// It handles potential file uploads for FilePrompt and BytesPrompt types.
//   - For FilePrompt: it reads the content, detects/converts MIME type if necessary, uploads the file,
//     and returns an updated FilePrompt with the server-assigned filename and FileData.
//   - For BytesPrompt: it detects/converts MIME type, uploads the bytes as a file, and returns
//     a new FilePrompt representing the uploaded file with its server-assigned filename and FileData.
//   - For TextPrompt and URIPrompt: it converts them to their genai.Part representation directly.
//
// It returns:
//   - *genai.Part: The genai.Part representation of the processed prompt.
//   - Prompt: The updated prompt (e.g., FilePrompt with populated .data, or BytesPrompt converted to FilePrompt).
//   - *string: The server-assigned filename if an upload occurred, used for waiting for the file to become active. nil otherwise.
//   - error: Any error encountered during processing.
func (c *Client) processPromptToPartAndInfo(
	ctx context.Context,
	p Prompt, // The prompt to process.
	promptIndex int,
	ignoreMimeType ...bool,
) (part *genai.Part, updatedPrompt Prompt, filenameForWaiting *string, err error) {
	ignoreMime := len(ignoreMimeType) > 0 && ignoreMimeType[0]

	switch prompt := p.(type) {
	case TextPrompt, URIPrompt:
		return new(prompt.ToPart()), prompt, nil, nil

	case FilePrompt:
		currentReader := prompt.Reader
		if currentReader == nil {
			return nil, prompt, nil, fmt.Errorf(
				"prompts[%d] has a nil reader (%s)",
				promptIndex,
				prompt.Filename,
			)
		}

		// check mimeType type
		var matchedMimeType string
		var supported bool
		if prompt.ForcedMIMEType == "" { // detect the MIME type
			var mimeType *mimetype.MIME
			var err error
			mimeType, currentReader, err = readMimeAndRecycle(currentReader) // Reuse the recycled reader
			if err != nil {
				return nil, prompt, nil, fmt.Errorf(
					"failed to detect MIME type of prompts[%d] (%s): %w",
					promptIndex,
					prompt.Filename,
					err,
				)
			}
			matchedMimeType, supported = checkMimeTypeForFile(mimeType)
		} else { // if a forced MIME type is provided, use it
			matchedMimeType = prompt.ForcedMIMEType
			supported = true
		}

		if !ignoreMime && !supported {
			fn, exists := c.fileConvertFuncs[matchedMimeType]
			if !exists {
				return nil, prompt, nil, fmt.Errorf(
					"MIME type of prompts[%d] (%s) not supported: %s",
					promptIndex,
					prompt.Filename,
					matchedMimeType,
				)
			}
			bs, readErr := io.ReadAll(currentReader)
			if readErr != nil {
				return nil, prompt, nil, fmt.Errorf(
					"read failed while converting %s for prompts[%d] (%s): %w",
					matchedMimeType,
					promptIndex,
					prompt.Filename,
					readErr,
				)
			}
			if c.Verbose {
				log.Printf(
					"> converting with custom file converter for %s for prompts[%d] (%s)...",
					matchedMimeType,
					promptIndex,
					prompt.Filename,
				)
			}
			converted, convertedMimeType, convErr := fn(bs)
			if convErr != nil {
				return nil, prompt, nil, fmt.Errorf(
					"converting %s for prompts[%d] (%s) failed: %w",
					matchedMimeType,
					promptIndex,
					prompt.Filename,
					convErr,
				)
			}
			currentReader = bytes.NewBuffer(converted)
			matchedMimeType = convertedMimeType
		}

		uploadedFile, uploadErr := c.UploadFile(
			ctx,
			currentReader,
			prompt.Filename, // Use original filename for display
			matchedMimeType,
		)
		if uploadErr != nil {
			return nil, prompt, nil, fmt.Errorf(
				"failed to upload prompts[%d] (%s): %w",
				promptIndex,
				prompt.Filename,
				uploadErr,
			)
		}

		var updatedFilePrompt FilePrompt
		switch c.Type {
		case genai.BackendGeminiAPI:
			updatedFilePrompt = FilePrompt{
				Filename: uploadedFile.Name, // Store the server-generated unique name
				Data: &genai.FileData{
					// DisplayName: uploadedFile.Name, // FIXME: uncomment this line when Gemini API supports it
					FileURI:  uploadedFile.URI,
					MIMEType: uploadedFile.MIMEType,
				},
			}
		case genai.BackendVertexAI:
			updatedFilePrompt = FilePrompt{
				Filename: uploadedFile.DisplayName,
				Data: &genai.FileData{
					DisplayName: uploadedFile.DisplayName,
					FileURI:     uploadedFile.URI,
					MIMEType:    uploadedFile.MIMEType,
				},
			}
		}
		return new(updatedFilePrompt.ToPart()), updatedFilePrompt, new(uploadedFile.Name), nil

	case BytesPrompt:
		currentBytes := prompt.Bytes

		// check mime type
		var matchedMimeType string
		var supported bool
		if prompt.ForcedMIMEType == "" { // detect the MIME type
			matchedMimeType, supported = checkMimeTypeForFile(mimetype.Detect(currentBytes))
		} else { // if a forced MIME type is provided, use it
			matchedMimeType = prompt.ForcedMIMEType
			supported = true
		}

		if !ignoreMime && !supported {
			fn, exists := c.fileConvertFuncs[matchedMimeType]
			if !exists {
				return nil, prompt, nil, fmt.Errorf(
					"MIME type of prompts[%d] (%d bytes) not supported: %s",
					promptIndex,
					len(currentBytes),
					matchedMimeType,
				)
			}
			if c.Verbose {
				log.Printf(
					"> converting prompts[%d] (%d bytes) with custom file converter for %s...",
					promptIndex,
					len(currentBytes),
					matchedMimeType,
				)
			}
			converted, convertedMimeType, convErr := fn(currentBytes)
			if convErr != nil {
				return nil, prompt, nil, fmt.Errorf(
					"converting prompts[%d] (%d bytes) with MIME type %s failed: %w",
					promptIndex,
					len(currentBytes),
					matchedMimeType,
					convErr,
				)
			}
			currentBytes = converted
			matchedMimeType = convertedMimeType
		}

		displayName := prompt.Filename
		if displayName == "" {
			displayName = fmt.Sprintf("prompts[%d] (%d bytes)", promptIndex, len(currentBytes))
		}

		uploadedFile, uploadErr := c.UploadFile(
			ctx,
			bytes.NewReader(currentBytes),
			displayName,
			matchedMimeType,
		)
		if uploadErr != nil {
			return nil, prompt, nil, fmt.Errorf(
				"failed to upload prompts[%d] (%d bytes) as file: %w",
				promptIndex,
				len(prompt.Bytes),
				uploadErr,
			)
		}

		// Convert BytesPrompt to FilePrompt after upload
		var fileDataPrompt FilePrompt
		switch c.Type {
		case genai.BackendGeminiAPI:
			fileDataPrompt = FilePrompt{
				Filename: uploadedFile.Name, // Store the server-generated unique name
				Data: &genai.FileData{
					FileURI:  uploadedFile.URI,
					MIMEType: uploadedFile.MIMEType,
				},
			}
		case genai.BackendVertexAI:
			fileDataPrompt = FilePrompt{
				Filename: uploadedFile.DisplayName,
				Data: &genai.FileData{
					DisplayName: uploadedFile.DisplayName,
					FileURI:     uploadedFile.URI,
					MIMEType:    uploadedFile.MIMEType,
				},
			}
		}
		return new(fileDataPrompt.ToPart()), fileDataPrompt, new(uploadedFile.Name), nil

	default:
		return nil, p, nil, fmt.Errorf(
			"unknown or unsupported type of prompts[%d]: %T",
			promptIndex,
			p,
		)
	}
}

// check if there is a function call in the candidates
func hasFunctionCall(candidates []*genai.Candidate) (*genai.FunctionCall, bool) {
	for _, cand := range candidates {
		if cand.Content != nil {
			for _, part := range cand.Content.Parts {
				if part.FunctionCall != nil {
					return part.FunctionCall, true
				}
			}
		}
	}
	return nil, false
}

// UploadFilesAndWait processes a slice of Prompts,
// handling file uploads for FilePrompt and BytesPrompt types.
// It waits for all uploaded files to become "ACTIVE".
//
// For FilePrompt and BytesPrompt instances that require uploading:
//   - FilePrompt instances will have their `data` field populated with the `*genai.FileData`
//     (URI and MIME type) returned by the server after successful upload. Their `filename`
//     field will be updated to the server-assigned unique name.
//   - BytesPrompt instances are converted into FilePrompt instances after their content
//     is uploaded as a file. The new FilePrompt will have its `filename` and `data`
//     fields populated accordingly.
//
// TextPrompt and URIPrompt instances are returned as is.
//
// Parameters:
//   - ctx: The context for the operation.
//   - prompts: A slice of Prompt interfaces to process.
//
// Returns:
//   - processedPrompts: A new slice of Prompt interfaces. For uploaded files,
//     FilePrompt instances are updated with server data, and BytesPrompt instances
//     are converted to FilePrompt instances. Other prompt types remain unchanged.
//   - err: An error if any part of the processing (like file upload or waiting) fails.
func (c *Client) UploadFilesAndWait(
	ctx context.Context,
	prompts []Prompt,
	ignoreMimeType ...bool,
) (processedPrompts []Prompt, err error) {
	processedPrompts = make([]Prompt, 0, len(prompts))
	fileNamesToWaitFor := []string{}

	for i, p := range prompts {
		_, updatedPrompt, filenameForWaiting, processErr := c.processPromptToPartAndInfo(
			ctx,
			p,
			i,
			ignoreMimeType...,
		)

		if processErr != nil {
			return nil, fmt.Errorf(
				"error processing prompts[%d]: %w",
				i,
				processErr,
			)
		}
		processedPrompts = append(processedPrompts, updatedPrompt)
		if filenameForWaiting != nil {
			fileNamesToWaitFor = append(fileNamesToWaitFor, *filenameForWaiting)
		}
	}

	if c.Type == genai.BackendGeminiAPI && len(fileNamesToWaitFor) > 0 {
		if c.Verbose {
			log.Printf(
				"> waiting for %d file(s) to become active: %v",
				len(fileNamesToWaitFor),
				fileNamesToWaitFor,
			)
		}
		c.waitForFiles(ctx, fileNamesToWaitFor)
		if c.Verbose {
			log.Printf(
				"> all %d file(s) are active.",
				len(fileNamesToWaitFor),
			)
		}
	}

	return processedPrompts, nil
}

// FuncArg is a generic helper function to safely extract and type-assert a value
// from a map[string]any, which is commonly used for function call arguments
// in `genai.FunctionCall.Args`.
//
// Parameters:
//   - from: The map (typically `FunctionCall.Args`) to extract the value from.
//   - key: The key of the desired argument.
//
// Type Parameter:
//   - T: The expected type of the argument.
//
// Returns:
//   - *T: A pointer to the extracted value of type T if found and type assertion is successful.
//     Returns nil if the key is not found.
//   - error: An error if the key is found but the value cannot be cast to type T.
//     Returns nil if the key is not found or if extraction and casting are successful.
func FuncArg[T any](from map[string]any, key string) (*T, error) {
	if v, exists := from[key]; exists {
		if cast, ok := v.(T); ok {
			return &cast, nil
		}
		return nil, fmt.Errorf(
			"could not cast %[2]T '%[1]s' (%[2]v) to %[3]T",
			key,
			v,
			*new(T),
		)
	}
	return nil, nil // not found
}

// PromptsToContents constructs a `[]*genai.Content` slice with given `prompts` and `histories`.
func (c *Client) PromptsToContents(
	ctx context.Context,
	prompts []Prompt, histories []genai.Content,
) (contents []*genai.Content, err error) {
	// Process prompts (uploads files, etc.) and get updated prompts
	// where FilePrompt.data is populated and BytesPrompt (if uploaded) is converted to FilePrompt.
	processedPromptsAfterUpload, err := c.UploadFilesAndWait(ctx, prompts)
	if err != nil {
		return nil, fmt.Errorf(
			"failed to upload files or process prompts: %w",
			err,
		)
	}

	// Add history contents first
	for _, history := range histories {
		// It's safer to copy parts if h.Parts could be modified elsewhere,
		// but typically history is immutable here.
		contents = append(contents, &genai.Content{
			Role:  history.Role,
			Parts: history.Parts,
		})
	}

	// Add user prompts
	// Each prompt from processedPromptsAfterUpload should now directly yield its genai.Part
	// via its ToPart() method, especially FilePrompt which now uses its populated .data field.
	userPromptParts := []*genai.Part{}
	for _, p := range processedPromptsAfterUpload {
		part := p.ToPart()
		userPromptParts = append(userPromptParts, &part)
	}
	if len(userPromptParts) > 0 {
		contents = append(contents, &genai.Content{
			Role:  string(RoleUser),
			Parts: userPromptParts,
		})
	}

	return contents, nil
}

// GenerateSafetySettings creates a slice of *genai.SafetySetting for all supported harm categories,
// applying the given threshold. If threshold is nil, it defaults to HarmBlockThresholdOff.
// This function is an internal helper.
func GenerateSafetySettings(threshold *genai.HarmBlockThreshold) (settings []*genai.SafetySetting) {
	if threshold == nil {
		threshold = new(genai.HarmBlockThresholdOff)
	}

	for _, category := range []genai.HarmCategory{
		// all categories supported by Gemini models
		genai.HarmCategoryHateSpeech,
		genai.HarmCategoryDangerousContent,
		genai.HarmCategoryHarassment,
		genai.HarmCategorySexuallyExplicit,
		/*
			genai.HarmCategoryImageHate,
			genai.HarmCategoryImageDangerousContent,
			genai.HarmCategoryImageHarassment,
			genai.HarmCategoryImageSexuallyExplicit,
		*/
	} {
		settings = append(settings, &genai.SafetySetting{
			// Method:    genai.HarmBlockMethodSeverity, // FIXME: => error: 'method parameter is not supported in Gemini API'
			Category:  category,
			Threshold: *threshold,
		})
	}

	return settings
}

// checkMimeTypeForFile is an internal helper function that checks
// if a given MIME type (as detected by the mimetype library) is
// supported by the Gemini API.
// It returns the matched MIME type string
// (which might be a more general type if an alias matches)
// and a boolean indicating if it's supported.
//
// See:
//   - Images: https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-image
//   - Audios: https://ai.google.dev/gemini-api/docs/audio?lang=go#supported-formats
//   - Videos: https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-video
//   - Documents: https://ai.google.dev/gemini-api/docs/document-processing?lang=go#technical-details
func checkMimeTypeForFile(
	mimeType *mimetype.MIME,
) (matched string, supported bool) {
	if mimeType == nil { // FIXME
		return `application/octet-stream`, false
	}

	return func(
		mimeType *mimetype.MIME,
	) (matchedMimeType string, supportedMimeType bool) {
		matchedMimeType = mimeType.String() // fallback, used if a more specific alias isn't found but it's still supported.

		switch {
		case slices.ContainsFunc([]string{
			// images
			//
			// https://ai.google.dev/gemini-api/docs/image-understanding?lang=go#supported-formats
			`image/png`,
			`image/jpeg`,
			`image/webp`,
			`image/heic`,
			`image/heif`,

			// audios
			//
			// https://ai.google.dev/gemini-api/docs/audio?lang=go#supported-formats
			`audio/wav`,
			`audio/mp3`,
			`audio/aiff`,
			`audio/aac`,
			`audio/ogg`,
			`audio/flac`,

			// videos
			//
			// https://ai.google.dev/gemini-api/docs/vision?lang=go#technical-details-video
			`video/mp4`,
			`video/mpeg`,
			`video/mov`,
			`video/avi`,
			`video/x-flv`,
			`video/mpg`,
			`video/webm`,
			`video/wmv`,
			`video/3gpp`,

			// document formats
			//
			// https://ai.google.dev/gemini-api/docs/document-processing?lang=go#document-types
			//
			// https://ai.google.dev/gemini-api/docs/file-input-methods#text
			`text/html`,
			`text/css`,
			`text/plain`,
			`text/xml`,
			`text/md`,
			`text/csv`,
			`text/rtf`,
			// https://ai.google.dev/gemini-api/docs/file-input-methods#application
			`application/json`, `text/javascript`, `application/x-javascript`,
			`application/pdf`,
			// https://ai.google.dev/gemini-api/docs/file-input-methods#image
			`image/bmp`,
			//`image/jpeg`,
			//`image/png`,
			//`image/webp`,
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

// checkMimeTypeForFileSearch is an internal helper function that checks
// if a given MIME type (as detected by the mimetype library) is
// supported for file search by the Gemini API.
// It returns the matched MIME type string
// (which might be a more general type if an alias matches)
// and a boolean indicating if it's supported.
//
// See:
//   - Application: https://ai.google.dev/gemini-api/docs/file-search#application
//   - Text:        https://ai.google.dev/gemini-api/docs/file-search#text
func checkMimeTypeForFileSearch(
	mimeType *mimetype.MIME,
) (matched string, supported bool) {
	return func(
		mimeType *mimetype.MIME,
	) (matchedMimeType string, supportedMimeType bool) {
		matchedMimeType = mimeType.String() // fallback, used if a more specific alias isn't found but it's still supported.

		switch {
		case slices.ContainsFunc([]string{
			// applications
			//
			// https://ai.google.dev/gemini-api/docs/file-search#application
			`application/dart`,
			`application/ecmascript`,
			`application/json`,
			`application/ms-java`,
			`application/msword`,
			`application/pdf`,
			`application/sql`,
			`application/typescript`,
			`application/vnd.curl`,
			`application/vnd.dart`,
			`application/vnd.ibm.secure-container`,
			`application/vnd.jupyter`,
			`application/vnd.ms-excel`,
			`application/vnd.oasis.opendocument.text`,
			`application/vnd.openxmlformats-officedocument.presentationml.presentation`,
			`application/vnd.openxmlformats-officedocument.spreadsheetml.sheet`,
			`application/vnd.openxmlformats-officedocument.wordprocessingml.document`,
			`application/vnd.openxmlformats-officedocument.wordprocessingml.template`,
			`application/x-csh`,
			`application/x-hwp`,
			`application/x-hwp-v5`,
			`application/x-latex`,
			`application/x-php`,
			`application/x-powershell`,
			`application/x-sh`,
			`application/x-shellscript`,
			`application/x-tex`,
			`application/x-zsh`,
			`application/xml`,
			`application/zip`,

			// text
			//
			// https://ai.google.dev/gemini-api/docs/file-search#text
			`text/1d-interleaved-parityfec`,
			`text/RED`,
			`text/SGML`,
			`text/cache-manifest`,
			`text/calendar`,
			`text/cql`,
			`text/cql-extension`,
			`text/cql-identifier`,
			`text/css`,
			`text/csv`,
			`text/csv-schema`,
			`text/dns`,
			`text/encaprtp`,
			`text/enriched`,
			`text/example`,
			`text/fhirpath`,
			`text/flexfec`,
			`text/fwdred`,
			`text/gff3`,
			`text/grammar-ref-list`,
			`text/hl7v2`,
			`text/html`,
			`text/javascript`,
			`text/jcr-cnd`,
			`text/jsx`,
			`text/markdown`,
			`text/mizar`,
			`text/n3`,
			`text/parameters`,
			`text/parityfec`,
			`text/php`,
			`text/plain`,
			`text/provenance-notation`,
			`text/prs.fallenstein.rst`,
			`text/prs.lines.tag`,
			`text/prs.prop.logic`,
			`text/raptorfec`,
			`text/rfc822-headers`,
			`text/rtf`,
			`text/rtp-enc-aescm128`,
			`text/rtploopback`,
			`text/rtx`,
			`text/sgml`,
			`text/shaclc`,
			`text/shex`,
			`text/spdx`,
			`text/strings`,
			`text/t140`,
			`text/tab-separated-values`,
			`text/texmacs`,
			`text/troff`,
			`text/tsv`,
			`text/tsx`,
			`text/turtle`,
			`text/ulpfec`,
			`text/uri-list`,
			`text/vcard`,
			`text/vnd.DMClientScript`,
			`text/vnd.IPTC.NITF`,
			`text/vnd.IPTC.NewsML`,
			`text/vnd.a`,
			`text/vnd.abc`,
			`text/vnd.ascii-art`,
			`text/vnd.curl`,
			`text/vnd.debian.copyright`,
			`text/vnd.dvb.subtitle`,
			`text/vnd.esmertec.theme-descriptor`,
			`text/vnd.exchangeable`,
			`text/vnd.familysearch.gedcom`,
			`text/vnd.ficlab.flt`,
			`text/vnd.fly`,
			`text/vnd.fmi.flexstor`,
			`text/vnd.gml`,
			`text/vnd.graphviz`,
			`text/vnd.hans`,
			`text/vnd.hgl`,
			`text/vnd.in3d.3dml`,
			`text/vnd.in3d.spot`,
			`text/vnd.latex-z`,
			`text/vnd.motorola.reflex`,
			`text/vnd.ms-mediapackage`,
			`text/vnd.net2phone.commcenter.command`,
			`text/vnd.radisys.msml-basic-layout`,
			`text/vnd.senx.warpscript`,
			`text/vnd.sosi`,
			`text/vnd.sun.j2me.app-descriptor`,
			`text/vnd.trolltech.linguist`,
			`text/vnd.wap.si`,
			`text/vnd.wap.sl`,
			`text/vnd.wap.wml`,
			`text/vnd.wap.wmlscript`,
			`text/vtt`,
			`text/wgsl`,
			`text/x-asm`,
			`text/x-bibtex`,
			`text/x-boo`,
			`text/x-c`,
			`text/x-c++hdr`,
			`text/x-c++src`,
			`text/x-cassandra`,
			`text/x-chdr`,
			`text/x-coffeescript`,
			`text/x-component`,
			`text/x-csh`,
			`text/x-csharp`,
			`text/x-csrc`,
			`text/x-cuda`,
			`text/x-d`,
			`text/x-diff`,
			`text/x-dsrc`,
			`text/x-emacs-lisp`,
			`text/x-erlang`,
			`text/x-gff3`,
			`text/x-go`,
			`text/x-haskell`,
			`text/x-java`,
			`text/x-java-properties`,
			`text/x-java-source`,
			`text/x-kotlin`,
			`text/x-lilypond`,
			`text/x-lisp`,
			`text/x-literate-haskell`,
			`text/x-lua`,
			`text/x-moc`,
			`text/x-objcsrc`,
			`text/x-pascal`,
			`text/x-pcs-gcd`,
			`text/x-perl`,
			`text/x-perl-script`,
			`text/x-python`,
			`text/x-python-script`,
			`text/x-r-markdown`,
			`text/x-rsrc`,
			`text/x-rst`,
			`text/x-ruby-script`,
			`text/x-rust`,
			`text/x-sass`,
			`text/x-scala`,
			`text/x-scheme`,
			`text/x-script.python`,
			`text/x-scss`,
			`text/x-setext`,
			`text/x-sfv`,
			`text/x-sh`,
			`text/x-siesta`,
			`text/x-sos`,
			`text/x-sql`,
			`text/x-swift`,
			`text/x-tcl`,
			`text/x-tex`,
			`text/x-vbasic`,
			`text/x-vcalendar`,
			`text/xml`,
			`text/xml-dtd`,
			`text/xml-external-parsed-entity`,
			`text/yaml`,
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

// SupportedMimeType detects the MIME type of the given byte data and checks if it's a supported
// format for the Gemini API (as defined in `checkMimeType`).
//
// Parameters:
//   - data: A byte slice containing the file data.
//
// Returns:
//   - matchedMimeType: The string representation of the detected MIME type if successful,
//     or the result of `http.DetectContentType` if `mimetype.DetectReader` fails.
//   - supported: A boolean indicating true if the MIME type is supported, false otherwise.
//   - err: An error if MIME type detection fails significantly (e.g., reader error, though less likely with bytes.Reader).
func SupportedMimeType(data []byte) (matchedMimeType string, supported bool, err error) {
	var mimeType *mimetype.MIME
	if mimeType, err = mimetype.DetectReader(bytes.NewReader(data)); err == nil {
		matchedMimeType, supported = checkMimeTypeForFile(mimeType)

		return matchedMimeType, supported, nil
	}

	return http.DetectContentType(data), false, err
}

// SupportedMimeTypePath opens a file at the given path, detects its MIME type,
// and checks if it's a supported format for the Gemini API (as defined in `checkMimeType`).
//
// Parameters:
//   - filepath: The path to the file.
//
// Returns:
//   - matchedMimeType: The string representation of the detected MIME type if successful.
//   - supported: A boolean indicating true if the MIME type is supported, false otherwise.
//   - err: An error if opening the file or detecting the MIME type fails.
func SupportedMimeTypePath(filepath string) (matchedMimeType string, supported bool, err error) {
	var f *os.File
	if f, err = os.Open(filepath); err == nil {
		var mimeType *mimetype.MIME
		if mimeType, err = mimetype.DetectReader(f); err == nil {
			matchedMimeType, supported = checkMimeTypeForFile(mimeType)

			return matchedMimeType, supported, nil
		}
	}

	return "", false, err
}

// prettify given thing in JSON format
func prettify(v any, flatten ...bool) string {
	if len(flatten) > 0 && flatten[0] {
		if bytes, err := json.Marshal(v); err == nil {
			return string(bytes)
		}
	} else {
		if bytes, err := json.MarshalIndent(v, "", "  "); err == nil {
			return string(bytes)
		}
	}
	return fmt.Sprintf("%+v", v)
}

// APIError checks if the provided error is a `*genai.APIError`
// and returns it if it is.
func APIError(err error) (ae genai.APIError, isAPIError bool) {
	if errors.As(err, &ae) {
		return ae, true
	}
	return genai.APIError{}, false
}

// ErrToStr converts an error into a string.
// If the error is a `*genai.APIError`, it formats it with a prefix "genai API error: ".
// Otherwise, it calls the error's `Error()` method.
func ErrToStr(err error) (str string) {
	if ae, isAPIError := APIError(err); isAPIError {
		var errStr string
		if bs, e := json.MarshalIndent(ae, "", "  "); e == nil {
			errStr = string(bs)
		} else {
			errStr = ae.Error()
		}
		return fmt.Sprintf("%s: %s", PrefixAPIError, errStr)
	}

	// For non-API errors, or if ae is nil after errors.As (shouldn't happen if As returns true)
	if err != nil {
		return err.Error()
	}

	return "" // Should not happen if err was not nil
}

// ErrDetails returns the `Details` of given `genai.APIError`.
// Returns nil if it is not a `genai.APIError`, or something goes wrong with it.
func ErrDetails(err error) []map[string]any {
	if ae, isAPIError := APIError(err); isAPIError {
		return ae.Details
	}
	return nil
}

var (
	// regular expressions for checking HTTP error strings
	regexpHTTP429 = regexp.MustCompile(`[Ee]rror\s+429\s+`)    // Error 429
	regexpHTTP503 = regexp.MustCompile(`[Ee]rror\s+503\s+`)    // Error 503
	regexpHTTP5xx = regexp.MustCompile(`[Ee]rror\s+5\d{2}\s+`) // Error 5xx

	// messages for checking genai errors
	msgQuotaExceeded   = `exceeded your current quota`
	msgModelOverloaded = `model is overloaded`
)

// IsQuotaExceeded checks if the provided error is a `*genai.APIError`
// with a status code 429 indicating that a quota limit has been exceeded.
func IsQuotaExceeded(err error) bool {
	ae, isAPIError := APIError(err)
	if isAPIError &&
		ae.Code == 429 && //nolint:gomnd // Standard HTTP status code
		strings.Contains(ae.Message, msgQuotaExceeded) {
		return true
	} else {
		errStr := err.Error()
		if regexpHTTP429.MatchString(errStr) &&
			strings.Contains(errStr, msgQuotaExceeded) {
			return true
		}
	}
	return false
}

// IsModelOverloaded checks if the provided error is a `*genai.APIError`
// with a status code 503 and a message indicating that the model is currently
// overloaded.
func IsModelOverloaded(err error) bool {
	ae, isAPIError := APIError(err)
	if isAPIError &&
		ae.Code == 503 && //nolint:gomnd // Standard HTTP status code
		strings.Contains(ae.Message, msgModelOverloaded) {
		return true
	} else {
		errStr := err.Error()
		if regexpHTTP503.MatchString(errStr) &&
			strings.Contains(errStr, msgModelOverloaded) {
			return true
		}
	}
	return false
}

// readMimeAndRecycle detects the MIME type from an io.Reader and returns the MIME type
// along with a new io.Reader that contains the original full content (including the bytes read for detection).
// This is crucial because mimetype.DetectReader consumes part of the reader.
// This is an internal helper function.
// See https://pkg.go.dev/github.com/gabriel-vasile/mimetype#example-package-DetectReader for the pattern.
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

// TextChunkOption provides configuration for the ChunkText function.
type TextChunkOption struct {
	ChunkSize                uint   // ChunkSize is the desired maximum size of each text chunk in bytes.
	OverlappedSize           uint   // OverlappedSize specifies how many bytes from the end of the previous chunk should be included at the beginning of the next chunk.
	KeepBrokenUTF8Characters bool   // KeepBrokenUTF8Characters, if true, preserves potentially broken UTF-8 characters at chunk boundaries. If false (default), ToValidUTF8 is used to replace them.
	EllipsesText             string // EllipsesText is a string (e.g., "...") to append to the beginning of subsequent chunks and the end of chunks that are not the final part of the text.
}

// ChunkedText holds the original text and the generated chunks.
type ChunkedText struct {
	Original string   // Original is the input text that was chunked.
	Chunks   []string // Chunks is a slice of strings, where each string is a chunk of the original text.
}

// ChunkText splits a given text into smaller chunks based on the provided options.
// This can be useful for processing large texts that might exceed model input limits.
// It supports overlapping chunks and handling of UTF-8 characters at boundaries.
//
// Parameters:
//   - text: The string to be chunked.
//   - opts: Optional TextChunkOption to customize chunking behavior. If not provided,
//     default chunk size, overlap size, and UTF-8 handling will be used.
//
// Returns:
//   - ChunkedText: A struct containing the original text and a slice of its chunks.
//   - error: An error if the configuration is invalid (e.g., overlappedSize >= chunkSize).
func ChunkText(text string, opts ...TextChunkOption) (ChunkedText, error) {
	opt := TextChunkOption{
		ChunkSize:      defaultChunkedTextLengthInBytes,    // Default chunk size.
		OverlappedSize: defaultOverlappedTextLengthInBytes, // Default overlap size.
		// KeepBrokenUTF8Characters defaults to false.
		// EllipsesText defaults to empty.
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
		return ChunkedText{}, fmt.Errorf(
			"overlapped size(= %d) must be less than chunk size(= %d)",
			overlappedSize,
			chunkSize,
		)
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

// for exiting an iterator with an error
func yieldErrorAndEndIterator[T any](err error) iter.Seq2[*T, error] {
	return func(yield func(*T, error) bool) {
		if !yield(nil, err) {
			return
		}
	}
}
