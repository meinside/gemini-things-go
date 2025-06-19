// convert.go

package gt

import (
	"encoding/base64"
	"encoding/json"
	"fmt"

	"github.com/mark3labs/mcp-go/mcp"
	"google.golang.org/genai"
)

// MCPToGeminiTools converts given []mcp.Tool to []genai.FunctionDeclaration.
func MCPToGeminiTools(
	from []mcp.Tool,
) (to []*genai.FunctionDeclaration, err error) {
	to = make([]*genai.FunctionDeclaration, len(from))

	for i, f := range from {
		to[i] = &genai.FunctionDeclaration{
			Name:        f.Name,
			Description: f.Description,
		}
		if marshalled, err := f.InputSchema.MarshalJSON(); err == nil {
			var schema map[string]any
			if err := json.Unmarshal(marshalled, &schema); err == nil {
				to[i].ParametersJsonSchema = schema
			} else {
				return nil, fmt.Errorf("could not convert json to map: %w", err)
			}
		} else {
			return nil, fmt.Errorf("could not convert input schema to json: %w", err)
		}
	}

	return to, nil
}

// MCPCallToolResultToGeminiPrompts converts given *mcp.CallToolResult to []Prompt.
func MCPCallToolResultToGeminiPrompts(
	from *mcp.CallToolResult,
) (to []Prompt, err error) {
	to = make([]Prompt, len(from.Content))

	for i, c := range from.Content {
		if ct, ok := c.(mcp.TextContent); ok {
			to[i] = PromptFromText(ct.Text)
		} else if ci, ok := c.(mcp.ImageContent); ok {
			if bytes, err := base64.RawStdEncoding.DecodeString(ci.Data); err == nil {
				to[i] = PromptFromBytes(bytes)
			} else {
				return nil, fmt.Errorf("failed to decode image from call tool result: %w", err)
			}
		} else if ca, ok := c.(mcp.AudioContent); ok {
			if bytes, err := base64.RawStdEncoding.DecodeString(ca.Data); err == nil {
				to[i] = PromptFromBytes(bytes)
			} else {
				return nil, fmt.Errorf("failed to decode audio from call tool result: %w", err)
			}
		} else {
			return nil, fmt.Errorf("unhandled content type from call tool result: %T", c)
		}
	}

	return to, nil
}
