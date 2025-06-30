// convert.go
//
// functions for converting things across protocols and/or libraries

package gt

import (
	"encoding/json"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

// MCPToGeminiTools converts given []mcp.Tool to []genai.FunctionDeclaration.
func MCPToGeminiTools(
	from []*mcp.Tool,
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
		switch t := c.(type) {
		case *mcp.TextContent:
			to[i] = PromptFromText(t.Text)
		case *mcp.ImageContent:
			to[i] = PromptFromBytes(t.Data)
		case *mcp.AudioContent:
			to[i] = PromptFromBytes(t.Data)
		case *mcp.EmbeddedResource:
			if t.Resource != nil {
				to[i] = PromptFromBytesWithName(t.Resource.Blob, t.Resource.URI)
			} else {
				return nil, fmt.Errorf("embedded resource is nil")
			}
		default:
			return nil, fmt.Errorf("unhandled content type from call tool result: %T", c)
		}
	}

	return to, nil
}
