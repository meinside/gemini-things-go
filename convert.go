// convert.go
//
// functions for converting things across protocols and/or libraries

package gt

import (
	"encoding/json"
	"fmt"

	"github.com/google/jsonschema-go/jsonschema"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"google.golang.org/genai"
)

// MCPToGeminiTools converts given []mcp.Tool to []genai.FunctionDeclaration.
//
// InputSchema value of each mcp.Tool should be in type: `jsonschema.Schema` or `map[string]any`.
func MCPToGeminiTools(
	from []*mcp.Tool,
) (to []*genai.FunctionDeclaration, err error) {
	to = make([]*genai.FunctionDeclaration, len(from))

	for i, f := range from {
		to[i] = &genai.FunctionDeclaration{
			Name:        f.Name,
			Description: f.Description,
		}

		if inputSchema, ok := f.InputSchema.(map[string]any); ok {
			to[i].ParametersJsonSchema = inputSchema
			continue
		}

		schema, err := marshalJsonSchema(f.InputSchema)
		if err != nil {
			return nil, fmt.Errorf("tools[%d].InputSchema: %w", i, err)
		}
		if schema != nil {
			to[i].ParametersJsonSchema = schema
			continue
		}

		return nil, fmt.Errorf("tools[%d].InputSchema is not in type `jsonschema.Schema` or `map[string]any`: %T", i, f.InputSchema)
	}

	return to, nil
}

// marshalJsonSchema tries to marshal a jsonschema.Schema or *jsonschema.Schema
// into a map[string]any. Returns (nil, nil) if the input is neither type.
func marshalJsonSchema(input any) (map[string]any, error) {
	type jsonMarshaler interface {
		MarshalJSON() ([]byte, error)
	}

	var m jsonMarshaler
	switch v := input.(type) {
	case jsonschema.Schema:
		m = &v
	case *jsonschema.Schema:
		m = v
	default:
		return nil, nil
	}

	marshalled, err := m.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("could not convert input schema to json: %w", err)
	}

	var schema map[string]any
	if err := json.Unmarshal(marshalled, &schema); err != nil {
		return nil, fmt.Errorf("could not convert json to map: %w", err)
	}

	return schema, nil
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
