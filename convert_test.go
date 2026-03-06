// convert_test.go
//
// test cases for convert.go

package gt

import (
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// TestMCPToGeminiToolsWithMapSchema tests converting MCP tools with map[string]any InputSchema.
func TestMCPToGeminiToolsWithMapSchema(t *testing.T) {
	tools := []*mcp.Tool{
		{
			Name:        "test_tool",
			Description: "A test tool",
			InputSchema: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"query": map[string]any{
						"type":        "string",
						"description": "search query",
					},
				},
				"required": []string{"query"},
			},
		},
	}

	result, err := MCPToGeminiTools(tools)
	if err != nil {
		t.Fatalf("MCPToGeminiTools failed: %s", err)
	}

	if len(result) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result))
	}
	if result[0].Name != "test_tool" {
		t.Errorf("expected name 'test_tool', got '%s'", result[0].Name)
	}
	if result[0].Description != "A test tool" {
		t.Errorf("expected description 'A test tool', got '%s'", result[0].Description)
	}
	if result[0].ParametersJsonSchema == nil {
		t.Errorf("expected ParametersJsonSchema to be set")
	}
}

// TestMCPToGeminiToolsWithUnsupportedSchema tests that unsupported InputSchema types return an error.
func TestMCPToGeminiToolsWithUnsupportedSchema(t *testing.T) {
	tools := []*mcp.Tool{
		{
			Name:        "bad_tool",
			Description: "A tool with bad schema",
			InputSchema: 12345, // unsupported type
		},
	}

	_, err := MCPToGeminiTools(tools)
	if err == nil {
		t.Fatalf("expected error for unsupported InputSchema type, got nil")
	}
}

// TestMCPToGeminiToolsEmpty tests converting an empty slice of MCP tools.
func TestMCPToGeminiToolsEmpty(t *testing.T) {
	result, err := MCPToGeminiTools([]*mcp.Tool{})
	if err != nil {
		t.Fatalf("MCPToGeminiTools failed: %s", err)
	}
	if len(result) != 0 {
		t.Errorf("expected 0 tools, got %d", len(result))
	}
}

// TestMCPCallToolResultToGeminiPromptsText tests converting text content.
func TestMCPCallToolResultToGeminiPromptsText(t *testing.T) {
	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: "hello world"},
		},
	}

	prompts, err := MCPCallToolResultToGeminiPrompts(result)
	if err != nil {
		t.Fatalf("MCPCallToolResultToGeminiPrompts failed: %s", err)
	}

	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}

	tp, ok := prompts[0].(TextPrompt)
	if !ok {
		t.Fatalf("expected TextPrompt, got %T", prompts[0])
	}
	if tp.Text != "hello world" {
		t.Errorf("expected text 'hello world', got '%s'", tp.Text)
	}
}

// TestMCPCallToolResultToGeminiPromptsImage tests converting image content.
func TestMCPCallToolResultToGeminiPromptsImage(t *testing.T) {
	imageData := []byte{0x89, 0x50, 0x4E, 0x47} // PNG magic bytes (partial)

	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.ImageContent{Data: imageData, MIMEType: "image/png"},
		},
	}

	prompts, err := MCPCallToolResultToGeminiPrompts(result)
	if err != nil {
		t.Fatalf("MCPCallToolResultToGeminiPrompts failed: %s", err)
	}

	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}

	bp, ok := prompts[0].(BytesPrompt)
	if !ok {
		t.Fatalf("expected BytesPrompt, got %T", prompts[0])
	}
	if len(bp.Bytes) != len(imageData) {
		t.Errorf("expected %d bytes, got %d", len(imageData), len(bp.Bytes))
	}
}

// TestMCPCallToolResultToGeminiPromptsAudio tests converting audio content.
func TestMCPCallToolResultToGeminiPromptsAudio(t *testing.T) {
	audioData := []byte{0xFF, 0xFB, 0x90, 0x00} // MP3 frame header (partial)

	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.AudioContent{Data: audioData, MIMEType: "audio/mp3"},
		},
	}

	prompts, err := MCPCallToolResultToGeminiPrompts(result)
	if err != nil {
		t.Fatalf("MCPCallToolResultToGeminiPrompts failed: %s", err)
	}

	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}

	bp, ok := prompts[0].(BytesPrompt)
	if !ok {
		t.Fatalf("expected BytesPrompt, got %T", prompts[0])
	}
	if len(bp.Bytes) != len(audioData) {
		t.Errorf("expected %d bytes, got %d", len(audioData), len(bp.Bytes))
	}
}

// TestMCPCallToolResultToGeminiPromptsEmbeddedResource tests converting embedded resource content.
func TestMCPCallToolResultToGeminiPromptsEmbeddedResource(t *testing.T) {
	blobData := []byte("some file content")

	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.EmbeddedResource{
				Resource: &mcp.ResourceContents{
					URI:  "file:///test.txt",
					Blob: blobData,
				},
			},
		},
	}

	prompts, err := MCPCallToolResultToGeminiPrompts(result)
	if err != nil {
		t.Fatalf("MCPCallToolResultToGeminiPrompts failed: %s", err)
	}

	if len(prompts) != 1 {
		t.Fatalf("expected 1 prompt, got %d", len(prompts))
	}

	bp, ok := prompts[0].(BytesPrompt)
	if !ok {
		t.Fatalf("expected BytesPrompt, got %T", prompts[0])
	}
	if bp.Filename != "file:///test.txt" {
		t.Errorf("expected filename 'file:///test.txt', got '%s'", bp.Filename)
	}
}

// TestMCPCallToolResultToGeminiPromptsNilResource tests that nil embedded resource returns an error.
func TestMCPCallToolResultToGeminiPromptsNilResource(t *testing.T) {
	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.EmbeddedResource{Resource: nil},
		},
	}

	_, err := MCPCallToolResultToGeminiPrompts(result)
	if err == nil {
		t.Fatalf("expected error for nil embedded resource, got nil")
	}
}

// TestMCPCallToolResultToGeminiPromptsMultipleContents tests converting multiple content items.
func TestMCPCallToolResultToGeminiPromptsMultipleContents(t *testing.T) {
	result := &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: "first"},
			&mcp.TextContent{Text: "second"},
		},
	}

	prompts, err := MCPCallToolResultToGeminiPrompts(result)
	if err != nil {
		t.Fatalf("MCPCallToolResultToGeminiPrompts failed: %s", err)
	}

	if len(prompts) != 2 {
		t.Fatalf("expected 2 prompts, got %d", len(prompts))
	}

	for i, expected := range []string{"first", "second"} {
		tp, ok := prompts[i].(TextPrompt)
		if !ok {
			t.Fatalf("prompts[%d]: expected TextPrompt, got %T", i, prompts[i])
		}
		if tp.Text != expected {
			t.Errorf("prompts[%d]: expected text '%s', got '%s'", i, expected, tp.Text)
		}
	}
}
