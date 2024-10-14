package gt

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/iterator"
)

const (
	modelForTest = `gemini-1.5-flash`
)

// check and return environment variable for given key
func mustHaveEnvVar(t *testing.T, key string) string {
	if value, exists := os.LookupEnv(key); !exists {
		t.Fatalf("no environment variable: %s", key)
	} else {
		return value
	}
	return ""
}

// TestGenerationIterated tests various types of generations (iterator).
func TestGenerationIterated(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc := NewClient(modelForTest, apiKey)

	// text-only prompt
	if iterated, err := gtc.GenerateStreamIterated(
		context.TODO(),
		"What is the answer to life, the universe, and everything?",
		nil,
	); err != nil {
		t.Errorf("failed to generate from text prompt: %s", err)
	} else {
		log.Printf(">>> generated iterator: %s", prettify(iterated))

		for {
			if it, err := iterated.Next(); err == nil {
				log.Printf(">>> iterating response: %s", prettify(it))
			} else {
				if err != iterator.Done {
					t.Errorf("failed to iterate stream: %s", err)
				}
				break
			}
		}
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		if iterated, err := gtc.GenerateStreamIterated(
			context.TODO(),
			"What's the golang package name of this file? Can you give me a short sample code of using this file?",
			[]io.Reader{file},
		); err != nil {
			t.Errorf("failed to generate from text prompt and file: %s", err)
		} else {
			log.Printf(">>> generated iterator: %s", prettify(iterated))

			for {
				if it, err := iterated.Next(); err == nil {
					log.Printf(">>> iterating response: %s", prettify(it))
				} else {
					if err != iterator.Done {
						t.Errorf("failed to iterate stream: %s", err)
					}
					break
				}
			}
		}
	} else {
		t.Errorf("failed to open file for test: %s", err)
	}

	// prompt with bytes array
	if iterated, err := gtc.GenerateStreamIterated(
		context.TODO(),
		"Translate the text in the given file into English.",
		[]io.Reader{strings.NewReader("동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세")},
	); err != nil {
		t.Errorf("failed to generate from text prompt and bytes: %s", err)
	} else {
		log.Printf(">>> generated iterator: %s", prettify(iterated))

		for {
			if it, err := iterated.Next(); err == nil {
				log.Printf(">>> iterating response: %s", prettify(it))
			} else {
				if err != iterator.Done {
					t.Errorf("failed to iterate stream: %s", err)
				}
				break
			}
		}
	}
}

// TestGenerationStreamed tests various types of generations (streamed).
func TestGenerationStreamed(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")
	isVerbose := os.Getenv("VERBOSE") == "true"

	gtc := NewClient(modelForTest, apiKey)
	gtc.Verbose = isVerbose

	// text-only prompt
	if err := gtc.GenerateStreamed(
		context.TODO(),
		"What is the answer to life, the universe, and everything?",
		nil,
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				fmt.Print(*data.TextDelta) // print text stream
			} else if data.NumTokens != nil {
				log.Printf(">>> input tokens: %d, output tokens: %d", data.NumTokens.Input, data.NumTokens.Output)
			} else if data.FinishReason != nil {
				t.Errorf("generation was finished with reason: %s", data.FinishReason.String())
			} else if data.Error != nil {
				t.Errorf("error while processing text generation: %s", data.Error)
			}
		},
		nil,
	); err != nil {
		t.Errorf("failed to generate from text prompt: %s", err)
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			"What's the golang package name of this file? Can you give me a short sample code of using this file?",
			[]io.Reader{file},
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					fmt.Print(*data.TextDelta) // print text stream
				} else if data.NumTokens != nil {
					log.Printf(">>> input tokens: %d, output tokens: %d", data.NumTokens.Input, data.NumTokens.Output)
				} else if data.FinishReason != nil {
					t.Errorf("generation was finished with reason: %s", data.FinishReason.String())
				} else if data.Error != nil {
					t.Errorf("error while processing generation with files: %s", data.Error)
				}
			},
			nil,
		); err != nil {
			t.Errorf("failed to generate from text prompt and file: %s", err)
		}
	} else {
		t.Errorf("failed to open file for test: %s", err)
	}

	// prompt with bytes array
	if err := gtc.GenerateStreamed(
		context.TODO(),
		"Translate the text in the given file into English.",
		[]io.Reader{strings.NewReader("동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세")},
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				fmt.Print(*data.TextDelta) // print text stream
			} else if data.NumTokens != nil {
				log.Printf(">>> input tokens: %d, output tokens: %d", data.NumTokens.Input, data.NumTokens.Output)
			} else if data.FinishReason != nil {
				t.Errorf("generation was finished with reason: %s", data.FinishReason.String())
			} else if data.Error != nil {
				t.Errorf("error while processing generation with bytes: %s", data.Error)
			}
		},
		nil,
	); err != nil {
		t.Errorf("failed to generate from text prompt and bytes: %s", err)
	}
}

// TestGenerationNonStreamed tests various types of generations (streamed).
func TestGenerationNonStreamed(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc := NewClient(modelForTest, apiKey)

	// text-only prompt
	if generated, err := gtc.Generate(
		context.TODO(),
		"What is the answer to life, the universe, and everything?",
		nil,
	); err != nil {
		t.Errorf("failed to generate from text prompt: %s", err)
	} else {
		log.Printf(">>> generated: %s", prettify(generated))
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		if generated, err := gtc.Generate(
			context.TODO(),
			"What's the golang package name of this file? Can you give me a short sample code of using this file?",
			[]io.Reader{file},
		); err != nil {
			t.Errorf("failed to generate from text prompt and file: %s", err)
		} else {
			log.Printf(">>> generated: %s", prettify(generated))
		}
	} else {
		t.Errorf("failed to open file for test: %s", err)
	}

	// prompt with bytes array
	if generated, err := gtc.Generate(
		context.TODO(),
		"Translate the text in the given file into English.",
		[]io.Reader{strings.NewReader("동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세")},
	); err != nil {
		t.Errorf("failed to generate from text prompt and bytes: %s", err)
	} else {
		log.Printf(">>> generated: %s", prettify(generated))
	}
}

// TestGenerationWithFunctionCall tests various types of generations with function call declarations.
func TestGenerationWithFunctionCall(t *testing.T) {
	const (
		fnNameGenerateImages      = "generate_images"
		fnDescGenerateImages      = `This function generates beautiful images by extracting and optimizing positive and/or negative prompts from the text given by the user.`
		fnParamNamePositivePrompt = "positive_prompt"
		fnParamDescPositivePrompt = `A text prompt for generating images with image generation models(eg. Stable Diffusion, or DALL-E). This prompt describes what to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
		fnParamNameNegativePrompt = "negative_prompt"
		fnParamDescNegativePrompt = `A text prompt for image generation models(eg. Stable Diffusion, or DALL-E) to define what not to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
	)

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc := NewClient(modelForTest, apiKey)

	// prompt with function calls
	if err := gtc.GenerateStreamed(
		context.TODO(),
		`Please generate an image which shows a man standing in front of a vast dessert. The man is watching an old pyramid completely destroyed by a giant sandstorm. The mood is sad and gloomy.`,
		nil,
		func(data StreamCallbackData) {
			if data.FunctionCall != nil {
				if data.FunctionCall.Name == fnNameGenerateImages {
					positivePrompt, _ := FuncArg[string](data.FunctionCall.Args, fnParamNamePositivePrompt)
					negativePrompt, _ := FuncArg[string](data.FunctionCall.Args, fnParamNameNegativePrompt)

					if positivePrompt != nil {
						log.Printf(">>> positive prompt: %s", *positivePrompt)

						if negativePrompt != nil {
							log.Printf(">>> negative prompt: %s", *negativePrompt)
						}
					} else {
						t.Errorf("failed to parse function args (%s)", prettify(data.FunctionCall.Args))
					}
				} else {
					t.Errorf("function name does not match '%s': %s", fnNameGenerateImages, prettify(data.FunctionCall))
				}
			} else if data.NumTokens != nil {
				log.Printf(">>> input tokens: %d, output tokens: %d", data.NumTokens.Input, data.NumTokens.Output)
			} else if data.Error != nil {
				t.Errorf("failed to generate with function calls: %s", data.Error)
			} else {
				if data.TextDelta == nil || len(*data.TextDelta) > 0 { // FIXME: sometimes only `data.TextDelta` is returned as ""
					t.Fatalf("should not reach here, data: %s", prettify(data))
				}
			}
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        fnNameGenerateImages,
							Description: fnDescGenerateImages,
							Parameters: &genai.Schema{
								Type:     genai.TypeObject,
								Nullable: false,
								Properties: map[string]*genai.Schema{
									fnParamNamePositivePrompt: {
										Type:        genai.TypeString,
										Description: fnParamDescPositivePrompt,
										Nullable:    false,
									},
									fnParamNameNegativePrompt: {
										Type:        genai.TypeString,
										Description: fnParamDescNegativePrompt,
										Nullable:    true,
									},
								},
								Required: []string{fnParamNamePositivePrompt, fnParamNameNegativePrompt},
							},
						},
					},
				},
			},
		},
	); err != nil {
		t.Errorf("failed to generate with function calls: %s", err)
	}
}

// TestGenerationWithStructuredOutput tests generations with structured outputs.
func TestGenerationWithStructuredOutput(t *testing.T) {
	const (
		paramNamePositivePrompt = "positive_prompt"
		paramDescPositivePrompt = `A text prompt for generating images with image generation models(eg. Stable Diffusion, or DALL-E). This prompt describes what to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
		paramNameNegativePrompt = "negative_prompt"
		paramDescNegativePrompt = `A text prompt for image generation models(eg. Stable Diffusion, or DALL-E) to define what not to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
	)

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc := NewClient(modelForTest, apiKey)

	// prompt with function calls
	if generated, err := gtc.Generate(
		context.TODO(),
		`Extract and optimize positive and/or negative prompts from the following text for generating beautiful images: Please generate an image which shows a man standing in front of a vast dessert. The man is watching an old pyramid completely destroyed by a giant sandstorm. The mood is sad and gloomy.`,
		nil,
		&GenerationOptions{
			Config: &genai.GenerationConfig{
				ResponseMIMEType: "application/json",
				ResponseSchema: &genai.Schema{
					Type:     genai.TypeObject,
					Nullable: false,
					Properties: map[string]*genai.Schema{
						paramNamePositivePrompt: {
							Type:        genai.TypeString,
							Description: paramDescPositivePrompt,
							Nullable:    false,
						},
						paramNameNegativePrompt: {
							Type:        genai.TypeString,
							Description: paramDescNegativePrompt,
							Nullable:    true,
						},
					},
					Required: []string{paramNamePositivePrompt, paramNameNegativePrompt},
				},
			},
		},
	); err == nil {
		for _, part := range generated.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				var args map[string]any
				if err := json.Unmarshal([]byte(text), &args); err == nil {
					positivePrompt, _ := FuncArg[string](args, paramNamePositivePrompt)
					negativePrompt, _ := FuncArg[string](args, paramNameNegativePrompt)

					if positivePrompt != nil {
						log.Printf(">>> positive prompt: %s", *positivePrompt)

						if negativePrompt != nil {
							log.Printf(">>> negative prompt: %s", *negativePrompt)
						}
					} else {
						t.Errorf("failed to parse structured output (%s)", prettify(args))
					}
				} else {
					t.Errorf("failed to parse structured output text '%s': %s", text, err)
				}
			} else {
				t.Errorf("wrong type of generated part: (%T) %s", part, prettify(part))
			}
		}
	} else {
		t.Errorf("failed to generate with structured output: %s", err)
	}
}

// TestGenerationWithCodeExecution tests generations with code executions.
func TestGenerationWithCodeExecution(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc := NewClient(modelForTest, apiKey)

	// prompt with function calls
	if generated, err := gtc.Generate(
		context.TODO(),
		`Generate unique 6 numbers between 1 and 45. Make sure there is no duplicated number, and list the numbers in ascending order.`,
		nil,
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					CodeExecution: &genai.CodeExecution{},
				},
			},
		},
	); err == nil {
		for _, part := range generated.Candidates[0].Content.Parts {
			if text, ok := part.(genai.Text); ok {
				log.Printf(">>> generated text: %s", text)
			} else if code, ok := part.(*genai.ExecutableCode); ok {
				log.Printf(">>> executable code (%s):\n%s", code.Language.String(), code.Code)
			} else if result, ok := part.(*genai.CodeExecutionResult); ok {
				if result.Outcome != genai.CodeExecutionResultOutcomeOK {
					t.Errorf("code execution failed: %s", prettify(result))
				} else {
					log.Printf(">>> code output: %s", result.Output)
				}
			} else {
				t.Errorf("wrong type of generated part: (%T) %s", part, prettify(part))
			}
		}
	} else {
		t.Errorf("failed to generate with code execution: %s", err)
	}
}

// TestGenerationWithHistory tests generations with history.
func TestGenerationWithHistory(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")

	client := NewClient(modelForTest, apiKey)

	// text-only prompt with history (streamed)
	if err := client.GenerateStreamed(
		context.TODO(),
		"Isn't that 42?",
		nil,
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				fmt.Print(*data.TextDelta) // print text stream
			} else if data.NumTokens != nil {
				log.Printf(">>> input tokens: %d, output tokens: %d", data.NumTokens.Input, data.NumTokens.Output)
			} else if data.FinishReason != nil {
				t.Errorf("generation was finished with reason: %s", data.FinishReason.String())
			} else if data.Error != nil {
				t.Errorf("error while processing text generation: %s", data.Error)
			}
		},
		&GenerationOptions{
			History: []*genai.Content{
				{
					Role: "user",
					Parts: []genai.Part{
						genai.Text("What is the answer to life, the universe, and everything?"),
					},
				},
				{
					Role: "model",
					Parts: []genai.Part{
						genai.Text("43."),
					},
				},
			},
		},
	); err != nil {
		t.Errorf("failed to generate from text prompt and history: %s", err)
	}

	// text-only prompt with history
	if generated, err := client.Generate(
		context.TODO(),
		"Isn't that 42?",
		nil,
		&GenerationOptions{
			History: []*genai.Content{
				{
					Role: "user",
					Parts: []genai.Part{
						genai.Text("What is the answer to life, the universe, and everything?"),
					},
				},
				{
					Role: "model",
					Parts: []genai.Part{
						genai.Text("43."),
					},
				},
			},
		},
	); err != nil {
		t.Errorf("failed to generate from text prompt and history: %s", err)
	} else {
		log.Printf(">>> generated: %s", prettify(generated))
	}
}
