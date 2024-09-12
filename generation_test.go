package gt

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"testing"

	"github.com/google/generative-ai-go/genai"
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

// TestGenerationStreamed tests various types of generations (streamed).
func TestGenerationStreamed(t *testing.T) {
	apiKey := mustHaveEnvVar(t, "API_KEY")

	client := NewClient(modelForTest, apiKey)

	// text-only prompt
	if err := client.GenerateStreamed(
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
		if err := client.GenerateStreamed(
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
	if err := client.GenerateStreamed(
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

	client := NewClient(modelForTest, apiKey)

	// text-only prompt
	if generated, err := client.Generate(
		context.TODO(),
		"What is the answer to life, the universe, and everything?",
		nil,
	); err != nil {
		t.Errorf("failed to generate from text prompt: %s", err)
	} else {
		log.Printf("generated: %s", prettify(generated))
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		if generated, err := client.Generate(
			context.TODO(),
			"What's the golang package name of this file? Can you give me a short sample code of using this file?",
			[]io.Reader{file},
		); err != nil {
			t.Errorf("failed to generate from text prompt and file: %s", err)
		} else {
			log.Printf("generated: %s", prettify(generated))
		}
	} else {
		t.Errorf("failed to open file for test: %s", err)
	}

	// prompt with bytes array
	if generated, err := client.Generate(
		context.TODO(),
		"Translate the text in the given file into English.",
		[]io.Reader{strings.NewReader("동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세")},
	); err != nil {
		t.Errorf("failed to generate from text prompt and bytes: %s", err)
	} else {
		log.Printf("generated: %s", prettify(generated))
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

	client := NewClient(modelForTest, apiKey)

	// prompt with function calls
	if err := client.GenerateStreamed(
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
				t.Fatalf("should not reach here")
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
