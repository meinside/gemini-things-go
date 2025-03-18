package gt

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

const (
	// FIXME: context caching is not working for gemini-2.0 yet
	modelForContextCaching = `gemini-1.5-flash-002` // NOTE: context caching is only available for stable versions of the model

	modelForTextGeneration  = `gemini-2.0-flash-001`
	modelForImageGeneration = `gemini-2.0-flash-exp`
	// modelForImageGeneration = `gemini-2.0-flash-exp-image-generation`

	modelForTextGenerationWithGrounding             = `gemini-2.0-flash-001`
	modelForTextGenerationWithGoogleSearchRetrieval = `gemini-1.5-flash` // FIXME: Google Search retrieval is only compatible with Gemini 1.5 models

	modelForEmbeddings = `text-embedding-004`
)

// flag for verbose log
var _isVerbose bool

func TestMain(m *testing.M) {
	_isVerbose = os.Getenv("VERBOSE") == "true"

	os.Exit(m.Run())
}

// sleep between each test case to not be rate limited by the API
func sleepForNotBeingRateLimited() {
	verbose(">>> sleeping for a while...")

	time.Sleep(10 * time.Second)
}

// print given message if verbose mode is enabled
func verbose(format string, v ...any) {
	if _isVerbose {
		log.Printf(format, v...)
	}
}

// check and return environment variable for given key
func mustHaveEnvVar(t *testing.T, key string) string {
	if value, exists := os.LookupEnv(key); !exists {
		t.Fatalf("no environment variable: %s", key)
	} else {
		return value
	}
	return ""
}

// TestContextCaching tests context caching and generation with the cached context.
func TestContextCaching(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForContextCaching)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.SetSystemInstructionFunc(nil) // FIXME: error: 'client error. Code: 400, Message: CachedContent can not be used with GenerateContent request setting system_instruction, tools or tool_config.'
	gtc.DeleteFilesOnClose = true
	gtc.DeleteCachesOnClose = true
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// open a file for testing
	file, err := os.Open("./client.go")
	if err != nil {
		t.Fatalf("failed to open file for caching context: %s", err)
	}
	defer file.Close()

	cachedSystemInstruction := `You are an arrogant and unhelpful chat bot who answers really shortly with a very sarcastic manner.`
	cachedContextDisplayName := `cached-context-for-test`

	// cache context,
	if cachedContextName, err := gtc.CacheContext(
		context.TODO(),
		&cachedSystemInstruction,
		[]Prompt{
			PromptFromFile("client.go", file),
		},
		nil,
		nil,
		&cachedContextDisplayName,
	); err != nil {
		t.Errorf("failed to cache context: %s", ErrToStr(err))
	} else {
		// generate iterated with the cached context
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			[]Prompt{
				PromptFromText("What is this file?"),
			},
			&GenerationOptions{
				CachedContent: cachedContextName,
			},
		) {
			if err != nil {
				t.Errorf("generation with cached context failed: %s", ErrToStr(err))
			} else {
				verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
			}
		}

		// generate streamed response with the cached context
		if err := gtc.GenerateStreamed(
			context.TODO(),
			[]Prompt{
				PromptFromText("Can you give me any insight about this file?"),
			},
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				} else if data.FinishReason != nil {
					t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
				} else if data.Error != nil {
					t.Errorf("error while processing generation with cached context: %s", data.Error)
				}
			},
			&GenerationOptions{
				CachedContent: cachedContextName,
			},
		); err != nil {
			t.Errorf("generation with cached context failed: %s", ErrToStr(err))
		}

		// generate with the cached context
		if generated, err := gtc.Generate(
			context.TODO(),
			[]Prompt{
				PromptFromText("How many standard golang libraries are used in this source code?"),
			},
			&GenerationOptions{
				CachedContent: cachedContextName,
			},
		); err != nil {
			t.Errorf("generation with cached context failed: %s", ErrToStr(err))
		} else {
			var promptTokenCount int32 = 0
			var cachedContentTokenCount int32 = 0
			if generated.UsageMetadata != nil {
				if generated.UsageMetadata.PromptTokenCount != nil {
					promptTokenCount = *generated.UsageMetadata.PromptTokenCount
				}
				if generated.UsageMetadata.CachedContentTokenCount != nil {
					cachedContentTokenCount = *generated.UsageMetadata.CachedContentTokenCount
				}
			}

			verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
				promptTokenCount,
				generated.UsageMetadata.TotalTokenCount-promptTokenCount,
				cachedContentTokenCount,
			)

			verbose(">>> generated: %s", prettify(generated.Candidates[0].Content.Parts[0]))
		}
	}

	// list all cached contexts
	if _, err := gtc.ListAllCachedContexts(context.TODO()); err != nil {
		t.Errorf("failed to list all cached contexts: %s", ErrToStr(err))
	}

	// NOTE: caches and files will be deleted on close
}

// TestGenerationIterated tests various types of generations (iterator).
func TestGenerationIterated(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt
	for it, err := range gtc.GenerateStreamIterated(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
	) {
		if err != nil {
			t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
		} else {
			verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
		}
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		defer file.Close()

		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			[]Prompt{
				PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`),
				PromptFromFile("client.go", file),
			},
		) {
			if err != nil {
				t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
			} else {
				verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
			}
		}
	} else {
		t.Errorf("failed to open file for test: %s", err)
	}

	// prompt with bytes array
	for it, err := range gtc.GenerateStreamIterated(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
	) {
		if err != nil {
			t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
		} else {
			verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
		}
	}

	// NOTE: files will be deleted on close
}

// TestGenerationStreamed tests various types of generations (streamed).
func TestGenerationStreamed(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt
	if err := gtc.GenerateStreamed(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				if _isVerbose {
					fmt.Print(*data.TextDelta) // print text stream
				}
			} else if data.NumTokens != nil {
				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
			} else if data.FinishReason != nil {
				t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
			} else if data.Error != nil {
				t.Errorf("error while processing text generation: %s", data.Error)
			}
		},
	); err != nil {
		t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
	}

	// prompt with files
	if file, err := os.Open("./client.go"); err == nil {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			[]Prompt{
				PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`),
				PromptFromFile("client.go", file),
			},
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				} else if data.FinishReason != nil {
					t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
				} else if data.Error != nil {
					t.Errorf("error while processing generation with files: %s", data.Error)
				}
			},
		); err != nil {
			t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
		}
	} else {
		t.Errorf("failed to open file for generation: %s", err)
	}

	// prompt with bytes array (streamed)
	if err := gtc.GenerateStreamed(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				if _isVerbose {
					fmt.Print(*data.TextDelta) // print text stream
				}
			} else if data.NumTokens != nil {
				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
			} else if data.FinishReason != nil {
				t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
			} else if data.Error != nil {
				t.Errorf("error while processing generation with bytes: %s", data.Error)
			}
		},
	); err != nil {
		t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
	}

	// NOTE: files will be deleted on close
}

// TestGenerationNonStreamed tests various types of generations.
func TestGenerationNonStreamed(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt
	if generated, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
	); err != nil {
		t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
	} else {
		verbose(">>> generated: %s", prettify(generated.Candidates[0].Content.Parts[0]))
	}

	// prompt with files (non-streamed)
	if file, err := os.Open("./client.go"); err == nil {
		if generated, err := gtc.Generate(
			context.TODO(),
			[]Prompt{
				PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`),
				PromptFromFile("client.go", file),
			},
		); err != nil {
			t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
		} else {
			var promptTokenCount int32 = 0
			var candidatesTokenCount int32 = 0
			var cachedContentTokenCount int32 = 0
			if generated.UsageMetadata != nil {
				if generated.UsageMetadata.PromptTokenCount != nil {
					promptTokenCount = *generated.UsageMetadata.PromptTokenCount
				}
				if generated.UsageMetadata.CandidatesTokenCount != nil {
					candidatesTokenCount = *generated.UsageMetadata.CandidatesTokenCount
				}
				if generated.UsageMetadata.CachedContentTokenCount != nil {
					cachedContentTokenCount = *generated.UsageMetadata.CachedContentTokenCount
				}
			}

			verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
				promptTokenCount,
				candidatesTokenCount,
				cachedContentTokenCount,
			)

			verbose(">>> generated: %s", prettify(generated.Candidates[0].Content.Parts[0]))
		}
	} else {
		t.Errorf("failed to open file for generation: %s", err)
	}

	// prompt with bytes array (non-streamed)
	if generated, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
	); err != nil {
		t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
	} else {
		var promptTokenCount int32 = 0
		var candidatesTokenCount int32 = 0
		var cachedContentTokenCount int32 = 0
		if generated.UsageMetadata != nil {
			if generated.UsageMetadata.PromptTokenCount != nil {
				promptTokenCount = *generated.UsageMetadata.PromptTokenCount
			}
			if generated.UsageMetadata.CandidatesTokenCount != nil {
				candidatesTokenCount = *generated.UsageMetadata.CandidatesTokenCount
			}
			if generated.UsageMetadata.CachedContentTokenCount != nil {
				cachedContentTokenCount = *generated.UsageMetadata.CachedContentTokenCount
			}
		}

		verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
			promptTokenCount,
			candidatesTokenCount,
			cachedContentTokenCount,
		)

		verbose(">>> generated: %s", prettify(generated.Candidates[0].Content.Parts[0]))
	}

	// NOTE: files will be deleted on close
}

// TestGenerationWithFileConverter tests generations with custom file converters.
func TestGenerationWithFileConverter(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	// set custom file converters
	gtc.SetFileConverter(
		"application/x-ndjson", // for: 'application/jsonl' (application/x-ndjson)
		func(bs []byte) ([]byte, string, error) {
			// NOTE: a simple JSONL -> CSV converter
			type record struct {
				Name   string `json:"name"`
				Age    int    `json:"age"`
				Gender string `json:"gender"`
			}
			converted := strings.Builder{}
			converted.Write([]byte("name,age,gender\n"))
			for _, line := range strings.Split(string(bs), "\n") {
				var decoded record
				if err := json.Unmarshal([]byte(line), &decoded); err == nil {
					converted.Write(fmt.Appendf(nil, `"%s",%d,"%s"\n`, decoded.Name, decoded.Age, decoded.Gender))
				}
			}
			return []byte(converted.String()), "text/csv", nil
		},
	)
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	jsonlForTest := `{"name": "John Doe", "age": 45, "gender": "m"}
{"name": "Janet Doe", "age": 42, "gender": "f"}
{"name": "Jane Doe", "age": 15, "gender": "f"}`

	if generated, err := gtc.Generate(context.TODO(), []Prompt{
		PromptFromText(`Infer the relationships between the characters from the given information.`),
		PromptFromBytes([]byte(jsonlForTest)),
	}, &GenerationOptions{}); err != nil {
		t.Errorf("generation with file converter failed: %s", ErrToStr(err))
	} else {
		var promptTokenCount int32 = 0
		var candidatesTokenCount int32 = 0
		var cachedContentTokenCount int32 = 0
		if generated.UsageMetadata != nil {
			if generated.UsageMetadata.PromptTokenCount != nil {
				promptTokenCount = *generated.UsageMetadata.PromptTokenCount
			}
			if generated.UsageMetadata.CandidatesTokenCount != nil {
				candidatesTokenCount = *generated.UsageMetadata.CandidatesTokenCount
			}
			if generated.UsageMetadata.CachedContentTokenCount != nil {
				cachedContentTokenCount = *generated.UsageMetadata.CachedContentTokenCount
			}
		}

		verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
			promptTokenCount,
			candidatesTokenCount,
			cachedContentTokenCount,
		)

		verbose(">>> generated with file converter: %s", prettify(generated.Candidates[0].Content.Parts[0]))
	}
}

// TestGenerationWithFunctionCall tests various types of generations with function call declarations.
func TestGenerationWithFunctionCall(t *testing.T) {
	sleepForNotBeingRateLimited()

	// function declarations
	const (
		// for extracting positive/negative prompts for image generation
		fnNameExtractPrompts      = "extract_prompts_for_image_generation"
		fnDescExtractPrompts      = `This function extracts positive and/or negative prompts from the text given by the user which will be used for generating images.`
		fnParamNamePositivePrompt = "positive_prompt"
		fnParamDescPositivePrompt = `A text prompt for generating images with image generation models(eg. Stable Diffusion, or DALL-E). This prompt describes what to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
		fnParamNameNegativePrompt = "negative_prompt"
		fnParamDescNegativePrompt = `A text prompt for image generation models(eg. Stable Diffusion, or DALL-E) to define what not to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`

		// for returning the result of image generation
		fnNameImageGenerationFinished    = "image_generation_finished"
		fnDescImageGenerationFinished    = `This function is called when the image generation finishes with the given parameters.`
		fnParamNameGeneratedSuccessfully = "success"
		fnParamDescGeneratedSuccessfully = "If the generation was successful or not."
		fnParamNameGeneratedSize         = "size"
		fnParamDescGeneratedSize         = `The size of the generated image file in bytes.`
		fnParamNameGeneratedResolution   = "resolution"
		fnParamDescGeneratedResolution   = `The resolution of the generated image file.`
		fnParamNameGeneratedFilepath     = "filepath"
		fnParamDescGeneratedFilepath     = "The filepath of the generated image file."
	)
	fnDeclarations := []*genai.FunctionDeclaration{
		{
			Name:        fnNameExtractPrompts,
			Description: fnDescExtractPrompts,
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
				Required: []string{
					fnParamNamePositivePrompt,
					fnParamNameNegativePrompt,
				},
			},
		},
		{
			Name:        fnNameImageGenerationFinished,
			Description: fnDescImageGenerationFinished,
			Parameters: &genai.Schema{
				Type:     genai.TypeObject,
				Nullable: false,
				Properties: map[string]*genai.Schema{
					fnParamNameGeneratedSuccessfully: {
						Type:        genai.TypeBoolean,
						Description: fnParamDescGeneratedSuccessfully,
						Nullable:    false,
					},
					fnParamNameGeneratedSize: {
						Type:        genai.TypeNumber,
						Description: fnParamDescGeneratedSize,
						Nullable:    true,
					},
					fnParamNameGeneratedResolution: {
						Type:        genai.TypeString,
						Description: fnParamDescGeneratedResolution,
						Nullable:    true,
					},
					fnParamNameGeneratedFilepath: {
						Type:        genai.TypeString,
						Description: fnParamDescGeneratedFilepath,
						Nullable:    true,
					},
				},
				Required: []string{
					fnParamNameGeneratedSuccessfully,
					fnParamNameGeneratedSize,
					fnParamNameGeneratedResolution,
					fnParamNameGeneratedFilepath,
				},
			},
		},
	}

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	const prompt = `Generate an image file which shows a man standing in front of a vast dessert. The man is watching an old pyramid completely destroyed by a giant sandstorm. The mood should be sad and gloomy.`

	// prompt with function calls (streamed)
	if err := gtc.GenerateStreamed(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		func(data StreamCallbackData) {
			if data.FunctionCall != nil {
				if data.FunctionCall.Name == fnNameExtractPrompts {
					positivePrompt, _ := FuncArg[string](data.FunctionCall.Args, fnParamNamePositivePrompt)
					negativePrompt, _ := FuncArg[string](data.FunctionCall.Args, fnParamNameNegativePrompt)

					if positivePrompt != nil {
						verbose(">>> positive prompt: %s", *positivePrompt)

						if negativePrompt != nil {
							verbose(">>> negative prompt: %s", *negativePrompt)
						}

						pastGenerations := []genai.Content{
							{
								Parts: []*genai.Part{
									genai.NewPartFromText(prompt),
								},
								Role: "user",
							},
							{
								Parts: []*genai.Part{
									genai.NewPartFromFunctionCall(data.FunctionCall.Name, map[string]any{
										fnParamNamePositivePrompt: positivePrompt,
										fnParamNameNegativePrompt: negativePrompt,
									}),
								},
								Role: "model",
							},
							{
								Parts: []*genai.Part{
									// NOTE:
									// run your own function with the parameters returned from function call,
									// then send a function response built with the result of your function.
									genai.NewPartFromFunctionResponse(fnNameImageGenerationFinished, map[string]any{
										fnParamNameGeneratedSuccessfully: true,
										fnParamNameGeneratedSize:         424242,
										fnParamNameGeneratedResolution:   "800x800",
										fnParamNameGeneratedFilepath:     "/home/marvin/generated.jpg",
									}),
								},
								Role: "user",
							},
						}

						// generate again with a function response
						if err := gtc.GenerateStreamed(
							context.TODO(),
							[]Prompt{},
							func(data StreamCallbackData) {
								if data.TextDelta != nil {
									verbose(">>> generated from function response: %s", *data.TextDelta)
								}
							},
							&GenerationOptions{
								Tools: []*genai.Tool{
									{
										FunctionDeclarations: fnDeclarations,
									},
								},

								History: pastGenerations,
							},
						); err != nil {
							t.Errorf("failed to generate with function response: %s", ErrToStr(err))
						}
					} else {
						t.Errorf("failed to parse function args (%s)", prettify(data.FunctionCall.Args))
					}
				} else {
					t.Errorf("function name does not match '%s': %s", fnNameExtractPrompts, prettify(data.FunctionCall))
				}
			} else if data.NumTokens != nil {
				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
			} else if data.Error != nil {
				t.Errorf("generation with function calls failed: %s", data.Error)
			} else {
				if data.TextDelta == nil || len(*data.TextDelta) > 0 { // FIXME: sometimes only `data.TextDelta` is returned as ""
					t.Fatalf("should not reach here; data: %s", prettify(data))
				}
			}
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: fnDeclarations,
				},
			},
			ToolConfig: &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode: genai.FunctionCallingConfigModeAny,
					AllowedFunctionNames: []string{
						fnNameExtractPrompts,
					},
				},
			},
		},
	); err != nil {
		t.Errorf("generation with function calls failed: %s", ErrToStr(err))
	}
}

// TestGenerationWithStructuredOutput tests generations with structured outputs.
func TestGenerationWithStructuredOutput(t *testing.T) {
	sleepForNotBeingRateLimited()

	const (
		paramNamePositivePrompt = "positive_prompt"
		paramDescPositivePrompt = `A text prompt for generating images with image generation models(eg. Stable Diffusion, or DALL-E). This prompt describes what to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
		paramNameNegativePrompt = "negative_prompt"
		paramDescNegativePrompt = `A text prompt for image generation models(eg. Stable Diffusion, or DALL-E) to define what not to be included in the resulting images. It should be in English and optimized following the image generation models' prompt guides.`
	)

	const prompt = `Extract and optimize positive and/or negative prompts from the following text for generating beautiful images: "Please generate an image which shows a man standing in front of a vast dessert. The man is watching an old pyramid completely destroyed by a giant sandstorm. The mood is sad and gloomy".`

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// prompt with function calls (non-streamed)
	if generated, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
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
			if len(part.Text) > 0 {
				var args map[string]any
				if err := json.Unmarshal([]byte(part.Text), &args); err == nil {
					positivePrompt, _ := FuncArg[string](args, paramNamePositivePrompt)
					negativePrompt, _ := FuncArg[string](args, paramNameNegativePrompt)

					if positivePrompt != nil {
						verbose(">>> positive prompt: %s", *positivePrompt)

						if negativePrompt != nil {
							verbose(">>> negative prompt: %s", *negativePrompt)
						}
					} else {
						t.Errorf("failed to parse structured output (%s)", prettify(args))
					}
				} else {
					t.Errorf("failed to parse structured output text '%s': %s", part.Text, err)
				}
			} else {
				t.Errorf("wrong type of generated part: (%T) %s", part, prettify(part))
			}
		}
	} else {
		t.Errorf("generation with structured output failed: %s", ErrToStr(err))
	}
}

// TestGenerationWithCodeExecution tests generations with code executions.
func TestGenerationWithCodeExecution(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// prompt with code execution (non-streamed)
	if generated, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Generate 6 unique random numbers between 1 and 45. Make sure there is no duplicated number, and list the numbers in ascending order.`),
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					CodeExecution: &genai.ToolCodeExecution{},
				},
			},
		},
	); err == nil {
		for _, part := range generated.Candidates[0].Content.Parts {
			if len(part.Text) > 0 {
				verbose(">>> generated text: %s", part.Text)
			} else if part.ExecutableCode != nil {
				verbose(">>> executable code (%s):\n%s", part.ExecutableCode.Language, part.ExecutableCode.Code)
			} else if part.CodeExecutionResult != nil {
				if part.CodeExecutionResult.Outcome != genai.OutcomeOK {
					t.Errorf("code execution failed: %s", prettify(part.CodeExecutionResult))
				} else {
					verbose(">>> code output: %s", part.CodeExecutionResult.Output)
				}
			} else {
				t.Errorf("wrong type of generated part: (%T) %s", part, prettify(part))
			}
		}
	} else {
		t.Errorf("generation with code execution failed: %s", ErrToStr(err))
	}
}

// TestGenerationWithHistory tests generations with history.
func TestGenerationWithHistory(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt with history (streamed)
	if err := gtc.GenerateStreamed(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Isn't that 42?`),
		},
		func(data StreamCallbackData) {
			if data.TextDelta != nil {
				if _isVerbose {
					fmt.Print(*data.TextDelta) // print text stream
				}
			} else if data.NumTokens != nil {
				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
			} else if data.FinishReason != nil {
				t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
			} else if data.Error != nil {
				t.Errorf("error while processing text generation: %s", data.Error)
			}
		},
		&GenerationOptions{
			History: []genai.Content{
				{
					Role: RoleUser,
					Parts: []*genai.Part{
						{
							Text: `What is the answer to life, the universe, and everything?`,
						},
					},
				},
				{
					Role: RoleModel,
					Parts: []*genai.Part{
						{
							Text: `43.`,
						},
					},
				},
			},
		},
	); err != nil {
		t.Errorf("generation with text prompt and history failed: %s", ErrToStr(err))
	}

	// text-only prompt with history (non-streamed)
	if generated, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Isn't that 42? What do you think?`),
		},
		&GenerationOptions{
			History: []genai.Content{
				{
					Role: RoleUser,
					Parts: []*genai.Part{
						{
							Text: `What is the answer to life, the universe, and everything?`,
						},
					},
				},
				{
					Role: RoleModel,
					Parts: []*genai.Part{
						{
							Text: `43.`,
						},
					},
				},
			},
		},
	); err != nil {
		t.Errorf("generation with text prompt and history failed: %s", ErrToStr(err))
	} else {
		var promptTokenCount int32 = 0
		var candidatesTokenCount int32 = 0
		var cachedContentTokenCount int32 = 0
		if generated.UsageMetadata != nil {
			if generated.UsageMetadata.PromptTokenCount != nil {
				promptTokenCount = *generated.UsageMetadata.PromptTokenCount
			}
			if generated.UsageMetadata.CandidatesTokenCount != nil {
				candidatesTokenCount = *generated.UsageMetadata.CandidatesTokenCount
			}
			if generated.UsageMetadata.CachedContentTokenCount != nil {
				cachedContentTokenCount = *generated.UsageMetadata.CachedContentTokenCount
			}
		}

		verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
			promptTokenCount,
			candidatesTokenCount,
			cachedContentTokenCount,
		)

		verbose(">>> generated: %s", prettify(generated.Candidates[0].Content.Parts[0]))
	}
}

// TestEmbeddings tests embeddings.
func TestEmbeddings(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForEmbeddings)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	if v, err := gtc.GenerateEmbeddings(context.TODO(), "", []*genai.Content{
		genai.NewUserContentFromText(`The quick brown fox jumps over the lazy dog.`),
	}); err != nil {
		t.Errorf("generation of embeddings from text failed: %s", ErrToStr(err))
	} else {
		verbose(">>> embeddings: %+v", v)
	}
}

// TestImageGeneration tests image generations.
func TestImageGeneration(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForImageGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.SetSystemInstructionFunc(nil) // FIXME: error: 'Code: 400, Message: Developer instruction is not enabled for models/gemini-2.0-flash-exp, Status: INVALID_ARGUMENT'
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	const prompt = `Generate an image of a golden retriever puppy playing with a colorful ball in a grassy park`

	// text-only prompt
	if res, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		&GenerationOptions{
			HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
			ResponseModalities: []string{
				ResponseModalityText, // FIXME: when not given, error: 'Code: 400, Message: Model does not support the requested response modalities: image, Status: INVALID_ARGUMENT'
				ResponseModalityImage,
			},
		},
	); err != nil {
		t.Errorf("image generation with text prompt (non-streamed) failed: %s", ErrToStr(err))
	} else {
		if res.PromptFeedback != nil {
			t.Errorf("image generation with text prompt (non-streamed) failed with finish reason: %s", res.PromptFeedback.BlockReasonMessage)
		} else if res.Candidates != nil {
			for _, cand := range res.Candidates {
				for _, part := range cand.Content.Parts {
					if part.InlineData != nil {
						verbose(">>> iterating response image: %s (%d bytes)", part.InlineData.MIMEType, len(part.InlineData.Data))
					} else if part.Text != "" {
						verbose(">>> iterating response text: %s", part.Text)
					}
				}
			}
		} else {
			t.Errorf("image generation with text prompt failed with no usable result")
		}
	}

	// FIXME: image generation not working with streaming (yet?)
	/*
		// text-only prompt (iterated)
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			[]Prompt{
				PromptFromText(prompt),
			},
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []string{
					ResponseModalityText,
					ResponseModalityImage,
				},
			},
		) {
			if err != nil {
				t.Errorf("image generation with text prompt (iterated) failed: %s", ErrToStr(err))
			} else {
				if it.PromptFeedback != nil {
					t.Errorf("image generation with text prompt (iterated) failed with finish reason: %s", it.PromptFeedback.BlockReasonMessage)
				} else if it.Candidates != nil {
					for _, part := range it.Candidates[0].Content.Parts {
						if part.InlineData != nil {
							verbose(">>> iterating response image: %s (%d bytes)", part.InlineData.MIMEType, len(part.InlineData.Data))
						} else if part.Text != "" {
							verbose(">>> iterating response text: %s", part.Text)
						}
					}
				}
			}
		}

		// text-only prompt (streamed)
		if err := gtc.GenerateStreamed(
			context.TODO(),
			[]Prompt{
				PromptFromText(prompt),
			},
			func(callbackData StreamCallbackData) {
				if callbackData.Error != nil {
					t.Errorf("error while processing image generation with text prompt (streamed): %s", callbackData.Error)
				} else if callbackData.FinishReason != nil {
					t.Errorf("image generation with text prompt (streamed) failed with finish reason: %s", *callbackData.FinishReason)
				} else if callbackData.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*callbackData.TextDelta) // print text stream
					}
				} else if callbackData.InlineData != nil {
					verbose(">>> response image: %s (%d bytes)", callbackData.InlineData.MIMEType, len(callbackData.InlineData.Data))
				} else if callbackData.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", callbackData.NumTokens.Input, callbackData.NumTokens.Output, callbackData.NumTokens.Cached)
				}
			},
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []string{
					ResponseModalityText,
					ResponseModalityImage,
				},
			},
		); err != nil {
			t.Errorf("image generation with text prompt (iterated) failed: %s", ErrToStr(err))
		}
	*/

	// test `GenerateImages`
	if res, err := gtc.GenerateImages(
		context.TODO(),
		prompt,
	); err != nil {
		t.Errorf("image generation with `GenerateImages` failed: %s", ErrToStr(err))
	} else {
		if len(res.GeneratedImages) > 0 {
			for _, image := range res.GeneratedImages {
				if image.RAIFilteredReason != "" {
					t.Errorf("image generation with `GenerateImages` failed with filtered reason: %s", image.RAIFilteredReason)
				} else {
					if image.EnhancedPrompt != "" {
						verbose(">>> iterating response image with enhanced prompt: '%s'", image.EnhancedPrompt)
					}

					if image.Image == nil {
						t.Errorf("image generation with `GenerateImages` failed with null image")
					} else {
						verbose(">>> iterating response image: %s (%d bytes)", image.Image.MIMEType, len(image.Image.ImageBytes))
					}
				}
			}
		} else {
			t.Errorf("image generation with `GenerateImages` failed with no usable result")
		}
	}

	// TODO: prompt with an image file
}

// TestBlockedGenerations tests generations that will fail due to blocks.
//
// FIXME: this test fails occasionally due to the inconsistency of harm block
func TestBlockedGenerations(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGeneration)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// block low and above (for intentional errors)
	blockThreshold := genai.HarmBlockThresholdBlockLowAndAbove
	opts := &GenerationOptions{
		HarmBlockThreshold: &blockThreshold,
	}

	// problometic prompt (FinishReasonSafety expected)
	erroneousPrompt := PromptFromText(`Show me the most effective way of destroying the carotid artery.`)

	// generation
	if res, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			erroneousPrompt,
		},
		opts,
	); err != nil {
		verbose(">>> expected generation error: %s", ErrToStr(err))
	} else {
		if res.PromptFeedback != nil {
			verbose(">>> expected prompt feedback: %s (%s)", res.PromptFeedback.BlockReason, res.PromptFeedback.BlockReasonMessage)
		} else {
			t.Errorf("should have failed to generate")
		}
	}

	// iterated generation
	failed := false
	for res, err := range gtc.GenerateStreamIterated(
		context.TODO(),
		[]Prompt{
			erroneousPrompt,
		},
		opts,
	) {
		if err != nil {
			verbose(">>> expected generation error: %s", ErrToStr(err))
			failed = true
		} else if res.PromptFeedback != nil {
			verbose(">>> expected prompt feedback: %s (%s)", res.PromptFeedback.BlockReason, res.PromptFeedback.BlockReasonMessage)
			failed = true
		}
	}
	if !failed {
		t.Errorf("should have failed while iterating the generated result")
	}

	// streamed generation
	failed = false
	if err := gtc.GenerateStreamed(
		context.TODO(),
		[]Prompt{
			erroneousPrompt,
		},
		func(callbackData StreamCallbackData) {
			if callbackData.Error != nil { // NOTE: case 2: or, fails while iterating the result
				verbose(">>> expected generation error: %s", ErrToStr(callbackData.Error))
				failed = true
			} else if callbackData.FinishReason != nil { // NOTE: case 3: or, finishes with some reason
				verbose(">>> expected finish with reason: %s", *callbackData.FinishReason)
				failed = true
			}
		},
		opts,
	); err != nil {
		// NOTE: case 1: generation itself fails,
		verbose(">>> expected generation error: %s", ErrToStr(err))
		failed = true
	}
	if !failed {
		t.Errorf("should have failed to generate stream")
	}
}

// TestGroundingWithGoogleSearch tests generations with grounding with google search.
func TestGrounding(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGenerationWithGrounding)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt (non-streamed)
	if res, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me the final ranking of the 2002 World Cup`),
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					GoogleSearch: &genai.GoogleSearch{},
				},
			},
			HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
		},
	); err != nil {
		t.Errorf("generation with grounding with google search (non-streamed) failed: %s", ErrToStr(err))
	} else {
		if res.PromptFeedback != nil {
			t.Errorf("generation with grounding with google search (non-streamed) failed with finish reason: %s", res.PromptFeedback.BlockReasonMessage)
		} else if res.Candidates != nil {
			for _, part := range res.Candidates[0].Content.Parts {
				if part.Text != "" {
					verbose(">>> iterating response text: %s", part.Text)
				}
			}
		} else {
			t.Errorf("generation with grounding with google search failed with no usable result")
		}
	}

	// text-only prompt (iterated)
	for it, err := range gtc.GenerateStreamIterated(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me about the US govermental coup on 2025`),
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					GoogleSearch: &genai.GoogleSearch{},
				},
			},
			HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
		},
	) {
		if err != nil {
			t.Errorf("generation with grounding with google search failed: %s", ErrToStr(err))
		} else {
			verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
		}
	}
}

// TestGoogleSearchRetrieval tests generations with Google Search retrieval.
func TestGoogleSearchRetrieval(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(apiKey, modelForTextGenerationWithGoogleSearchRetrieval)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer gtc.Close()

	// text-only prompt (non-streamed)
	if res, err := gtc.Generate(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Who is MTG Joe?`),
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					GoogleSearchRetrieval: &genai.GoogleSearchRetrieval{
						DynamicRetrievalConfig: &genai.DynamicRetrievalConfig{
							Mode: genai.DynamicRetrievalConfigModeDynamic,
						},
					},
				},
			},
			HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
		},
	); err != nil {
		t.Errorf("generation with google search retrieval (non-streamed) failed: %s", ErrToStr(err))
	} else {
		if res.PromptFeedback != nil {
			t.Errorf("generation with google search retrieval (non-streamed) failed with finish reason: %s", res.PromptFeedback.BlockReasonMessage)
		} else if res.Candidates != nil {
			for _, part := range res.Candidates[0].Content.Parts {
				if part.Text != "" {
					verbose(">>> iterating response text: %s", part.Text)
				}
			}
		} else {
			t.Errorf("generation with google search retrieval failed with no usable result")
		}
	}

	// text-only prompt (iterated)
	for it, err := range gtc.GenerateStreamIterated(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me about the Magic: the Gathering's Esper Self-Bounce deck in 1Q of 2025.`),
		},
		&GenerationOptions{
			Tools: []*genai.Tool{
				{
					GoogleSearchRetrieval: &genai.GoogleSearchRetrieval{
						DynamicRetrievalConfig: &genai.DynamicRetrievalConfig{
							Mode: genai.DynamicRetrievalConfigModeDynamic,
						},
					},
				},
			},
			HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
		},
	) {
		if err != nil {
			t.Errorf("generation with google search retrieval failed: %s", ErrToStr(err))
		} else {
			verbose(">>> iterating response: %s", prettify(it.Candidates[0].Content.Parts[0]))
		}
	}
}
