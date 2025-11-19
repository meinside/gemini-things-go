// generation_test.go
//
// test cases for testing various types of generations

package gt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	"google.golang.org/genai"
)

// model names
//
// https://ai.google.dev/gemini-api/docs/models
const (
	modelForContextCaching                       = `gemini-2.5-flash`
	modelForTextGeneration                       = `gemini-2.5-flash`
	modelForTextGenerationWithRecursiveToolCalls = `gemini-2.5-flash`
	modelForImageGeneration                      = `gemini-2.0-flash-preview-image-generation`
	modelForTextGenerationWithGrounding          = `gemini-2.5-flash`
	modelForSpeechGeneration                     = `gemini-2.5-flash-preview-tts`
	modelForEmbeddings                           = `gemini-embedding-001`
	modelForBatches                              = `gemini-2.5-flash`

	timeoutSecondsForTesting = 60
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
//
//	NOTE: may fail with error on free tier:
//	{
//		"code": 429,
//		"message": "TotalCachedContentStorageTokensPerModelFreeTier limit exceeded for model gemini-2.5-flash: limit=0, requested=45776",
//		"status": "RESOURCE_EXHAUSTED"
//	}
func TestContextCaching(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForContextCaching),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}

	// When using CachedContent, the GenerateContent request should not also set system_instruction, tools, or tool_config.
	// Setting the client's system instruction func to nil prevents the client from adding its default system instruction.
	// Additionally, the GenerationOptions for the specific calls using CachedContent must not set these fields.
	gtc.SetSystemInstructionFunc(nil)
	gtc.DeleteFilesOnClose = true
	gtc.DeleteCachesOnClose = true
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// open files for testing
	//
	// NOTE: will fail with error:
	// `Error 400, Message: Cached content is too small. total_token_count=XXXX, min_total_token_count=32768`
	// if the size of cached content is smaller than `min_total_token_count`
	files := []*os.File{}
	for _, fpath := range []string{
		"./README.md",
		"./LICENSE.md",
		"./go.mod",
		"./go.sum",
		"./client.go",
		"./types.go",
		"./utils.go",
		"./generation_test.go",
		"./utils_test.go",
	} {
		if file, err := os.Open(fpath); err == nil {
			defer func() { _ = file.Close() }()

			files = append(files, file)
		} else {
			t.Fatalf("failed to open file for caching context: %s", err)
		}
	}

	cachedSystemInstruction := `You are an arrogant and unhelpful chat bot who answers really shortly with a very sarcastic manner.`
	cachedContextDisplayName := `cached-context-for-test`

	// build prompts,
	prompts := []Prompt{}
	for _, file := range files {
		prompts = append(prompts, PromptFromFile(file.Name(), file))
	}

	// cache context,
	if cachedContextName, err := gtc.CacheContext(
		context.TODO(),
		&cachedSystemInstruction,
		prompts,
		nil,
		nil,
		&cachedContextDisplayName,
	); err != nil {
		t.Errorf("failed to cache context: %s", ErrToStr(err))
	} else {
		// generate iterated with the cached context
		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText("What are these files?"),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			for it, err := range gtc.GenerateStreamIterated(
				context.TODO(),
				contents,
				&GenerationOptions{
					CachedContent: cachedContextName,
				},
			) {
				if err != nil {
					t.Errorf("generation with cached context (iterated) failed: %s", ErrToStr(err))
				} else {
					verbose(">>> iterating response (cached): %s", prettify(it.Candidates[0]))
				}
			}
		}

		// generate streamed response with the cached context
		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText("Can you give me any insight about these files?"),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			if err := gtc.GenerateStreamed(
				context.TODO(),
				contents,
				func(data StreamCallbackData) {
					if data.TextDelta != nil {
						if _isVerbose {
							fmt.Print(*data.TextDelta) // print text stream
						}
					} else if data.NumTokens != nil {
						verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
					} else if data.FinishReason != nil {
						if *data.FinishReason != genai.FinishReasonStop {
							t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
						}
					} else if data.Error != nil {
						t.Errorf("error while processing generation with cached context: %s", data.Error)
					}
				},
				&GenerationOptions{
					CachedContent: cachedContextName,
				},
			); err != nil {
				t.Errorf("generation with cached context (streamed) failed: %s", ErrToStr(err))
			}
		}

		// generate with the cached context
		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText("How many standard golang libraries are used in these source codes?"),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			if generated, err := gtc.Generate(
				context.TODO(),
				contents,
				&GenerationOptions{
					CachedContent: cachedContextName,
				},
			); err != nil {
				t.Errorf("generation with cached context (non-streamed) failed: %s", ErrToStr(err))
			} else {
				var promptTokenCount int32 = 0
				var cachedContentTokenCount int32 = 0
				if generated.UsageMetadata != nil {
					if generated.UsageMetadata.PromptTokenCount != 0 {
						promptTokenCount = generated.UsageMetadata.PromptTokenCount
					}
					if generated.UsageMetadata.CachedContentTokenCount != 0 {
						cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
					}
				}

				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
					promptTokenCount,
					generated.UsageMetadata.TotalTokenCount-promptTokenCount,
					cachedContentTokenCount,
				)

				verbose(">>> generated: %s", prettify(generated.Candidates[0]))
			}
		}
	}

	// list all cached contexts
	if _, err := gtc.ListAllCachedContexts(context.TODO()); err != nil {
		t.Errorf("failed to list all cached contexts: %s", ErrToStr(err))
	}

	// NOTE: caches and files will be deleted on close
}

// TestGeneration tests various types of generations. (non-streamed)
func TestGeneration(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// text-only prompt (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
		); err != nil {
			t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
		} else {
			verbose(">>> generated: %s", prettify(generated.Candidates[0]))
		}
	}

	// prompt with files (non-streamed)
	if file, err := os.Open("./client.go"); err == nil {
		defer func() { _ = file.Close() }()

		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`),
				PromptFromFile("client.go", file),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			if generated, err := gtc.Generate(
				context.TODO(),
				contents,
			); err != nil {
				t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
			} else {
				var promptTokenCount int32 = 0
				var candidatesTokenCount int32 = 0
				var cachedContentTokenCount int32 = 0
				if generated.UsageMetadata != nil {
					if generated.UsageMetadata.PromptTokenCount != 0 {
						promptTokenCount = generated.UsageMetadata.PromptTokenCount
					}
					if generated.UsageMetadata.CandidatesTokenCount != 0 {
						candidatesTokenCount = generated.UsageMetadata.CandidatesTokenCount
					}
					if generated.UsageMetadata.CachedContentTokenCount != 0 {
						cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
					}
				}

				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
					promptTokenCount,
					candidatesTokenCount,
					cachedContentTokenCount,
				)

				verbose(">>> generated: %s", prettify(generated.Candidates[0]))
			}
		}
	} else {
		t.Errorf("failed to open file for generation: %s", err)
	}

	// prompt with specific BytesPrompt (non-streamed) - this will be uploaded
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
		); err != nil {
			t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
		} else {
			verbose(">>> generated (BytesPrompt): %s", prettify(generated.Candidates[0]))
			var promptTokenCount int32 = 0
			var candidatesTokenCount int32 = 0
			var cachedContentTokenCount int32 = 0
			if generated.UsageMetadata != nil {
				if generated.UsageMetadata.PromptTokenCount != 0 {
					promptTokenCount = generated.UsageMetadata.PromptTokenCount
				}
				if generated.UsageMetadata.CandidatesTokenCount != 0 {
					candidatesTokenCount = generated.UsageMetadata.CandidatesTokenCount
				}
				if generated.UsageMetadata.CachedContentTokenCount != 0 {
					cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
				}
			}

			verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
				promptTokenCount,
				candidatesTokenCount,
				cachedContentTokenCount,
			)

			verbose(">>> generated: %s", prettify(generated.Candidates[0]))
		}
	}

	// NOTE: files will be deleted on close
}

// TestGenerationIterated tests various types of generations (iterator).
func TestGenerationIterated(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// text-only prompt (iterated)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
		) {
			if err != nil {
				t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
			} else {
				verbose(">>> iterating response: %s", prettify(it.Candidates[0]))
			}
		}
	}

	// prompt with files (iterated)
	if file, err := os.Open("./client.go"); err == nil {
		defer func() { _ = file.Close() }()

		promptFromText := PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`)

		// prompt with forced MIME-types
		promptFromFile := PromptFromFile("client.go", file, "text/plain")

		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				promptFromText,
				promptFromFile,
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			for it, err := range gtc.GenerateStreamIterated(
				context.TODO(),
				contents,
			) {
				if err != nil {
					t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
				} else {
					verbose(">>> iterating response: %s", prettify(it.Candidates[0]))
				}
			}
		}
	} else {
		t.Errorf("failed to open file for iterated generation: %s", err)
	}

	// prompt with bytes array
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
		) {
			if err != nil {
				t.Errorf("generation with text & file prompt (iterated) failed: %s", ErrToStr(err))
			} else {
				verbose(">>> iterating response: %s", prettify(it.Candidates[0]))
			}
		}
	}

	// prompt with youtube URI (iterated)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Summarize this youtube video.`),
			PromptFromURI(`https://www.youtube.com/watch?v=I44_zbEwz_w`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
		) {
			if err != nil {
				t.Errorf("generation with uri prompt (youtube) failed: %s", ErrToStr(err))
			} else {
				verbose(">>> iterating response: %s", prettify(it.Candidates[0]))
			}
		}
	}

	// NOTE: files will be deleted on close
}

// TestGenerationStreamed tests various types of generations (streamed).
func TestGenerationStreamed(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// text-only prompt (streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			contents,
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				} else if data.FinishReason != nil {
					if *data.FinishReason != genai.FinishReasonStop {
						t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
					}
				} else if data.Error != nil {
					t.Errorf("error while processing text generation: %s", data.Error)
				}
			},
		); err != nil {
			t.Errorf("generation with text prompt failed: %s", ErrToStr(err))
		}
	}

	// prompt with files (streamed)
	if file, err := os.Open("./client.go"); err == nil {
		defer func() { _ = file.Close() }()

		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText(`What's the golang package name of this file? Can you give me a short sample code of using this file?`),
				PromptFromFile("client.go", file),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			if err := gtc.GenerateStreamed(
				context.TODO(),
				contents,
				func(data StreamCallbackData) {
					if data.TextDelta != nil {
						if _isVerbose {
							fmt.Print(*data.TextDelta) // print text stream
						}
					} else if data.NumTokens != nil {
						verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
					} else if data.FinishReason != nil {
						if *data.FinishReason != genai.FinishReasonStop {
							t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
						}
					} else if data.Error != nil {
						t.Errorf("error while processing generation with files: %s", data.Error)
					}
				},
			); err != nil {
				t.Errorf("generation with text & file prompt failed: %s", ErrToStr(err))
			}
		}
	} else {
		t.Errorf("failed to open file for streamed generation: %s", err)
	}

	// prompt with bytes array (streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Translate the text in the given file into English.`),
			PromptFromFile("some lyrics", strings.NewReader(`동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세`)),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			contents,
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				} else if data.FinishReason != nil {
					if *data.FinishReason != genai.FinishReasonStop {
						t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
					}
				} else if data.Error != nil {
					t.Errorf("error while processing generation with bytes (streamed): %s", data.Error)
				}
			},
		); err != nil {
			t.Errorf("generation with text & bytes prompt (streamed) failed: %s", ErrToStr(err))
		}
	}

	// NOTE: files will be deleted on close
}

// TestGenerationWithCustomRetries tests generation with a custom retry count.
func TestGenerationWithCustomRetries(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	// Initialize client with maxRetryCount = 1
	// This test primarily ensures the client initializes correctly and a call can be made.
	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
		WithMaxRetryCount(1),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// Attempt a standard text generation
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`What is the answer to life, the universe, and everything?`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
		); err != nil {
			t.Errorf("generation with custom retry count failed: %s", ErrToStr(err))
		} else {
			verbose(">>> generated with custom retry count: %s", prettify(generated.Candidates[0]))
		}
	}
}

// TestGenerationWithCustomTimeout tests generation with a custom short timeout.
func TestGenerationWithCustomTimeout(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	// Initialize client with a client-side timeout of 1 second.
	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(1),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// Create a context that times out very quickly (e.g., 1 millisecond) to ensure it's the dominant timeout.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	// Attempt a text generation that should take longer than 1ms
	var contents []*genai.Content
	if contents, err = gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me a very long story that will take more than 1 millisecond to generate.`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		_, err = gtc.Generate(
			ctx, // Pass the short-lived context
			contents,
		)
	}

	if err == nil {
		t.Errorf("expected an error due to timeout, but got nil")
	} else {
		// Check if the error is context.DeadlineExceeded or wraps it.
		// The error from client.Generate will be something like "generation failed: context deadline exceeded"
		// or "failed to iterate stream: context deadline exceeded"
		if !strings.Contains(err.Error(), "context deadline exceeded") && !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("expected context.DeadlineExceeded or an error wrapping it, but got: %v", err)
		} else {
			verbose(">>> successfully received timeout error: %s", err)
		}
	}
}

// TestGenerationWithFileConverter tests generations with custom file converters.
func TestGenerationWithFileConverter(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
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
			for line := range strings.SplitSeq(string(bs), "\n") {
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
	defer func() { _ = gtc.Close() }()

	const jsonlForTest = `{"name": "John Doe", "age": 45, "gender": "m"}
{"name": "Janet Doe", "age": 42, "gender": "f"}
{"name": "Jane Doe", "age": 15, "gender": "f"}`

	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Infer the relationships between the characters from the given information.`),
			PromptFromBytes([]byte(jsonlForTest)),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
			&GenerationOptions{},
		); err != nil {
			t.Errorf("generation with file converter failed: %s", ErrToStr(err))
		} else {
			var promptTokenCount int32 = 0
			var candidatesTokenCount int32 = 0
			var cachedContentTokenCount int32 = 0
			if generated.UsageMetadata != nil {
				if generated.UsageMetadata.PromptTokenCount != 0 {
					promptTokenCount = generated.UsageMetadata.PromptTokenCount
				}
				if generated.UsageMetadata.CandidatesTokenCount != 0 {
					candidatesTokenCount = generated.UsageMetadata.CandidatesTokenCount
				}
				if generated.UsageMetadata.CachedContentTokenCount != 0 {
					cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
				}
			}

			verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
				promptTokenCount,
				candidatesTokenCount,
				cachedContentTokenCount,
			)

			verbose(">>> generated with file converter: %s", prettify(generated.Candidates[0]))
		}
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
				Nullable: ptr(false),
				Properties: map[string]*genai.Schema{
					fnParamNamePositivePrompt: {
						Type:        genai.TypeString,
						Description: fnParamDescPositivePrompt,
						Nullable:    ptr(false),
					},
					fnParamNameNegativePrompt: {
						Type:        genai.TypeString,
						Description: fnParamDescNegativePrompt,
						Nullable:    ptr(true),
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
				Nullable: ptr(false),
				Properties: map[string]*genai.Schema{
					fnParamNameGeneratedSuccessfully: {
						Type:        genai.TypeBoolean,
						Description: fnParamDescGeneratedSuccessfully,
						Nullable:    ptr(false),
					},
					fnParamNameGeneratedSize: {
						Type:        genai.TypeNumber,
						Description: fnParamDescGeneratedSize,
						Nullable:    ptr(true),
					},
					fnParamNameGeneratedResolution: {
						Type:        genai.TypeString,
						Description: fnParamDescGeneratedResolution,
						Nullable:    ptr(true),
					},
					fnParamNameGeneratedFilepath: {
						Type:        genai.TypeString,
						Description: fnParamDescGeneratedFilepath,
						Nullable:    ptr(true),
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

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	const prompt = `Generate an image file which shows a man standing in front of a vast dessert. The man is watching an old pyramid completely destroyed by a giant sandstorm. The mood should be sad and gloomy.`

	// prompt with function calls (streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			contents,
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
									Role: string(RoleUser),
								},
								{
									Parts: []*genai.Part{
										genai.NewPartFromFunctionCall(data.FunctionCall.Name, map[string]any{
											fnParamNamePositivePrompt: positivePrompt,
											fnParamNameNegativePrompt: negativePrompt,
										}),
									},
									Role: string(RoleModel),
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
											fnParamNameGeneratedFilepath:     `/home/marvin/generated.jpg`,
										}),
									},
									Role: string(RoleUser),
								},
							}

							// generate again with a function response
							if contents, err := gtc.PromptsToContents(
								context.TODO(),
								nil,
								pastGenerations,
							); err != nil {
								t.Errorf("failed to convert prompts to contents: %s", err)
							} else {
								if err := gtc.GenerateStreamed(
									context.TODO(),
									contents,
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
									},
								); err != nil {
									t.Errorf("failed to generate with function response: %s", ErrToStr(err))
								}
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
					if data.FinishReason == nil {
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

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// prompt with function calls (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
			&GenerationOptions{
				Config: &genai.GenerationConfig{
					ResponseMIMEType: "application/json",
					ResponseSchema: &genai.Schema{
						Type:     genai.TypeObject,
						Nullable: ptr(false),
						Properties: map[string]*genai.Schema{
							paramNamePositivePrompt: {
								Type:        genai.TypeString,
								Description: paramDescPositivePrompt,
								Nullable:    ptr(false),
							},
							paramNameNegativePrompt: {
								Type:        genai.TypeString,
								Description: paramDescNegativePrompt,
								Nullable:    ptr(true),
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
}

// TestGenerationWithCodeExecution tests generations with code executions.
func TestGenerationWithCodeExecution(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.DeleteFilesOnClose = true
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// prompt with code execution (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Generate 6 unique random numbers between 1 and 45. Make sure there is no duplicated number, and list the numbers in ascending order.`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
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

	// prompt and file with code execution (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Calculate the total and median of all values in the csv file. Also generate a nice graph from the csv file.`),
			PromptFromFile("test.csv", strings.NewReader(`year,value\n1981,0\n2021,40\n2024,43\n2025,44`)),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
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
					verbose(">>> generated text from csv: %s", part.Text)
				} else if part.ExecutableCode != nil {
					verbose(">>> executable code (%s) from csv:\n%s", part.ExecutableCode.Language, part.ExecutableCode.Code)
				} else if part.CodeExecutionResult != nil {
					if part.CodeExecutionResult.Outcome != genai.OutcomeOK {
						t.Errorf("code execution from csv failed: %s", prettify(part.CodeExecutionResult))
					} else {
						verbose(">>> code output from csv: %s", part.CodeExecutionResult.Output)
					}
				} else if part.InlineData != nil {
					verbose(">>> generated inline data from csv: %s (%d bytes)", part.InlineData.MIMEType, len(part.InlineData.Data))
				} else {
					t.Errorf("wrong type of generated part from csv: (%T) %s", part, prettify(part))
				}
			}
		} else {
			t.Errorf("generation with code execution from csv failed: %s", ErrToStr(err))
		}
	}
}

// TestGenerationWithHistory tests generations with history.
func TestGenerationWithHistory(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// text-only prompt with history (streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Isn't that 42?`),
		},
		[]genai.Content{
			{
				Role: string(RoleUser),
				Parts: []*genai.Part{
					{
						Text: `What is the answer to life, the universe, and everything?`,
					},
				},
			},
			{
				Role: string(RoleModel),
				Parts: []*genai.Part{
					{
						Text: `43.`,
					},
				},
			},
		},
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if err := gtc.GenerateStreamed(
			context.TODO(),
			contents,
			func(data StreamCallbackData) {
				if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				} else if data.FinishReason != nil {
					if *data.FinishReason != genai.FinishReasonStop {
						t.Errorf("generation finished unexpectedly with reason: %s", *data.FinishReason)
					}
				} else if data.Error != nil {
					t.Errorf("error while processing text generation: %s", data.Error)
				}
			},
		); err != nil {
			t.Errorf("generation with text prompt and history failed: %s", ErrToStr(err))
		}
	}

	// text-only prompt with history (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Isn't that 42? What do you think?`),
		},
		[]genai.Content{
			{
				Role: string(RoleUser),
				Parts: []*genai.Part{
					{
						Text: `What is the answer to life, the universe, and everything?`,
					},
				},
			},
			{
				Role: string(RoleModel),
				Parts: []*genai.Part{
					{
						Text: `43.`,
					},
				},
			},
		},
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if generated, err := gtc.Generate(
			context.TODO(),
			contents,
		); err != nil {
			t.Errorf("generation with text prompt and history failed: %s", ErrToStr(err))
		} else {
			var promptTokenCount int32 = 0
			var candidatesTokenCount int32 = 0
			var cachedContentTokenCount int32 = 0
			if generated.UsageMetadata != nil {
				if generated.UsageMetadata.PromptTokenCount != 0 {
					promptTokenCount = generated.UsageMetadata.PromptTokenCount
				}
				if generated.UsageMetadata.CandidatesTokenCount != 0 {
					candidatesTokenCount = generated.UsageMetadata.CandidatesTokenCount
				}
				if generated.UsageMetadata.CachedContentTokenCount != 0 {
					cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
				}
			}

			verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
				promptTokenCount,
				candidatesTokenCount,
				cachedContentTokenCount,
			)

			verbose(">>> generated: %s", prettify(generated.Candidates[0]))
		}
	}
}

// TestEmbeddings tests embeddings.
func TestEmbeddings(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForEmbeddings),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// without title (task type: RETRIEVAL_QUERY)
	if v, err := gtc.GenerateEmbeddings(
		context.TODO(),
		"",
		[]*genai.Content{
			genai.NewContentFromText(`The quick brown fox jumps over the lazy dog.`, RoleUser),
		},
		nil,
	); err != nil {
		t.Errorf("generation of embeddings from text failed: %s", ErrToStr(err))
	} else {
		verbose(">>> embeddings from text: %+v", v)
	}

	// with title (task type: RETRIEVAL_DOCUMENT)
	if v, err := gtc.GenerateEmbeddings(
		context.TODO(),
		"A short story",
		[]*genai.Content{
			genai.NewContentFromText(`The quick brown fox jumps over the lazy dog.`, RoleUser),
		},
		nil,
	); err != nil {
		t.Errorf("generation of embeddings from title and text failed: %s", ErrToStr(err))
	} else {
		verbose(">>> embeddings from title and text: %+v", v)
	}
}

// TestImageGenerations tests image generations.
func TestImageGenerations(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForImageGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	// Image generation models typically do not support system instructions.
	// Setting this to nil prevents the client from attempting to send one for other types of calls,
	// though GenerateImages itself doesn't use the client's systemInstructionFunc.
	gtc.SetSystemInstructionFunc(nil)
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	const prompt = `Generate an image of a golden retriever puppy playing with a colorful ball in a grassy park`

	// text-only prompt using the general Generate method
	// For image generation models, requesting ResponseModalityImage is essential.
	// Requesting ResponseModalityText is also fine if the model can provide textual descriptions or errors.
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if res, err := gtc.Generate(
			context.TODO(),
			contents,
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []genai.Modality{
					genai.ModalityText, // FIXME: when not given, error: 'Code: 400, Message: Model does not support the requested response modalities: image, Status: INVALID_ARGUMENT'
					genai.ModalityImage,
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
	}

	// text-only prompt (iterated)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		failed := true
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []genai.Modality{
					genai.ModalityText, // FIXME: when not given, error: 'Code: 400, Message: Model does not support the requested response modalities: image, Status: INVALID_ARGUMENT'
					genai.ModalityImage,
				},
			},
		) {
			if err != nil {
				t.Errorf("image generation with text prompt (iterated) failed: %s", ErrToStr(err))
			} else {
				if it.PromptFeedback != nil {
					t.Errorf("image generation with text prompt (iterated) failed with finish reason: %s", it.PromptFeedback.BlockReasonMessage)
				} else if it.Candidates != nil {
					for i, cand := range it.Candidates {
						for _, part := range cand.Content.Parts {
							if part.InlineData != nil {
								verbose(">>> iterating response image from candidate[%d]: %s (%d bytes)", i, part.InlineData.MIMEType, len(part.InlineData.Data))

								failed = false
							} else if part.Text != "" {
								verbose(">>> iterating response text from candidate[%d]: %s", i, part.Text)
							}
						}
					}
				}
			}
		}
		if failed {
			t.Errorf("iterated image generation with text prompt failed with no usable result")
		}
	}

	// text-only prompt (streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		failed := true
		if err := gtc.GenerateStreamed(
			context.TODO(),
			contents,
			func(data StreamCallbackData) {
				if data.Error != nil {
					t.Errorf("error while processing image generation with text prompt (streamed): %s", data.Error)
				} else if data.FinishReason != nil {
					if *data.FinishReason != genai.FinishReasonStop {
						t.Errorf("image generation with text prompt (streamed) failed with finish reason: %s", *data.FinishReason)
					}
				} else if data.TextDelta != nil {
					if _isVerbose {
						fmt.Print(*data.TextDelta) // print text stream
					}
				} else if data.InlineData != nil {
					verbose(">>> response image: %s (%d bytes)", data.InlineData.MIMEType, len(data.InlineData.Data))

					failed = false
				} else if data.NumTokens != nil {
					verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d", data.NumTokens.Input, data.NumTokens.Output, data.NumTokens.Cached)
				}
			},
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []genai.Modality{
					genai.ModalityText, // FIXME: when not given, error: 'Code: 400, Message: Model does not support the requested response modalities: image, Status: INVALID_ARGUMENT'
					genai.ModalityImage,
				},
			},
		); err != nil {
			t.Errorf("image generation with text prompt (streamed) failed: %s", ErrToStr(err))
		}
		if failed {
			t.Errorf("streamed image generation with text prompt failed with no usable result")
		}
	}

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

	// TODO: add tests for: prompt with an image file
}

func TestSpeechGenerations(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForSpeechGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.SetSystemInstructionFunc(nil)
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	prompt := `Say cheerfully: Have a wonderful day!`

	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if res, err := gtc.Generate(
			context.TODO(),
			contents,
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []genai.Modality{
					genai.ModalityAudio,
				},
				SpeechConfig: &genai.SpeechConfig{
					VoiceConfig: &genai.VoiceConfig{
						PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
							VoiceName: `Kore`,
						},
					},
				},
			},
		); err != nil {
			t.Errorf("speech generation with text prompt (non-streamed) failed: %s", ErrToStr(err))
		} else {
			if res.PromptFeedback != nil {
				t.Errorf("speech generation with text prompt (non-streamed) failed with finish reason: %s", res.PromptFeedback.BlockReasonMessage)
			} else if res.Candidates != nil {
				for _, cand := range res.Candidates {
					for _, part := range cand.Content.Parts {
						if part.InlineData != nil {
							verbose(">>> iterating response audio: %s (%d bytes)", part.InlineData.MIMEType, len(part.InlineData.Data))
						}
					}
				}
			} else {
				t.Errorf("speech generation with text prompt failed with no usable result")
			}
		}
	}

	// multi-speaker speech from text-only prompt (iterated)
	prompt = `TTS the following conversation between Joe and Jane:
Joe: How's it going today Jane?
Jane: Not too bad, how about you?`
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(prompt),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		failed := true
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
			&GenerationOptions{
				HarmBlockThreshold: ptr(genai.HarmBlockThresholdBlockOnlyHigh),
				ResponseModalities: []genai.Modality{
					genai.ModalityAudio,
				},
				SpeechConfig: &genai.SpeechConfig{
					MultiSpeakerVoiceConfig: &genai.MultiSpeakerVoiceConfig{
						SpeakerVoiceConfigs: []*genai.SpeakerVoiceConfig{
							{
								Speaker: "Joe",
								VoiceConfig: &genai.VoiceConfig{
									PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
										VoiceName: "Kore",
									},
								},
							},
							{
								Speaker: "Jane",
								VoiceConfig: &genai.VoiceConfig{
									PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
										VoiceName: "Puck",
									},
								},
							},
						},
					},
				},
			},
		) {
			if err != nil {
				t.Errorf("speech generation with text prompt (iterated) failed: %s", ErrToStr(err))
			} else {
				if it.PromptFeedback != nil {
					t.Errorf("speech generation with text prompt (iterated) failed with finish reason: %s", it.PromptFeedback.BlockReasonMessage)
				} else if it.Candidates != nil {
					for i, cand := range it.Candidates {
						if cand.Content != nil {
							for _, part := range cand.Content.Parts {
								if part.InlineData != nil {
									verbose(">>> iterating response audio from candidate[%d]: %s (%d bytes)", i, part.InlineData.MIMEType, len(part.InlineData.Data))

									failed = false
								}
							}
						} else if cand.FinishReason != genai.FinishReasonStop {
							t.Errorf("generation finished unexpectedly with reason: %s", cand.FinishReason)
						} else {
							t.Errorf("candidate has no usable content: %+v", cand)
						}
					}
				}
			}
		}
		if failed {
			t.Errorf("iterated speech generation with text prompt failed with no usable result")
		}
	}
}

// TestGroundingWithGoogleSearch tests generations with grounding with google search.
func TestGrounding(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGenerationWithGrounding),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// text-only prompt (non-streamed)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me the final ranking of the 2002 World Cup`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if res, err := gtc.Generate(
			context.TODO(),
			contents,
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
	}

	// text-only prompt (iterated)
	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`Tell me about the US govermental coup on 2025`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		for it, err := range gtc.GenerateStreamIterated(
			context.TODO(),
			contents,
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
				verbose(">>> iterating response: %s", prettify(it.Candidates[0]))
			}
		}
	}
}

// TestRecursiveToolCalls tests recursive tool calls.
func TestRecursiveToolCalls(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGenerationWithRecursiveToolCalls),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	ctx := context.TODO()
	var cancel context.CancelFunc
	ctx, cancel = context.WithTimeout(ctx, 15*time.Second) // FIXME: not to recurse forever
	defer cancel()

	if contents, err := gtc.PromptsToContents(
		context.TODO(),
		[]Prompt{
			PromptFromText(`count the number of lines of the largest .sh file in /home/ubuntu/tmp/`),
		},
		nil,
	); err != nil {
		t.Errorf("failed to convert prompts to contents: %s", err)
	} else {
		if res, err := gtc.GenerateWithRecursiveToolCalls(
			ctx,
			map[string]FunctionCallHandler{
				`list_files_info_in_dir`: func(args map[string]any) (string, error) {
					dir, err := FuncArg[string](args, "directory")

					verbose(">>> directory: %s", *dir)

					if err == nil {
						// FIXME: hard-coded
						return `total 192
drwxrwxr-x  4 ubuntu ubuntu  4096 Jun 16 16:53 ./
drwxr-xr-x 28 ubuntu ubuntu  4096 Jun 17 15:42 ../
-rwxrwxr-x  1 ubuntu ubuntu  1256 Jun  1 16:57 categorize_image.sh*
-rwxrwxr-x  1 ubuntu ubuntu    72 May 30 15:28 list_files_info_in_dir.sh*
-rw-r--r--  1 ubuntu ubuntu 66145 Jan 13 17:11 test1.jpg
-rw-r--r--  1 ubuntu ubuntu 85336 Jan 13 17:11 test2.jpg
`, nil
					}
					return "", fmt.Errorf("failed to get directory: %w", err)
				},
				`count_lines`: func(args map[string]any) (string, error) {
					filepath, err := FuncArg[string](args, "filepath")

					verbose(">>> filepath: %s", *filepath)

					if err == nil {
						// FIXME: hard-coded
						return `55
`, nil
					}
					return "", fmt.Errorf("failed to count lines: %w", err)
				},
			},
			contents,
			&GenerationOptions{
				Tools: []*genai.Tool{
					{
						FunctionDeclarations: []*genai.FunctionDeclaration{
							{
								Name:        `list_files_info_in_dir`,
								Description: `this function lists files' info in the given directory`,
								Parameters: &genai.Schema{
									Type: genai.TypeObject,
									Properties: map[string]*genai.Schema{
										"directory": {
											Type:        genai.TypeString,
											Description: `the absolute path of a directory`,
										},
									},
								},
							},
							{
								Name:        `count_lines`,
								Description: `this function counts the number of lines of given file`,
								Parameters: &genai.Schema{
									Type: genai.TypeObject,
									Properties: map[string]*genai.Schema{
										"filepath": {
											Type:        genai.TypeString,
											Description: `the absolute path of a file`,
										},
									},
								},
							},
						},
					},
				},
				ToolConfig: &genai.ToolConfig{
					FunctionCallingConfig: &genai.FunctionCallingConfig{
						Mode: genai.FunctionCallingConfigModeAuto,
					},
				},
			},
		); err != nil {
			t.Errorf("failed to generate with recursive tool calls: %s", err)
		} else {
			verbose(">>> response: %s", prettify(res))
		}
	}
}

// TestCountingTokens tests tokens counting.
func TestCountingTokens(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// open a file for testing
	file, err := os.Open("./client.go")
	if err != nil {
		t.Fatalf("failed to open file for counting tokens: %s", err)
	}
	defer func() { _ = file.Close() }()
	bytes, err := io.ReadAll(file)
	if err != nil {
		t.Fatalf("failed to read file for counting tokens")
	}

	if res, err := gtc.CountTokens(
		context.TODO(),
		[]*genai.Content{
			genai.NewContentFromText("Analyze this file.", RoleUser),
			genai.NewContentFromText("Provide a file for analysis.", RoleModel),
			genai.NewContentFromBytes(bytes, "text/plain", RoleUser),
		},
		&genai.CountTokensConfig{},
	); err != nil {
		t.Errorf("counting tokens failed: %s", ErrToStr(err))
	} else {
		verbose(">>> counted tokens: %s", prettify(res))
	}
}

// TestListingModels tests models listing.
func TestListingModels(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		// WithModel(modelForTextGeneration), // NOTE: `model` is not needed for some tasks (eg. listing models)
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	if models, err := gtc.ListModels(context.TODO()); err != nil {
		t.Errorf("listing models failed: %s", ErrToStr(err))
	} else {
		verbose(">>> listed models: %s", prettify(models))
	}
}

// TestFileSearch tests file search.
func TestFileSearch(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForTextGeneration),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	// create a file search store
	if store, err := gtc.CreateFileSearchStore(context.TODO(), "file-search-store-for-test"); err != nil {
		t.Errorf("failed to create file search store: %s", err)
	} else {
		// upload a file to file search store
		if file, err := os.Open(`./generation_test.go`); err == nil {
			defer func() { _ = file.Close() }()

			if _, err := gtc.UploadFileForSearch(
				context.TODO(),
				store.Name,
				file,
				"golang test file for gmn (generation_test.go)",
				[]*genai.CustomMetadata{
					{
						Key:         "filename",
						StringValue: "generation_test.go",
					},
				},
				nil,
			); err == nil {
				// gtc.waitForFilesForSearch(context.TODO(), []string{uploaded.Name}) // FIXME: not working yet
				time.Sleep(10 * time.Second)
			} else {
				t.Errorf("failed to upload file for search: %s", ErrToStr(err))
			}
		} else {
			t.Fatalf("failed to open file for file search store: %s", err)
		}

		// upload a file and import it to search store
		if file, err := os.Open(`./utils_test.go`); err == nil {
			defer func() { _ = file.Close() }()

			if uploaded, err := gtc.UploadFile(
				context.TODO(),
				file,
				"golang test file for gmn (utils_test.go)",
			); err == nil {
				// import to file search store
				if _, err := gtc.ImportFileForSearch(
					context.TODO(),
					store.Name,
					uploaded.Name,
					[]*genai.CustomMetadata{
						{
							Key:         "filename",
							StringValue: "utils_test.go",
						},
					},
					nil,
				); err == nil {
					// gtc.waitForFilesForSearch(context.TODO(), []string{uploaded.Name}) // FIXME: file? file search?
					time.Sleep(10 * time.Second)
				} else {
					t.Errorf("failed to import file to file search store: %s", ErrToStr(err))
				}
			} else {
				t.Errorf("failed to upload file: %s", ErrToStr(err))
			}
		} else {
			t.Fatalf("failed to open file: %s", err)
		}

		// list files in a search store
		for file, err := range gtc.ListFilesInFileSearchStore(
			context.TODO(),
			store.Name,
		) {
			if err != nil {
				t.Errorf("failed to list files in file search store: %s", ErrToStr(err))
			} else {
				verbose(">>> listed files in file search store: %s", prettify(file))
			}
		}

		// delete a file in a search store
		if file, err := os.Open(`./utils.go`); err == nil {
			defer func() { _ = file.Close() }()

			if uploaded, err := gtc.UploadFileForSearch(
				context.TODO(),
				store.Name,
				file,
				"utils for gmn (utils.go)",
				[]*genai.CustomMetadata{
					{
						Key:         "filename",
						StringValue: "utils.go",
					},
				},
				nil,
			); err == nil {
				// gtc.waitForFilesForSearch(context.TODO(), []string{uploaded.Name}) // FIXME: not working yet
				time.Sleep(10 * time.Second)

				if err := gtc.DeleteFileInFileSearchStore(
					context.TODO(),
					uploaded.Response.DocumentName,
				); err != nil {
					t.Errorf("failed to delete file in file search store: %s", ErrToStr(err))
				}
			} else {
				t.Errorf("failed to upload file for search: %s", ErrToStr(err))
			}
		} else {
			t.Fatalf("failed to open file for file search store: %s", err)
		}

		// generate with file search
		if contents, err := gtc.PromptsToContents(
			context.TODO(),
			[]Prompt{
				PromptFromText(`how many test cases are there in the 'generation_test.go' file?`),
			},
			nil,
		); err != nil {
			t.Errorf("failed to convert prompts to contents: %s", err)
		} else {
			if generated, err := gtc.Generate(context.TODO(), contents, &GenerationOptions{
				Tools: []*genai.Tool{
					{
						FileSearch: &genai.FileSearch{
							FileSearchStoreNames: []string{
								store.Name,
							},
						},
					},
				},
			}); err == nil {
				var promptTokenCount int32 = 0
				var cachedContentTokenCount int32 = 0
				if generated.UsageMetadata != nil {
					if generated.UsageMetadata.PromptTokenCount != 0 {
						promptTokenCount = generated.UsageMetadata.PromptTokenCount
					}
					if generated.UsageMetadata.CachedContentTokenCount != 0 {
						cachedContentTokenCount = generated.UsageMetadata.CachedContentTokenCount
					}
				}

				verbose(">>> input tokens: %d, output tokens: %d, cached tokens: %d (usage metadata)",
					promptTokenCount,
					generated.UsageMetadata.TotalTokenCount-promptTokenCount,
					cachedContentTokenCount,
				)

				verbose(">>> generated: %s", prettify(generated.Candidates[0]))
			} else {
				t.Errorf("generation with file search (non-streamed) failed: %s", ErrToStr(err))
			}

			// delete file search store
			if err := gtc.DeleteFileSearchStore(context.TODO(), store.Name); err != nil {
				t.Errorf("failed to delete file search store: %s", ErrToStr(err))
			}
		}
	}
}

// TestBatchRequests tests batch requests.
//
//	NOTE: may fail with error on free tier:
//	{
//		"code": 429,
//		"message": "Resource has been exhausted (e.g. check quota).",
//		"status": "RESOURCE_EXHAUSTED"
//	}
func TestBatchRequests(t *testing.T) {
	sleepForNotBeingRateLimited()

	apiKey := mustHaveEnvVar(t, "API_KEY")

	gtc, err := NewClient(
		apiKey,
		WithModel(modelForBatches),
		WithTimeoutSeconds(timeoutSecondsForTesting),
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	gtc.DeleteFilesOnClose = true
	defer func() { _ = gtc.Close() }()

	// inlined request
	if batch, err := gtc.RequestBatch(
		context.TODO(),
		&genai.BatchJobSource{
			InlinedRequests: []*genai.InlinedRequest{
				{
					Model: modelForTextGeneration,
					Contents: []*genai.Content{
						{
							Role: genai.RoleUser,
							Parts: []*genai.Part{
								ptr(
									PromptFromText(
										`Give me a detailed explanation of the terminal velocity.`,
									).ToPart(),
								),
							},
						},
						{
							Role: genai.RoleUser,
							Parts: []*genai.Part{
								ptr(
									PromptFromText(
										`Show me how to solve the quadratic equation.`,
									).ToPart(),
								),
							},
						},
						{
							Role: genai.RoleUser,
							Parts: []*genai.Part{
								ptr(
									PromptFromText(
										`How can I calculate the escape velocity from the Earth?`,
									).ToPart(),
								),
							},
						},
					},
					/*
						// FIXME: not working (yet?)
						Config: &genai.GenerateContentConfig{
							SystemInstruction: &genai.Content{
								Role: genai.RoleModel,
								Parts: []*genai.Part{
									ptr(PromptFromText(`You are a helpful assistant.`).ToPart()),
								},
							},
						},
					*/
				},
			},
		},
		"test-batch-request-with-inlined",
	); err != nil {
		t.Errorf("batch request failed: %s", ErrToStr(err))
	} else {
		verbose(">>> batch request: %s", prettify(batch))

		if got, err := gtc.Batch(
			context.TODO(),
			batch.Name,
		); err != nil {
			t.Errorf("failed to get batch: %s", ErrToStr(err))
		} else {
			verbose(">>> batch status: %s", prettify(got.State))

			if err := gtc.CancelBatch(
				context.TODO(),
				got.Name,
			); err != nil {
				t.Errorf("failed to cancel batch: %s", ErrToStr(err))
			} else {
				verbose(">>> batch canceled")

				if err := gtc.DeleteBatch(
					context.TODO(),
					got.Name,
				); err != nil {
					t.Errorf("failed to delete batch: %s", ErrToStr(err))
				} else {
					verbose(">>> batch deleted")
				}
			}
		}
	}

	// file request (JSONL)
	if bytes, err := json.Marshal(struct {
		Contents []*genai.Content `json:"contents"`
	}{
		Contents: []*genai.Content{
			{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					ptr(
						PromptFromText(
							`What are the three laws of thermodynamics?`,
						).ToPart(),
					),
				},
			},
		},
	}); err != nil {
		t.Fatalf("failed to marshal jsonl: %s", err)
	} else {
		if uploaded, err := gtc.UploadFilesAndWait(
			context.TODO(),
			[]Prompt{
				PromptFromBytes(bytes),
			},
			true, // NOTE: for not checking mime types of uploaded files
		); err != nil {
			t.Errorf("failed to upload json file: %s", ErrToStr(err))
		} else {
			first := uploaded[0]

			var filename string
			switch typ := first.(type) {
			case FilePrompt:
				filename = typ.Filename
			}

			if filename != "" {
				if batch, err := gtc.RequestBatch(
					context.TODO(),
					&genai.BatchJobSource{
						// Format:   "jsonl", // FIXME: not supported in gemini?
						FileName: filename,
					},
					"test-batch-request-with-file",
				); err != nil {
					t.Errorf("batch request failed: %s", ErrToStr(err))
				} else {
					verbose(">>> batch request: %s", prettify(batch))

					if got, err := gtc.Batch(
						context.TODO(),
						batch.Name,
					); err != nil {
						t.Errorf("failed to get batch: %s", ErrToStr(err))
					} else {
						verbose(">>> batch status: %s", prettify(got.State))

						if err := gtc.CancelBatch(
							context.TODO(),
							got.Name,
						); err != nil {
							t.Errorf("failed to cancel batch: %s", ErrToStr(err))
						} else {
							verbose(">>> batch canceled")

							if err := gtc.DeleteBatch(
								context.TODO(),
								got.Name,
							); err != nil {
								t.Errorf("failed to delete batch: %s", ErrToStr(err))
							} else {
								verbose(">>> batch deleted")
							}
						}
					}
				}
			} else {
				t.Errorf("failed to get uploaded json file, got: %+v", first)
			}
		}
	}

	// embeddings batch request
	gtc.model = modelForEmbeddings
	if batch, err := gtc.RequestBatchEmbeddings(
		context.TODO(),
		&genai.EmbeddingsBatchJobSource{
			InlinedRequests: &genai.EmbedContentBatch{
				Contents: []*genai.Content{
					{
						Parts: []*genai.Part{
							{
								Text: `Life is like riding a bicycle. To keep your balance, you must keep moving.\n- Albert Einstein`,
							},
						},
					},
				},
				Config: &genai.EmbedContentConfig{},
			},
		},
		"test-embeddings-batch-request",
	); err != nil {
		t.Errorf("embeddings batch request failed: %s", ErrToStr(err))
	} else {
		verbose(">>> embeddings batch request: %s", prettify(batch))

		if got, err := gtc.Batch(
			context.TODO(),
			batch.Name,
		); err != nil {
			t.Errorf("failed to get batch: %s", ErrToStr(err))
		} else {
			verbose(">>> batch status: %s", prettify(got.State))

			if err := gtc.CancelBatch(
				context.TODO(),
				got.Name,
			); err != nil {
				t.Errorf("failed to cancel batch: %s", ErrToStr(err))
			} else {
				verbose(">>> batch canceled")

				if err := gtc.DeleteBatch(
					context.TODO(),
					got.Name,
				); err != nil {
					t.Errorf("failed to delete batch: %s", ErrToStr(err))
				} else {
					verbose(">>> batch deleted")
				}
			}
		}
	}
}
