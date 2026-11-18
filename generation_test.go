// generation_test.go
//
// test cases for testing various types of generations

package gt

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"google.golang.org/genai"
)

const (
	timeoutSecondsForTesting                = 60
	timeoutSecondsForTestingSlowGeneration  = 60 * 3 // 3 minutes, for generations with several round trips or tools (eg. grounding)
	timeoutSecondsForTestingVideoGeneration = 60 * 5 // 5 minutes

	// keys for environment variables
	keyVerbose                   = `VERBOSE`
	keyGeminiAPIKey              = `GEMINI_API_KEY`
	keyProjectID                 = `PROJECT_ID`
	keyLocation                  = `LOCATION`
	keyBucketName                = `BUCKET_NAME`
	keyVertexCredentialsFilepath = `CREDENTIALS_FILEPATH`

	defaultLocation              = `global`
	defaultLocationForEmbeddings = `global`
)

// flag for verbose log
var _isVerbose bool

// variables for API credentials
var (
	// gemini API
	_geminiAPIKey string

	// vertex API
	_projectID, _location, _bucketName string
	_vertexCredentialsFilepath         string
)

func TestMain(m *testing.M) {
	_isVerbose = os.Getenv(keyVerbose) == "true"

	_geminiAPIKey = os.Getenv(keyGeminiAPIKey)
	_vertexCredentialsFilepath = os.Getenv(keyVertexCredentialsFilepath)
	_projectID = os.Getenv(keyProjectID)
	_location = os.Getenv(keyLocation)
	if _location == "" {
		_location = defaultLocation
	}
	_bucketName = os.Getenv(keyBucketName)
	if _bucketName == "" {
		_bucketName = defaultBucketName
	}

	os.Exit(m.Run())
}

// create context with timeout for testing
func ctxWithTimeout(seconds ...int) (context.Context, context.CancelFunc) {
	secondsToWait := time.Duration(timeoutSecondsForTesting)
	if len(seconds) > 0 {
		secondsToWait = time.Duration(seconds[0])
	}

	return context.WithTimeout(context.Background(), secondsToWait*time.Second)
}

// waitForBatchCancellation polls the batch job state until it leaves a non-terminal state
// (e.g. PENDING, RUNNING, CANCELLING) so that a subsequent DeleteBatch call won't be rejected.
func waitForBatchCancellation(t *testing.T, gtc *Client, name string) {
	const (
		pollInterval = 5 * time.Second
		maxAttempts  = 12 // up to ~60s
	)
	for i := 0; i < maxAttempts; i++ {
		ctx, cancel := ctxWithTimeout(10)
		got, err := gtc.Batch(ctx, name)
		cancel()
		if err == nil {
			switch got.State {
			case genai.JobStateCancelled, genai.JobStateSucceeded,
				genai.JobStateFailed, genai.JobStateExpired,
				genai.JobStatePartiallySucceeded:
				return
			}
		}
		time.Sleep(pollInterval)
	}
	t.Logf("warning: batch %s did not reach a terminal state in time", name)
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

// checkCodeExecutionParts checks the parts generated with code execution.
//
// A failed execution alone does not fail the test, as the model often fixes
// and reruns its code (eg. after a missing import): it fails only when no
// execution succeeded, or no text was generated.
func checkCodeExecutionParts(t *testing.T, label string, parts []*genai.Part) {
	t.Helper()

	var succeeded, texts int
	for _, part := range parts {
		switch {
		case len(part.Text) > 0:
			texts++
			verbose(">>> generated text (%s): %s", label, part.Text)
		case part.ExecutableCode != nil:
			verbose(">>> executable code (%s, %s):\n%s", label, part.ExecutableCode.Language, part.ExecutableCode.Code)
		case part.CodeExecutionResult != nil:
			if part.CodeExecutionResult.Outcome == genai.OutcomeOK {
				succeeded++
				verbose(">>> code output (%s): %s", label, part.CodeExecutionResult.Output)
			} else {
				t.Logf("code execution (%s) failed, possibly retried by the model: %s", label, prettify(part.CodeExecutionResult))
			}
		case part.InlineData != nil:
			verbose(">>> generated inline data (%s): %s (%d bytes)", label, part.InlineData.MIMEType, len(part.InlineData.Data))
		default:
			t.Errorf("wrong type of generated part (%s): (%T) %s", label, part, prettify(part))
		}
	}

	if succeeded == 0 {
		t.Errorf("no code execution (%s) succeeded", label)
	}
	if texts == 0 {
		t.Errorf("no text (%s) was generated", label)
	}
}

// create a new client with possible credentials
func newClient(location string, opts ...ClientOption) (*Client, error) {
	if _geminiAPIKey != "" {
		return NewClient(_geminiAPIKey, opts...)
	} else if _vertexCredentialsFilepath != "" {
		var err error
		var bytes []byte
		if bytes, err = os.ReadFile(_vertexCredentialsFilepath); err == nil {
			return NewVertexClient(context.TODO(), bytes, location, _bucketName, opts...)
		}

		return nil, fmt.Errorf("failed to create client with credentials: %w", err)
	}

	return nil, fmt.Errorf("no API key(%s) or credentials file(%s) provided for test", keyGeminiAPIKey, keyVertexCredentialsFilepath)
}

// TestListingModels tests models listing.
func TestListingModelsFree(t *testing.T) {
	sleepForNotBeingRateLimited()

	gtc, err := newClient(
		_location,
	// WithModel(modelForTextGeneration), // NOTE: `model` is not needed for some tasks (eg. listing models)
	)
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.Verbose = _isVerbose
	defer func() { _ = gtc.Close() }()

	ctx, cancel := ctxWithTimeout()
	defer cancel()

	if models, err := gtc.ListModels(ctx); err != nil {
		t.Errorf("listing models failed: %s", ErrToStr(err))
	} else {
		verbose(">>> listed models: %s", prettify(models))
	}
}
