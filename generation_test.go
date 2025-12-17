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

	"cloud.google.com/go/auth"
	"cloud.google.com/go/auth/credentials"
)

const (
	timeoutSecondsForTesting = 60

	// keys for environment variables
	keyVerbose                   = `VERBOSE`
	keyGeminiAPIKey              = `GEMINI_API_KEY`
	keyProjectID                 = `PROJECT_ID`
	keyLocation                  = `LOCATION`
	KeyVertexCredentialsFilepath = `GOOGLE_APPLICATION_CREDENTIALS`

	defaultLocation = `global`
)

// flag for verbose log
var _isVerbose bool

// variables for API credentials
var (
	// gemini API
	_geminiAPIKey string

	// vertex API
	_projectID, _location      string
	_vertexCredentialsFilepath string
)

func TestMain(m *testing.M) {
	_isVerbose = os.Getenv(keyVerbose) == "true"

	_geminiAPIKey = os.Getenv(keyGeminiAPIKey)
	_vertexCredentialsFilepath = os.Getenv(KeyVertexCredentialsFilepath)
	_projectID = os.Getenv(keyProjectID)
	_location = os.Getenv(keyLocation)
	if _location == "" {
		_location = defaultLocation
	}

	os.Exit(m.Run())
}

// create context with timeout for testing
func ctxWithTimeout() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), timeoutSecondsForTesting*time.Second)
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

// create a new client with possible credentials
func newClient(opts ...ClientOption) (*Client, error) {
	if _geminiAPIKey != "" {
		return NewClient(_geminiAPIKey, opts...)
	} else if _vertexCredentialsFilepath != "" {
		var err error
		var bytes []byte
		if bytes, err = os.ReadFile(_vertexCredentialsFilepath); err == nil {
			var creds *auth.Credentials
			if creds, err = credentials.NewCredentialsFromJSON(
				credentials.ServiceAccount,
				bytes,
				&credentials.DetectOptions{
					Scopes: []string{"https://www.googleapis.com/auth/cloud-platform"},
				},
			); err == nil {
				return NewVertextClient(_projectID, _location, creds, opts...)
			}
		}

		return nil, fmt.Errorf("failed to create client with credentials: %w", err)
	}

	return nil, fmt.Errorf("no API key(%s) or credentials file(%s) provided for test", keyGeminiAPIKey, KeyVertexCredentialsFilepath)
}

// TestListingModels tests models listing.
func TestListingModelsFree(t *testing.T) {
	sleepForNotBeingRateLimited()

	gtc, err := newClient(
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
