// generation_test.go
//
// test cases for testing various types of generations

package gt

import (
	"context"
	"log"
	"os"
	"testing"
	"time"
)

const (
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

// TestListingModels tests models listing.
func TestListingModelsFree(t *testing.T) {
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
