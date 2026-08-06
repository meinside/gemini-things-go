// client_test.go
//
// test cases for client.go

package gt

import (
	"testing"

	"google.golang.org/genai"
)

// TestAlteredGenerateContentConfigRetryOptions tests the injection of
// the SDK's HTTP retry options into generate-content configs.
func TestAlteredGenerateContentConfigRetryOptions(t *testing.T) {
	const maxRetryCount uint = 3
	const wantAttempts int32 = int32(maxRetryCount) + 1 // `Attempts` includes the initial request

	c := &Client{maxRetryCount: maxRetryCount}

	// (1) with a nil config, retry options should be injected
	if altered := c.alteredGenerateContentConfig(nil); altered.HTTPOptions == nil {
		t.Errorf("expected HTTPOptions to be set for a nil config")
	} else if altered.HTTPOptions.RetryOptions == nil {
		t.Errorf("expected RetryOptions to be set for a nil config")
	} else if altered.HTTPOptions.RetryOptions.Attempts == nil {
		t.Errorf("expected Attempts to be set for a nil config")
	} else if got := *altered.HTTPOptions.RetryOptions.Attempts; got != wantAttempts {
		t.Errorf("unexpected Attempts for a nil config (%d != %d)", got, wantAttempts)
	}

	// (2) a caller-supplied RetryOptions should take precedence
	const callerAttempts int32 = 10
	callerRetries := &genai.GenerateContentConfig{
		HTTPOptions: &genai.HTTPOptions{
			RetryOptions: &genai.HTTPRetryOptions{
				Attempts: new(callerAttempts),
			},
		},
	}
	if altered := c.alteredGenerateContentConfig(callerRetries); altered.HTTPOptions.RetryOptions.Attempts == nil {
		t.Errorf("expected the caller's Attempts to be kept")
	} else if got := *altered.HTTPOptions.RetryOptions.Attempts; got != callerAttempts {
		t.Errorf("the caller's Attempts should not be overridden (%d != %d)", got, callerAttempts)
	}

	// (3) other HTTPOptions fields should survive the injection
	const apiVersion = `v1beta`
	callerHTTPOptions := &genai.GenerateContentConfig{
		HTTPOptions: &genai.HTTPOptions{
			APIVersion: apiVersion,
		},
	}
	altered := c.alteredGenerateContentConfig(callerHTTPOptions)
	if altered.HTTPOptions.APIVersion != apiVersion {
		t.Errorf("expected APIVersion to be kept ('%s' != '%s')", altered.HTTPOptions.APIVersion, apiVersion)
	}
	if altered.HTTPOptions.RetryOptions == nil {
		t.Errorf("expected RetryOptions to be injected into the existing HTTPOptions")
	}

	// (4) the caller's HTTPOptions should not be mutated
	if callerHTTPOptions.HTTPOptions.RetryOptions != nil {
		t.Errorf("the caller's HTTPOptions should not be mutated")
	}

	// (5) a zero retry count should mean a single attempt
	noRetries := &Client{maxRetryCount: 0}
	if altered := noRetries.alteredGenerateContentConfig(nil); altered.HTTPOptions.RetryOptions.Attempts == nil {
		t.Errorf("expected Attempts to be set for a zero retry count")
	} else if got := *altered.HTTPOptions.RetryOptions.Attempts; got != 1 {
		t.Errorf("expected 1 attempt for a zero retry count, but got %d", got)
	}
}
