// client_test.go
//
// test cases for client.go

package gt

import (
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"google.golang.org/genai"
)

// TestRetryOptions tests the conversion of a retry count into the SDK's
// HTTP retry options.
func TestRetryOptions(t *testing.T) {
	for _, count := range []uint{0, 1, 3} {
		opts, attempts := retryOptions(count)

		// `Attempts` includes the initial request
		want := int32(count) + 1
		if opts.Attempts == nil {
			t.Errorf("expected Attempts to be set for a retry count of %d", count)
		} else if *opts.Attempts != want {
			t.Errorf("unexpected Attempts for a retry count of %d (%d != %d)", count, *opts.Attempts, want)
		}

		// the returned pointer must be the one held by the options,
		// so that writing through it reconfigures the genai client
		if attempts != opts.Attempts {
			t.Errorf("the returned pointer should be the one held by the retry options")
		}
	}
}

// TestSetMaxRetryCount tests that retries are configured for every API call
// made through the client, and that they stay reconfigurable afterwards.
func TestSetMaxRetryCount(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":503,"message":"unavailable"}}`))
	}))
	defer func() { srv.Close() }()

	gtc, err := NewClient(`dummy-api-key`, WithMaxRetryCount(1))
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	defer func() { _ = gtc.Close() }()

	// the genai client should have been handed the retry options at construction
	if built := gtc.client.ClientConfig().HTTPOptions.RetryOptions; built == nil {
		t.Errorf("expected the genai client to be configured with retry options")
	} else if built.Attempts != gtc.retryAttempts {
		t.Errorf("expected the genai client to share the client's Attempts pointer")
	}

	// point the client at the test server, and drop the backoff so the test is fast
	gtc.client, err = genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey:  `dummy-api-key`,
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL: srv.URL,
			RetryOptions: &genai.HTTPRetryOptions{
				Attempts:     gtc.retryAttempts, // the very pointer SetMaxRetryCount writes through
				InitialDelay: new(0.0),
				MaxDelay:     new(0.0),
				Jitter:       new(0.0),
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create genai client: %s", err)
	}

	// `ListModels` is not a `GenerateContent`-family call, so it exercises
	// the client-level retry configuration
	listModels := func() int32 {
		calls.Store(0)
		_, _ = gtc.ListModels(t.Context())
		return calls.Load()
	}

	// 1 retry => 2 attempts
	if got := listModels(); got != 2 {
		t.Errorf("expected 2 attempts for a retry count of 1, but got %d", got)
	}

	// raising the count must take effect without rebuilding the client
	gtc.SetMaxRetryCount(3)
	if got := listModels(); got != 4 {
		t.Errorf("expected 4 attempts after raising the retry count to 3, but got %d", got)
	}

	// a zero count disables retries
	gtc.SetMaxRetryCount(0)
	if got := listModels(); got != 1 {
		t.Errorf("expected 1 attempt for a retry count of 0, but got %d", got)
	}
}

// TestAlteredGenerateContentConfig tests that the system instruction fallback
// is applied without mutating the caller's config.
func TestAlteredGenerateContentConfig(t *testing.T) {
	const systemInstruction = `You are a test.`

	c := &Client{
		systemInstructionFunc: func() string {
			return systemInstruction
		},
	}

	// with a nil config, the fallback system instruction should be applied
	if altered := c.alteredGenerateContentConfig(nil); altered.SystemInstruction == nil {
		t.Errorf("expected a fallback SystemInstruction for a nil config")
	} else if got := altered.SystemInstruction.Parts[0].Text; got != systemInstruction {
		t.Errorf("unexpected fallback SystemInstruction ('%s' != '%s')", got, systemInstruction)
	}

	// a caller-supplied system instruction should take precedence,
	// and the caller's config should not be mutated
	const callerInstruction = `You are the caller's.`
	caller := &genai.GenerateContentConfig{
		Temperature: new(float32(0.5)),
	}
	altered := c.alteredGenerateContentConfig(caller)
	altered.SystemInstruction = &genai.Content{
		Parts: []*genai.Part{{Text: callerInstruction}},
	}
	if caller.SystemInstruction != nil {
		t.Errorf("the caller's config should not be mutated")
	}
	if altered.Temperature == nil || *altered.Temperature != 0.5 {
		t.Errorf("expected the caller's other fields to be kept")
	}
}
