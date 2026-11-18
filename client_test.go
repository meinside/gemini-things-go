// client_test.go
//
// test cases for client.go

package gt

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

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

// TestWithAPIVersion tests the API version handed to the genai client.
func TestWithAPIVersion(t *testing.T) {
	for _, tc := range []struct {
		name   string
		env    string
		opts   []ClientOption
		expect string
	}{
		{name: "SDK default", expect: apiVersionStable},
		{name: "environment variable", env: "beta", expect: apiVersionBeta},
		{name: "option", opts: []ClientOption{WithAPIVersion(apiVersionBeta)}, expect: apiVersionBeta},
		{name: "option over environment variable", env: "beta", opts: []ClientOption{WithAPIVersion(apiVersionStable)}, expect: apiVersionStable},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_GENAI_API_VERSION", tc.env)

			gtc, err := NewClient(`dummy-api-key`, tc.opts...)
			if err != nil {
				t.Fatalf("failed to create client: %s", err)
			}
			defer func() { _ = gtc.Close() }()

			if got := gtc.client.ClientConfig().HTTPOptions.APIVersion; got != tc.expect {
				t.Errorf("unexpected API version ('%s' != '%s')", got, tc.expect)
			}
		})
	}
}

// TestNeedsBetaAPI tests the detection of `v1beta`-only generation configs.
func TestNeedsBetaAPI(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config *genai.GenerateContentConfig
		expect bool
	}{
		{name: "nil", config: nil, expect: false},
		{name: "empty", config: &genai.GenerateContentConfig{}, expect: false},
		{name: "text and image modalities", config: &genai.GenerateContentConfig{ResponseModalities: []string{string(genai.ModalityText), string(genai.ModalityImage)}}, expect: false},
		{name: "tools", config: &genai.GenerateContentConfig{Tools: []*genai.Tool{{GoogleSearch: &genai.GoogleSearch{}}}}, expect: false},
		{name: "cached content", config: &genai.GenerateContentConfig{CachedContent: `cachedContents/test`}, expect: true},
		{name: "speech config", config: &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}, expect: true},
		{name: "audio modality", config: &genai.GenerateContentConfig{ResponseModalities: []string{string(genai.ModalityAudio)}}, expect: true},
		{name: "audio transcription config", config: &genai.GenerateContentConfig{AudioTranscriptionConfig: &genai.AudioTranscriptionConfig{}}, expect: true},
		{name: "labels", config: &genai.GenerateContentConfig{Labels: map[string]string{"k": "v"}}, expect: true},
	} {
		if got := needsBetaAPI(tc.config); got != tc.expect {
			t.Errorf("[%s] unexpected result (%v != %v)", tc.name, got, tc.expect)
		}
	}
}

// pathRecorder is a test server which records the paths of requests.
type pathRecorder struct {
	*httptest.Server

	lock  sync.Mutex
	paths []string
}

func newPathRecorder(t *testing.T, status int) *pathRecorder {
	rec := &pathRecorder{}
	rec.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rec.lock.Lock()
		rec.paths = append(rec.paths, r.URL.Path)
		rec.lock.Unlock()

		// one body for every kind of response: cached content, list of cached contents,
		// list of models (without `models`, the SDK logs a warning), and others
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write([]byte(`{"name":"cachedContents/test","cachedContents":[{"name":"cachedContents/test"}],"models":[]}`))
	}))
	t.Cleanup(rec.Close)
	return rec
}

// take returns the recorded paths and clears them.
func (r *pathRecorder) take() []string {
	r.lock.Lock()
	defer r.lock.Unlock()
	paths := r.paths
	r.paths = nil
	return paths
}

// newClientForRecorder creates a Gemini API client which sends requests to `rec`.
func newClientForRecorder(t *testing.T, rec *pathRecorder, apiVersion string, maxRetryCount uint) *Client {
	t.Setenv("GOOGLE_GENAI_API_VERSION", "")

	gtc, err := NewClient(`dummy-api-key`, WithModel(`test-model`), WithAPIVersion(apiVersion))
	if err != nil {
		t.Fatalf("failed to create client: %s", err)
	}
	gtc.SetSystemInstructionFunc(nil)
	gtc.SetMaxRetryCount(maxRetryCount)

	// point the client at the test server, and drop the backoff so the test is fast
	gtc.client, err = genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey:  `dummy-api-key`,
		Backend: genai.BackendGeminiAPI,
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    rec.URL,
			APIVersion: apiVersion,
			RetryOptions: &genai.HTTPRetryOptions{
				Attempts:     gtc.retryAttempts,
				InitialDelay: new(0.0),
				MaxDelay:     new(0.0),
				Jitter:       new(0.0),
			},
		},
	})
	if err != nil {
		t.Fatalf("failed to create genai client: %s", err)
	}
	return gtc
}

// TestAPIVersionsOfRequests tests that requests using `v1beta`-only features
// are sent with `v1beta`, and the others with the client's API version.
//
// `beta` of each call follows `needsBetaAPI` and `cachingHTTPOptions`:
// update it along with them when a feature moves to `v1`.
func TestAPIVersionsOfRequests(t *testing.T) {
	speech := &genai.GenerateContentConfig{SpeechConfig: &genai.SpeechConfig{}}
	contents := genai.Text(`test`)

	calls := []struct {
		name string
		beta bool // whether it uses `v1beta`-only features
		call func(t *testing.T, gtc *Client)
	}{
		{"Generate", false, func(t *testing.T, gtc *Client) {
			_, _ = gtc.Generate(t.Context(), contents)
		}},
		{"Generate with cached content", true, func(t *testing.T, gtc *Client) {
			_, _ = gtc.Generate(t.Context(), contents, &genai.GenerateContentConfig{CachedContent: `cachedContents/test`})
		}},
		{"Generate with speech config", true, func(t *testing.T, gtc *Client) {
			_, _ = gtc.Generate(t.Context(), contents, speech)
		}},
		{"GenerateStreamIterated with speech config", true, func(t *testing.T, gtc *Client) {
			for range gtc.GenerateStreamIterated(t.Context(), contents, speech) {
			}
		}},
		{"CacheContext", true, func(t *testing.T, gtc *Client) {
			_, _ = gtc.CacheContext(t.Context(), nil, []Prompt{PromptFromText(`test`)}, nil, nil, nil)
		}},
		{"SetCachedContextTTL", true, func(t *testing.T, gtc *Client) {
			_ = gtc.SetCachedContextTTL(t.Context(), `cachedContents/test`, time.Hour)
		}},
		{"SetCachedContextExpireTime", true, func(t *testing.T, gtc *Client) {
			_ = gtc.SetCachedContextExpireTime(t.Context(), `cachedContents/test`, time.Now().Add(time.Hour))
		}},
		{"ListAllCachedContexts", true, func(t *testing.T, gtc *Client) {
			_, _ = gtc.ListAllCachedContexts(t.Context())
		}},
		{"DeleteAllCachedContexts", true, func(t *testing.T, gtc *Client) {
			_ = gtc.DeleteAllCachedContexts(t.Context())
		}},
		{"RequestBatch", false, func(t *testing.T, gtc *Client) {
			_, _ = gtc.RequestBatch(t.Context(), &genai.BatchJobSource{
				InlinedRequests: []*genai.InlinedRequest{{Contents: contents}},
			})
		}},
		{"RequestBatch with speech config", true, func(t *testing.T, gtc *Client) {
			_, _ = gtc.RequestBatch(t.Context(), &genai.BatchJobSource{
				InlinedRequests: []*genai.InlinedRequest{{Contents: contents, Config: speech}},
			})
		}},
		{"ListModels", false, func(t *testing.T, gtc *Client) {
			_, _ = gtc.ListModels(t.Context())
		}},
	}

	for _, clientVersion := range []string{apiVersionStable, apiVersionBeta} {
		rec := newPathRecorder(t, http.StatusOK)
		gtc := newClientForRecorder(t, rec, clientVersion, 0)

		for _, c := range calls {
			c.call(t, gtc)

			// `v1beta`-only features are sent with `v1beta`, the others with the client's version
			expected := clientVersion
			if c.beta {
				expected = apiVersionBeta
			}

			paths := rec.take()
			if len(paths) == 0 {
				t.Errorf("[%s, client %s] no request was sent", c.name, clientVersion)
			}
			for _, path := range paths {
				if !strings.HasPrefix(path, "/"+expected+"/") {
					t.Errorf("[%s, client %s] expected a request with %s, but got: %s", c.name, clientVersion, expected, path)
				}
			}
		}

		// the speech config passed as an option should not be mutated
		if speech.HTTPOptions != nil {
			t.Errorf("[client %s] the caller's config should not be mutated", clientVersion)
		}
	}
}

// TestBetaHTTPOptions tests the HTTP options for requests using `v1beta`-only features.
func TestBetaHTTPOptions(t *testing.T) {
	rec := newPathRecorder(t, http.StatusOK)
	gtc := newClientForRecorder(t, rec, apiVersionStable, 0)

	// the caller's options should be kept, and not be mutated
	caller := &genai.HTTPOptions{Headers: http.Header{"X-Test": []string{"test"}}}
	if got := gtc.betaHTTPOptions(caller); got.APIVersion != apiVersionBeta {
		t.Errorf("expected API version %s, but got '%s'", apiVersionBeta, got.APIVersion)
	} else if got.Headers.Get("X-Test") != "test" {
		t.Errorf("expected the caller's headers to be kept")
	}
	if caller.APIVersion != "" {
		t.Errorf("the caller's options should not be mutated")
	}

	// an API version set by the caller should take precedence
	if got := gtc.betaHTTPOptions(&genai.HTTPOptions{APIVersion: `v1alpha`}); got.APIVersion != `v1alpha` {
		t.Errorf("expected the caller's API version to be kept, but got '%s'", got.APIVersion)
	}

	// Vertex AI should not be affected
	vertex := &Client{Type: genai.BackendVertexAI}
	if got := vertex.betaHTTPOptions(nil); got != nil {
		t.Errorf("expected no HTTP options for Vertex AI, but got: %+v", got)
	}
}

// TestRetriesOfBetaRequests tests that client-level retries still apply
// to requests sent with `v1beta`.
func TestRetriesOfBetaRequests(t *testing.T) {
	rec := newPathRecorder(t, http.StatusServiceUnavailable)
	gtc := newClientForRecorder(t, rec, apiVersionStable, 2)

	_ = gtc.DeleteCachedContext(t.Context(), `cachedContents/test`)

	// 2 retries => 3 attempts
	paths := rec.take()
	if len(paths) != 3 {
		t.Errorf("expected 3 attempts for a retry count of 2, but got %d", len(paths))
	}
	for _, path := range paths {
		if !strings.HasPrefix(path, "/"+apiVersionBeta+"/") {
			t.Errorf("expected a request with %s, but got: %s", apiVersionBeta, path)
		}
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
