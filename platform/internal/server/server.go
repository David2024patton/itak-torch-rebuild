// Package server is the platform's HTTP face.
//
// The shape is the correction to the old platform's serving layer:
//
//   - Binds where main tells it (127.0.0.1 by default there); this
//     package never widens the bind on its own.
//   - Every route except /healthz sits behind the auth middleware.
//   - Handlers validate requests fully before touching the engine, and
//     engine failures map to named, honest HTTP errors. No handler
//     fabricates a success: unconfigured engine = 503 engine_unavailable,
//     unimplemented engine op = 501 not_implemented.
//   - Requests flow through the scheduler's priority lanes and
//     non-streaming completions through the exact-match response cache.
//   - /debug (behind auth) serves the diag registry plus scheduler and
//     cache stats, so the running system can explain itself.
package server

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/David2024patton/itak-torch-rebuild/platform/internal/auth"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/cache"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/diag"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/engine"
	"github.com/David2024patton/itak-torch-rebuild/platform/internal/scheduler"
)

const maxBodyBytes = 1 << 20 // 1 MiB request cap

// Config wires the server. Zero values get safe defaults in New.
type Config struct {
	// Engine is the token engine client. Nil means none configured:
	// routes then refuse with engine_unavailable.
	Engine engine.Client
	// APIKey is required on every route except /healthz. Empty fails
	// closed (the middleware rejects everything).
	APIKey string
	// Models is the advertised model list for /v1/models.
	Models []string
	// CacheTTL is the response-cache retention; <= 0 disables caching.
	CacheTTL time.Duration
	// MaxConcurrent bounds simultaneous engine jobs; < 1 becomes 4.
	MaxConcurrent int
	// Logger receives handler diagnostics; nil uses slog.Default().
	Logger *slog.Logger
}

// Server is the wired HTTP handler plus its modules.
type Server struct {
	cfg   Config
	log   *slog.Logger
	eng   engine.Client
	diag  *diag.Registry
	cache *cache.Cache[[]byte]
	sched *scheduler.Scheduler
	mux   http.Handler
}

// New builds the route table and modules.
func New(cfg Config) *Server {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	if cfg.MaxConcurrent < 1 {
		cfg.MaxConcurrent = 4
	}
	eng := cfg.Engine
	if eng == nil {
		eng = engine.Unavailable{}
	}
	s := &Server{
		cfg:   cfg,
		log:   cfg.Logger,
		eng:   eng,
		diag:  diag.New(),
		cache: cache.New[[]byte](cfg.CacheTTL),
		sched: scheduler.New(cfg.MaxConcurrent),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", s.handleHealthz)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /v1/embeddings", s.handleEmbeddings)
	mux.HandleFunc("GET /debug", s.handleDebug)

	mw := auth.New(cfg.APIKey, s.diag, "/healthz")
	s.mux = mw.Wrap(mux)
	return s
}

// Handler returns the fully wrapped route table.
func (s *Server) Handler() http.Handler { return s.mux }

// Close stops the scheduler, failing any queued requests.
func (s *Server) Close() { s.sched.Close() }

// HTTPServer builds the http.Server with production timeouts. There is
// deliberately no WriteTimeout or ReadTimeout: generations stream for
// minutes. Header reads and idle conns are bounded.
func (s *Server) HTTPServer(addr string) *http.Server {
	return &http.Server{
		Addr:              addr,
		Handler:           s.mux,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       90 * time.Second,
	}
}

// --- shared helpers ----------------------------------------------------

type errorBody struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param,omitempty"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func (s *Server) writeError(w http.ResponseWriter, status int, code, message, param string) {
	s.diag.RecordError()
	writeJSON(w, status, errorBody{Error: apiError{Code: code, Message: message, Param: param}})
}

func (s *Server) invalidRequest(w http.ResponseWriter, message, param string) {
	s.writeError(w, http.StatusBadRequest, "invalid_request", message, param)
}

// engineErrorStatus maps a named engine error to its honest HTTP
// status and wire code. It is the single mapping both the plain
// error path and the streaming terminal-error frame use, so a
// failure carries the same name however it surfaces.
func engineErrorStatus(err error) (status int, code string) {
	switch {
	case errors.Is(err, engine.ErrUnavailable):
		return http.StatusServiceUnavailable, "engine_unavailable"
	case errors.Is(err, engine.ErrNotImplemented):
		return http.StatusNotImplemented, "not_implemented"
	case errors.Is(err, engine.ErrBadRequest):
		return http.StatusBadRequest, "invalid_request"
	case errors.Is(err, engine.ErrEngineDied):
		return http.StatusBadGateway, "engine_died"
	case errors.Is(err, engine.ErrEngineBusy):
		return http.StatusServiceUnavailable, "engine_busy"
	case errors.Is(err, engine.ErrGenerationFailed):
		return http.StatusBadGateway, "generation_failed"
	default:
		return http.StatusBadGateway, "engine_error"
	}
}

// engineError maps a named engine error to its honest HTTP form.
func (s *Server) engineError(w http.ResponseWriter, err error) {
	s.diag.RecordEngineError()
	status, code := engineErrorStatus(err)
	if status == http.StatusBadGateway && code == "engine_error" {
		s.log.Error("engine call failed", "err", err)
	}
	s.writeError(w, status, code, err.Error(), "")
}

func decodeBody(w http.ResponseWriter, r *http.Request, v any) error {
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("body is not valid JSON: %w", err)
	}
	return nil
}

func newID(prefix string) string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return prefix + "static"
	}
	return prefix + hex.EncodeToString(b[:])
}

func priorityOf(r *http.Request) scheduler.Priority {
	if strings.EqualFold(r.Header.Get("X-Torch-Priority"), "high") {
		return scheduler.PriorityHigh
	}
	return scheduler.PriorityNormal
}

// acquire takes a scheduler slot for an engine call, writing the error
// response itself when the slot cannot be had. ok=false means the
// handler must return immediately.
func (s *Server) acquire(w http.ResponseWriter, r *http.Request) (release func(), ok bool) {
	release, err := s.sched.Acquire(r.Context(), priorityOf(r))
	if err != nil {
		if errors.Is(err, scheduler.ErrClosed) {
			s.writeError(w, http.StatusServiceUnavailable, "scheduler_closed", "server is shutting down", "")
		}
		// Otherwise the client went away mid-wait; nothing to write to.
		return nil, false
	}
	return release, true
}

// --- handlers ----------------------------------------------------------

func (s *Server) handleHealthz(w http.ResponseWriter, r *http.Request) {
	s.diag.RecordRequest("GET /healthz")
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	h, err := s.eng.Health(ctx)
	if err != nil {
		h = engine.HealthStatus{OK: false, Detail: err.Error()}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ok",
		"engine": map[string]any{"ok": h.OK, "detail": h.Detail},
	})
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	s.diag.RecordRequest("GET /v1/models")
	data := make([]map[string]any, 0, len(s.cfg.Models))
	for _, m := range s.cfg.Models {
		data = append(data, map[string]any{"id": m, "object": "model", "owned_by": "itak-torch"})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "data": data})
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Stream      bool          `json:"stream"`
	MaxTokens   int           `json:"max_tokens"`
	Temperature *float64      `json:"temperature"`
	TopP        *float64      `json:"top_p"`
	Stop        []string      `json:"stop"`
	// Sampling parity fields (TODO T-9.13), OpenAI-shaped.
	Seed             *uint64  `json:"seed"`
	TopK             *int     `json:"top_k"`
	PresencePenalty  *float64 `json:"presence_penalty"`
	FrequencyPenalty *float64 `json:"frequency_penalty"`
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.diag.RecordRequest("POST /v1/chat/completions")
	var req chatRequest
	if err := decodeBody(w, r, &req); err != nil {
		s.invalidRequest(w, err.Error(), "")
		return
	}
	if msg := validateChat(&req); msg != nil {
		s.invalidRequest(w, msg.text, msg.param)
		return
	}
	greq := engine.GenerateRequest{
		Model:            req.Model,
		Prompt:           flattenMessages(req.Messages),
		MaxTokens:        req.MaxTokens,
		Stop:             req.Stop,
		Seed:             req.Seed,
		TopK:             req.TopK,
		PresencePenalty:  req.PresencePenalty,
		FrequencyPenalty: req.FrequencyPenalty,
	}
	if req.Temperature != nil {
		greq.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		greq.TopP = *req.TopP
	}

	if req.Stream {
		s.streamChat(w, r, greq)
		return
	}

	cacheKey := chatCacheKey(&req)
	if body, hit := s.cache.Get(cacheKey); hit {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Torch-Cache", "hit")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(body)
		return
	}

	release, ok := s.acquire(w, r)
	if !ok {
		return
	}
	defer release()

	s.diag.RecordEngineCall()
	ch, err := s.eng.Generate(r.Context(), greq)
	if err != nil {
		s.engineError(w, err)
		return
	}
	var sb strings.Builder
	finish := "stop"
	sawDone := false
	for ev := range ch {
		if ev.Err != nil {
			// Terminal error event (TODO T-9.12): the generation
			// failed after it started. Nothing has been written
			// yet on this path, so the failure answers with its
			// honest HTTP status instead of a truncated 200.
			s.engineError(w, ev.Err)
			return
		}
		sb.WriteString(ev.Text)
		if ev.Done {
			sawDone = true
			if ev.FinishReason != "" {
				finish = ev.FinishReason
			}
		}
	}
	if !sawDone {
		if r.Context().Err() != nil {
			// The stream died with the client; no honest response exists.
			s.diag.RecordEngineError()
			return
		}
		// The client closed the channel with neither a done nor an
		// error event: a contract violation, reported as an engine
		// death rather than served as a short completion.
		s.engineError(w, fmt.Errorf("%w: generation stream ended without a terminal event", engine.ErrEngineDied))
		return
	}

	resp := map[string]any{
		"id":      newID("chatcmpl-"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []map[string]any{{
			"index":         0,
			"message":       map[string]any{"role": "assistant", "content": sb.String()},
			"finish_reason": finish,
		}},
	}
	body, err := json.Marshal(resp)
	if err != nil {
		s.engineError(w, fmt.Errorf("marshal completion: %w", err))
		return
	}
	s.cache.Set(cacheKey, body)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Torch-Cache", "miss")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

func (s *Server) streamChat(w http.ResponseWriter, r *http.Request, greq engine.GenerateRequest) {
	release, ok := s.acquire(w, r)
	if !ok {
		return
	}
	defer release()

	s.diag.RecordEngineCall()
	ch, err := s.eng.Generate(r.Context(), greq)
	if err != nil {
		s.engineError(w, err)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.writeError(w, http.StatusInternalServerError, "streaming_unsupported", "response writer cannot stream", "")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	id := newID("chatcmpl-")
	created := time.Now().Unix()
	writeChunk := func(content string, finish *string) {
		chunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   greq.Model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"content": content},
				"finish_reason": finish,
			}},
		}
		b, _ := json.Marshal(chunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}
	for ev := range ch {
		if ev.Err != nil {
			// Terminal error event (TODO T-9.12): the 200 and any
			// chunks are already gone, so the named failure rides
			// the stream itself as an error frame and the stream
			// ends without [DONE]. A client reading only tokens
			// sees the failure by name, not a truncated answer.
			_, code := engineErrorStatus(ev.Err)
			s.diag.RecordEngineError()
			b, _ := json.Marshal(errorBody{Error: apiError{Code: code, Message: ev.Err.Error()}})
			_, _ = fmt.Fprintf(w, "data: %s\n\n", b)
			flusher.Flush()
			return
		}
		if ev.Done {
			fr := ev.FinishReason
			if fr == "" {
				fr = "stop"
			}
			writeChunk(ev.Text, &fr)
		} else {
			writeChunk(ev.Text, nil)
		}
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

type fieldError struct{ text, param string }

func validateChat(req *chatRequest) *fieldError {
	if req.Model == "" {
		return &fieldError{"model is required", "model"}
	}
	if len(req.Messages) == 0 {
		return &fieldError{"messages must not be empty", "messages"}
	}
	for i, m := range req.Messages {
		if m.Role == "" {
			return &fieldError{fmt.Sprintf("messages[%d].role is required", i), "messages"}
		}
		if m.Content == "" {
			return &fieldError{fmt.Sprintf("messages[%d].content is required", i), "messages"}
		}
	}
	if req.MaxTokens < 0 {
		return &fieldError{"max_tokens must be >= 0", "max_tokens"}
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		return &fieldError{"temperature must be between 0 and 2", "temperature"}
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return &fieldError{"top_p must be between 0 and 1", "top_p"}
	}
	if req.TopK != nil && *req.TopK < 0 {
		return &fieldError{"top_k must be >= 0", "top_k"}
	}
	if req.PresencePenalty != nil && (*req.PresencePenalty < -2 || *req.PresencePenalty > 2) {
		return &fieldError{"presence_penalty must be between -2 and 2", "presence_penalty"}
	}
	if req.FrequencyPenalty != nil && (*req.FrequencyPenalty < -2 || *req.FrequencyPenalty > 2) {
		return &fieldError{"frequency_penalty must be between -2 and 2", "frequency_penalty"}
	}
	return nil
}

func flattenMessages(msgs []chatMessage) string {
	var sb strings.Builder
	for _, m := range msgs {
		sb.WriteString(m.Role)
		sb.WriteString(": ")
		sb.WriteString(m.Content)
		sb.WriteString("\n")
	}
	sb.WriteString("assistant:")
	return sb.String()
}

func chatCacheKey(req *chatRequest) string {
	// Pointer fields marshal distinctly from absent ones, so temperature
	// 0 and no temperature do not collide.
	b, _ := json.Marshal(req)
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

type embedRequestBody struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
}

func (s *Server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	s.diag.RecordRequest("POST /v1/embeddings")
	var body embedRequestBody
	if err := decodeBody(w, r, &body); err != nil {
		s.invalidRequest(w, err.Error(), "")
		return
	}
	if body.Model == "" {
		s.invalidRequest(w, "model is required", "model")
		return
	}
	inputs, err := parseEmbedInput(body.Input)
	if err != nil {
		s.invalidRequest(w, err.Error(), "input")
		return
	}

	release, ok := s.acquire(w, r)
	if !ok {
		return
	}
	defer release()

	s.diag.RecordEngineCall()
	resp, err := s.eng.Embed(r.Context(), engine.EmbedRequest{Model: body.Model, Input: inputs})
	if err != nil {
		s.engineError(w, err)
		return
	}
	if len(resp.Vectors) != len(inputs) {
		s.engineError(w, fmt.Errorf("%w: engine returned %d vectors for %d inputs",
			engine.ErrBadRequest, len(resp.Vectors), len(inputs)))
		return
	}
	data := make([]map[string]any, 0, len(resp.Vectors))
	for i, v := range resp.Vectors {
		data = append(data, map[string]any{"object": "embedding", "index": i, "embedding": v})
	}
	writeJSON(w, http.StatusOK, map[string]any{"object": "list", "model": body.Model, "data": data})
}

func parseEmbedInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, errors.New("input is required")
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil, errors.New("input must not be empty")
		}
		return []string{single}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, errors.New("input must be a string or an array of strings")
	}
	if len(many) == 0 {
		return nil, errors.New("input must not be empty")
	}
	for _, s := range many {
		if s == "" {
			return nil, errors.New("input entries must not be empty")
		}
	}
	return many, nil
}

func (s *Server) handleDebug(w http.ResponseWriter, r *http.Request) {
	s.diag.RecordRequest("GET /debug")
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	h, herr := s.eng.Health(ctx)
	if herr != nil {
		h = engine.HealthStatus{OK: false, Detail: herr.Error()}
	}
	out := map[string]any{
		"diag":      s.diag.Snapshot(),
		"scheduler": s.sched.Stats(),
		"cache":     s.cache.Stats(),
		"engine":    map[string]any{"ok": h.OK, "detail": h.Detail},
		"models":    s.cfg.Models,
	}
	// When the wired engine client can report on itself (the socket
	// client can), its self-report rides along: loaded model,
	// coverage result, serving counters.
	if d, ok := s.eng.(interface {
		DebugInfo(context.Context) (map[string]any, error)
	}); ok {
		if info, err := d.DebugInfo(ctx); err == nil {
			out["engine_debug"] = info
		} else {
			out["engine_debug"] = map[string]any{"error": err.Error()}
		}
	}
	writeJSON(w, http.StatusOK, out)
}
