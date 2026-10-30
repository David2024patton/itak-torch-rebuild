package registry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// DefaultClientTimeout is the fallback HTTP timeout for a registry Client
// when ClientConfig leaves Timeout unset. It is a default the caller can
// override through configuration, not a compiled-in limit: NewClientWithConfig
// honors any positive Timeout the caller supplies.
const DefaultClientTimeout = 10 * time.Second

// ClientConfig carries every operational setting a registry Client needs.
type ClientConfig struct {
	// BaseURL is the root URL of the registry server (no trailing path).
	BaseURL string

	// AuthToken, when non-empty, is sent as a Bearer token on every request.
	AuthToken string

	// Timeout bounds each HTTP request. Zero selects DefaultClientTimeout.
	Timeout time.Duration
}

// timeout resolves the effective request timeout: the configured value when
// one is set, the package default otherwise.
func (cfg ClientConfig) timeout() time.Duration {
	if cfg.Timeout > 0 {
		return cfg.Timeout
	}
	return DefaultClientTimeout
}

// Client provides remote access to a registry server over HTTP.
// Modules running on different machines use this instead of a direct
// *ServiceRegistry reference.
type Client struct {
	baseURL    string
	httpClient *http.Client
	authToken  string
}

// NewClient creates a registry client pointing at the broker's HTTP API,
// using DefaultClientTimeout for requests.
func NewClient(baseURL, authToken string) *Client {
	return NewClientWithConfig(ClientConfig{BaseURL: baseURL, AuthToken: authToken})
}

// NewClientWithConfig creates a registry client from explicit configuration.
func NewClientWithConfig(cfg ClientConfig) *Client {
	return &Client{
		baseURL:   cfg.BaseURL,
		authToken: cfg.AuthToken,
		httpClient: &http.Client{
			Timeout: cfg.timeout(),
		},
	}
}

// Register sends a service endpoint registration to the remote registry.
func (c *Client) Register(ep ServiceEndpoint) error {
	body, err := json.Marshal(ep)
	if err != nil {
		return fmt.Errorf("marshal endpoint: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, c.baseURL+"/registry/register", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("register request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("register: status %d", resp.StatusCode)
	}
	return nil
}

// Discover fetches all healthy endpoints for a module from the remote registry.
func (c *Client) Discover(module string) ([]ServiceEndpoint, error) {
	req, err := http.NewRequest(http.MethodGet, c.baseURL+"/registry/discover/"+module, nil)
	if err != nil {
		return nil, err
	}
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("discover request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("discover: status %d", resp.StatusCode)
	}

	var endpoints []ServiceEndpoint
	if err := json.NewDecoder(resp.Body).Decode(&endpoints); err != nil {
		return nil, fmt.Errorf("decode discover response: %w", err)
	}
	return endpoints, nil
}

// Heartbeat sends a heartbeat for an endpoint to the remote registry.
func (c *Client) Heartbeat(module, host string, port int) error {
	body, _ := json.Marshal(struct {
		Module string `json:"module"`
		Host   string `json:"host"`
		Port   int    `json:"port"`
	}{Module: module, Host: host, Port: port})

	req, err := http.NewRequest(http.MethodPost, c.baseURL+"/registry/heartbeat", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("heartbeat request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("heartbeat: status %d", resp.StatusCode)
	}
	return nil
}

// Deregister removes an endpoint from the remote registry.
func (c *Client) Deregister(module, host string, port int) error {
	body, _ := json.Marshal(struct {
		Module string `json:"module"`
		Host   string `json:"host"`
		Port   int    `json:"port"`
	}{Module: module, Host: host, Port: port})

	req, err := http.NewRequest(http.MethodDelete, c.baseURL+"/registry/deregister", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("deregister request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("deregister: status %d", resp.StatusCode)
	}
	return nil
}

// All returns every registered endpoint from the remote registry.
func (c *Client) All() (map[string][]ServiceEndpoint, error) {
	req, err := http.NewRequest(http.MethodGet, c.baseURL+"/registry/all", nil)
	if err != nil {
		return nil, err
	}
	if c.authToken != "" {
		req.Header.Set("Authorization", "Bearer "+c.authToken)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("all request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("all: status %d", resp.StatusCode)
	}

	var result map[string][]ServiceEndpoint
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode all response: %w", err)
	}
	return result, nil
}
