package registry

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Server wraps a ServiceRegistry with an HTTP API so it can be accessed
// over the network. The broker runs this alongside the tunnel listener.
//
// Endpoints:
//
//	POST   /registry/register        Register or update an endpoint
//	DELETE /registry/deregister      Remove an endpoint
//	GET    /registry/discover/{mod}  Find healthy endpoints for a module
//	GET    /registry/all             List all registered endpoints
//	POST   /registry/heartbeat       Heartbeat an endpoint
type Server struct {
	registry *ServiceRegistry
	mux      *http.ServeMux
}

// NewServer creates an HTTP server wrapping the given registry.
func NewServer(reg *ServiceRegistry) *Server {
	s := &Server{
		registry: reg,
		mux:      http.NewServeMux(),
	}
	s.mux.HandleFunc("/registry/register", s.handleRegister)
	s.mux.HandleFunc("/registry/deregister", s.handleDeregister)
	s.mux.HandleFunc("/registry/discover/", s.handleDiscover)
	s.mux.HandleFunc("/registry/all", s.handleAll)
	s.mux.HandleFunc("/registry/heartbeat", s.handleHeartbeat)
	return s
}

// Handler returns the HTTP handler for mounting in an existing server.
func (s *Server) Handler() http.Handler {
	return s.mux
}

// handleRegister accepts a ServiceEndpoint JSON body and registers it.
func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var ep ServiceEndpoint
	if err := json.NewDecoder(r.Body).Decode(&ep); err != nil {
		http.Error(w, fmt.Sprintf("invalid body: %v", err), http.StatusBadRequest)
		return
	}

	s.registry.Register(ep)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "registered"})
}

// handleDeregister removes an endpoint.
func (s *Server) handleDeregister(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Module string `json:"module"`
		Host   string `json:"host"`
		Port   int    `json:"port"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid body: %v", err), http.StatusBadRequest)
		return
	}

	s.registry.Deregister(req.Module, req.Host, req.Port)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "deregistered"})
}

// handleDiscover returns healthy endpoints for the module in the URL path.
// Path format: /registry/discover/{module}
func (s *Server) handleDiscover(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract module from path
	module := strings.TrimPrefix(r.URL.Path, "/registry/discover/")
	if module == "" {
		http.Error(w, "module name required", http.StatusBadRequest)
		return
	}

	endpoints := s.registry.Discover(module)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(endpoints)
}

// handleAll returns all registered endpoints.
func (s *Server) handleAll(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	all := s.registry.All()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(all)
}

// handleHeartbeat updates the LastSeen timestamp for an endpoint.
func (s *Server) handleHeartbeat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Module string `json:"module"`
		Host   string `json:"host"`
		Port   int    `json:"port"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid body: %v", err), http.StatusBadRequest)
		return
	}

	s.registry.Heartbeat(req.Module, req.Host, req.Port)
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}
