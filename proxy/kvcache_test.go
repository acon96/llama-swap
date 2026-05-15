package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestKVCacheManager_SaveRestoreUsesManifestPerSlot(t *testing.T) {
	oldDir := kvCacheDir
	kvCacheDir = t.TempDir()
	defer func() {
		kvCacheDir = oldDir
	}()

	var mu sync.Mutex
	savedFilenames := make(map[int]string)
	restoredFilenames := make(map[int]string)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/slots":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode([]map[string]any{
				{"id": 0, "n_ctx": 4096, "speculative": false, "is_processing": false, "id_task": 0, "params": map[string]any{}, "next_token": map[string]any{}},
				{"id": 1, "n_ctx": 4096, "speculative": false, "is_processing": false, "id_task": 0, "params": map[string]any{}, "next_token": map[string]any{}},
			})
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/slots/"):
			var body struct {
				Filename string `json:"filename"`
			}
			require.NoError(t, json.NewDecoder(r.Body).Decode(&body))

			parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/slots/"), "?")
			slotID := 0
			_, err := fmt.Sscanf(parts[0], "%d", &slotID)
			require.NoError(t, err)

			action := r.URL.Query().Get("action")
			mu.Lock()
			switch action {
			case "save":
				savedFilenames[slotID] = body.Filename
				json.NewEncoder(w).Encode(map[string]any{"id_slot": slotID, "filename": body.Filename, "n_saved": 32, "n_written": 1024, "timings": map[string]any{"save_ms": 1.2}})
			case "restore":
				restoredFilenames[slotID] = body.Filename
				json.NewEncoder(w).Encode(map[string]any{"id_slot": slotID, "filename": body.Filename, "n_restored": 32, "n_read": 1024, "timings": map[string]any{"restore_ms": 0.8}})
			default:
				w.WriteHeader(http.StatusBadRequest)
			}
			mu.Unlock()
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	manager := newKVCacheManager("model/with:unsafe", server.URL, 60, debugLogger)
	defer manager.Stop()

	require.NoError(t, manager.save())

	files, err := os.ReadDir(kvCacheDir)
	require.NoError(t, err)
	assert.Len(t, files, 1)
	assert.True(t, strings.HasSuffix(files[0].Name(), ".json"))

	manifestBytes, err := os.ReadFile(filepath.Join(kvCacheDir, files[0].Name()))
	require.NoError(t, err)

	var manifest kvCacheManifest
	require.NoError(t, json.Unmarshal(manifestBytes, &manifest))
	assert.Len(t, manifest.Slots, 2)
	assert.NotContains(t, manifest.Slots[0].Filename, "/")
	assert.NotContains(t, manifest.Slots[0].Filename, ":")

	require.NoError(t, manager.restore())

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, savedFilenames, restoredFilenames)
	assert.Len(t, savedFilenames, 2)
}

func TestKVCacheManager_RestoreSkipsDifferentSlotShape(t *testing.T) {
	oldDir := kvCacheDir
	kvCacheDir = t.TempDir()
	defer func() {
		kvCacheDir = oldDir
	}()

	var mu sync.Mutex
	shape := 1
	restoreCalls := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/slots":
			w.Header().Set("Content-Type", "application/json")
			if shape == 1 {
				json.NewEncoder(w).Encode([]map[string]any{{"id": 0, "n_ctx": 16384, "speculative": false, "is_processing": false, "id_task": 0, "params": map[string]any{}, "next_token": map[string]any{}}})
			} else {
				json.NewEncoder(w).Encode([]map[string]any{
					{"id": 0, "n_ctx": 4096, "speculative": false, "is_processing": false, "id_task": 0, "params": map[string]any{}, "next_token": map[string]any{}},
					{"id": 1, "n_ctx": 4096, "speculative": false, "is_processing": false, "id_task": 0, "params": map[string]any{}, "next_token": map[string]any{}},
				})
			}
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/slots/"):
			action := r.URL.Query().Get("action")
			if action == "save" {
				json.NewEncoder(w).Encode(map[string]any{"id_slot": 0, "filename": "ok", "n_saved": 32, "n_written": 1024, "timings": map[string]any{"save_ms": 1.2}})
				return
			}
			if action == "restore" {
				mu.Lock()
				restoreCalls++
				mu.Unlock()
				json.NewEncoder(w).Encode(map[string]any{"id_slot": 0, "filename": "ok", "n_restored": 32, "n_read": 1024, "timings": map[string]any{"restore_ms": 0.8}})
				return
			}
			w.WriteHeader(http.StatusBadRequest)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	manager := newKVCacheManager("model-a", server.URL, 60, debugLogger)
	defer manager.Stop()

	require.NoError(t, manager.save())
	shape = 2
	require.NoError(t, manager.restore())

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 0, restoreCalls)
}
