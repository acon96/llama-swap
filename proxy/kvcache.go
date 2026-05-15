package proxy

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

var kvCacheDir = "/dev/shm/llama-swap"

const (
	kvCacheFilePrefix      = "stash-"
	kvCacheManifestVersion = 1
)

// kvCacheManager handles saving, restoring, and cleaning up KV cache slot files
// for a llama-server process via the slot save/restore API endpoints.
type kvCacheManager struct {
	modelID    string
	proxyURL   string
	ttl        int // model TTL in seconds; cleanup deletes files older than 3x this
	logger     *LogMonitor
	httpClient *http.Client

	// periodic cleanup
	mu       sync.Mutex
	stopChan chan struct{}
	doneChan chan struct{}
}

// slotParams represents generation parameters for a slot
type slotParams struct {
	Seed        int      `json:"seed"`
	Temperature float64  `json:"temperature"`
	TopK        int      `json:"top_k"`
	TopP        float64  `json:"top_p"`
	MinP        float64  `json:"min_p"`
	TypicalP    float64  `json:"typical_p"`
	MaxTokens   int      `json:"max_tokens"`
	NPredict    int      `json:"n_predict"`
	ChatFormat  string   `json:"chat_format"`
	Samplers    []string `json:"samplers"`
}

// slotNextToken represents token prediction info
type slotNextToken struct {
	HasNextToken bool `json:"has_next_token"`
	HasNewLine   bool `json:"has_new_line"`
	NRemain      int  `json:"n_remain"`
	NDecoded     int  `json:"n_decoded"`
}

// slotInfo represents a slot returned by the /slots endpoint
type slotInfo struct {
	ID           int           `json:"id"`
	NCtx         int           `json:"n_ctx"`
	Speculative  bool          `json:"speculative"`
	IsProcessing bool          `json:"is_processing"`
	IDTask       int           `json:"id_task"`
	Params       slotParams    `json:"params"`
	NextToken    slotNextToken `json:"next_token"`
}

// slotsResponse represents the response from the /slots endpoint
type slotsResponse []slotInfo

// saveRequest is the JSON body for saving a slot
type saveRequest struct {
	Filename string `json:"filename"`
}

// saveResponse is the response from saving a slot
type saveResponse struct {
	IDSlot   int    `json:"id_slot"`
	Filename string `json:"filename"`
	NSaved   int    `json:"n_saved"`
	NWritten int    `json:"n_written"`
	Timings  struct {
		SaveMS float64 `json:"save_ms"`
	} `json:"timings"`
}

// restoreRequest is the JSON body for restoring a slot
type restoreRequest struct {
	Filename string `json:"filename"`
}

// restoreResponse is the response from restoring a slot
type restoreResponse struct {
	IDSlot    int    `json:"id_slot"`
	Filename  string `json:"filename"`
	NRestored int    `json:"n_restored"`
	NRead     int    `json:"n_read"`
	Timings   struct {
		RestoreMS float64 `json:"restore_ms"`
	} `json:"timings"`
}

type kvCacheManifest struct {
	Version     int                   `json:"version"`
	ModelID     string                `json:"model_id"`
	Fingerprint string                `json:"fingerprint"`
	SavedAt     time.Time             `json:"saved_at"`
	Slots       []kvCacheManifestSlot `json:"slots"`
}

type kvCacheManifestSlot struct {
	SlotID      int    `json:"slot_id"`
	Filename    string `json:"filename"`
	NCtx        int    `json:"n_ctx"`
	Speculative bool   `json:"speculative"`
}

func newKVCacheManager(modelID, proxyURL string, ttl int, logger *LogMonitor) *kvCacheManager {
	m := &kvCacheManager{
		modelID:  modelID,
		proxyURL: proxyURL,
		ttl:      ttl,
		logger:   logger,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		stopChan: make(chan struct{}),
		doneChan: make(chan struct{}),
	}

	// Start periodic cleanup goroutine if TTL is set
	if ttl > 0 {
		go m.periodicCleanup()
	} else {
		close(m.doneChan)
	}

	return m
}

func (m *kvCacheManager) newSlotActionRequest(slotID int, action string, body interface{}) (*http.Request, error) {
	url := fmt.Sprintf("%s/slots/%d?action=%s", m.proxyURL, slotID, action)

	requestBody := []byte("{}")
	if body != nil {
		marshalledBody, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal %s request body for slot %d: %w", action, slotID, err)
		}
		requestBody = marshalledBody
	}

	req, err := http.NewRequest("POST", url, bytes.NewReader(requestBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	return req, nil
}

// periodicCleanup runs a background goroutine that periodically checks for
// and removes stale KV cache files (older than 3x the model TTL).
func (m *kvCacheManager) periodicCleanup() {
	defer close(m.doneChan)

	// Run cleanup every 30 seconds
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.stopChan:
			return
		case <-ticker.C:
			m.cleanup()
		}
	}
}

// Stop signals the periodic cleanup goroutine to stop and waits for it to finish.
func (m *kvCacheManager) Stop() {
	m.mu.Lock()
	defer m.mu.Unlock()

	select {
	case <-m.stopChan:
		return // already stopped
	default:
	}

	close(m.stopChan)
	<-m.doneChan
}

// listSlots queries the llama-server for current slot state.
func (m *kvCacheManager) listSlots() ([]slotInfo, error) {
	url := m.proxyURL + "/slots"
	resp, err := m.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to list slots: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("list slots returned status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var slots slotsResponse
	if err := json.NewDecoder(resp.Body).Decode(&slots); err != nil {
		return nil, fmt.Errorf("failed to decode slots response: %w", err)
	}

	return slots, nil
}

// saveSlot saves a single slot's KV cache to a file
func (m *kvCacheManager) saveSlot(slotID int, filename string) error {
	req, err := m.newSlotActionRequest(slotID, "save", saveRequest{Filename: filename})
	if err != nil {
		return fmt.Errorf("failed to create save request for slot %d: %w", slotID, err)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to save slot %d: %w", slotID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("save slot %d returned status %d: %s", slotID, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var result saveResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode save response for slot %d: %w", slotID, err)
	}

	m.logger.Debugf("<%s> kvcache: saved slot %d -> %s (%d tokens, %.1fms)",
		m.modelID, slotID, filename, result.NSaved, result.Timings.SaveMS)
	return nil
}

// restoreSlot restores a single slot's KV cache from a file
func (m *kvCacheManager) restoreSlot(slotID int, filename string) error {
	req, err := m.newSlotActionRequest(slotID, "restore", restoreRequest{Filename: filename})
	if err != nil {
		return fmt.Errorf("failed to create restore request for slot %d: %w", slotID, err)
	}

	resp, err := m.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to restore slot %d: %w", slotID, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("restore slot %d returned status %d: %s", slotID, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	var result restoreResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return fmt.Errorf("failed to decode restore response for slot %d: %w", slotID, err)
	}

	m.logger.Debugf("<%s> kvcache: restored slot %d from %s (%d tokens, %.1fms)",
		m.modelID, slotID, filename, result.NRestored, result.Timings.RestoreMS)
	return nil
}

func (m *kvCacheManager) fingerprint(slots []slotInfo) string {
	hash := sha256.New()
	_, _ = io.WriteString(hash, m.modelID)
	for _, slot := range slots {
		_, _ = io.WriteString(hash, fmt.Sprintf("|%d:%d:%t", slot.ID, slot.NCtx, slot.Speculative))
	}
	return hex.EncodeToString(hash.Sum(nil))
}

func (m *kvCacheManager) manifestFilename(fingerprint string) string {
	return fmt.Sprintf("%s%s.json", kvCacheFilePrefix, fingerprint)
}

func (m *kvCacheManager) slotFilename(fingerprint string, slotID int) string {
	return fmt.Sprintf("%s%s-slot-%d.kv", kvCacheFilePrefix, fingerprint, slotID)
}

func (m *kvCacheManager) manifestPath(fingerprint string) string {
	return filepath.Join(kvCacheDir, m.manifestFilename(fingerprint))
}

func (m *kvCacheManager) writeManifest(manifest kvCacheManifest) error {
	manifestPath := m.manifestPath(manifest.Fingerprint)
	tempPath := manifestPath + ".tmp"

	data, err := json.Marshal(manifest)
	if err != nil {
		return fmt.Errorf("failed to marshal manifest: %w", err)
	}

	if err := os.WriteFile(tempPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write manifest temp file: %w", err)
	}

	if err := os.Rename(tempPath, manifestPath); err != nil {
		_ = os.Remove(tempPath)
		return fmt.Errorf("failed to rename manifest temp file: %w", err)
	}

	return nil
}

func (m *kvCacheManager) readManifest(fingerprint string) (*kvCacheManifest, error) {
	manifestPath := m.manifestPath(fingerprint)
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to read manifest: %w", err)
	}

	var manifest kvCacheManifest
	if err := json.Unmarshal(data, &manifest); err != nil {
		return nil, fmt.Errorf("failed to decode manifest: %w", err)
	}

	if manifest.Version != kvCacheManifestVersion || manifest.ModelID != m.modelID || manifest.Fingerprint != fingerprint {
		return nil, nil
	}

	return &manifest, nil
}

// save saves all available slots to the cache file
func (m *kvCacheManager) save() error {
	slots, err := m.listSlots()
	if err != nil {
		return fmt.Errorf("list slots before save: %w", err)
	}

	if len(slots) == 0 {
		m.logger.Debugf("<%s> kvcache: no slots available to save", m.modelID)
		return nil
	}

	// Ensure the save directory exists
	if err := os.MkdirAll(kvCacheDir, 0755); err != nil {
		return fmt.Errorf("failed to create cache directory %s: %w", kvCacheDir, err)
	}

	fingerprint := m.fingerprint(slots)
	manifest := kvCacheManifest{
		Version:     kvCacheManifestVersion,
		ModelID:     m.modelID,
		Fingerprint: fingerprint,
		SavedAt:     time.Now().UTC(),
		Slots:       make([]kvCacheManifestSlot, 0, len(slots)),
	}

	for _, slot := range slots {
		filename := m.slotFilename(fingerprint, slot.ID)
		if err := m.saveSlot(slot.ID, filename); err != nil {
			m.logger.Warnf("<%s> kvcache: failed to save slot %d: %v", m.modelID, slot.ID, err)
			continue
		}

		manifest.Slots = append(manifest.Slots, kvCacheManifestSlot{
			SlotID:      slot.ID,
			Filename:    filename,
			NCtx:        slot.NCtx,
			Speculative: slot.Speculative,
		})
	}

	if len(manifest.Slots) == 0 {
		m.logger.Debugf("<%s> kvcache: no slots were saved successfully", m.modelID)
		return nil
	}

	if err := m.writeManifest(manifest); err != nil {
		return fmt.Errorf("failed to write stash manifest: %w", err)
	}

	// Run cleanup after saving
	m.cleanup()

	return nil
}

// restore attempts to restore the cache file to available slots
func (m *kvCacheManager) restore() error {
	slots, err := m.listSlots()
	if err != nil {
		return fmt.Errorf("list slots before restore: %w", err)
	}

	if len(slots) == 0 {
		return nil
	}

	fingerprint := m.fingerprint(slots)
	manifest, err := m.readManifest(fingerprint)
	if err != nil {
		return fmt.Errorf("read manifest before restore: %w", err)
	}
	if manifest == nil {
		return nil
	}

	slotsByID := make(map[int]slotInfo, len(slots))
	for _, slot := range slots {
		slotsByID[slot.ID] = slot
	}

	for _, savedSlot := range manifest.Slots {
		currentSlot, ok := slotsByID[savedSlot.SlotID]
		if !ok {
			m.logger.Warnf("<%s> kvcache: skipping restore for missing slot %d", m.modelID, savedSlot.SlotID)
			continue
		}

		if currentSlot.NCtx != savedSlot.NCtx || currentSlot.Speculative != savedSlot.Speculative {
			m.logger.Warnf("<%s> kvcache: skipping restore for slot %d due to incompatible slot shape", m.modelID, savedSlot.SlotID)
			continue
		}

		if err := m.restoreSlot(savedSlot.SlotID, savedSlot.Filename); err != nil {
			m.logger.Warnf("<%s> kvcache: failed to restore to slot %d: %v", m.modelID, savedSlot.SlotID, err)
		}
	}

	return nil
}

// cleanup removes cache files older than 3x the model TTL
func (m *kvCacheManager) cleanup() {
	if m.ttl <= 0 {
		return
	}

	cutoff := time.Duration(m.ttl*3) * time.Second

	entries, err := os.ReadDir(kvCacheDir)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		m.logger.Warnf("<%s> kvcache: failed to read cache directory for cleanup: %v", m.modelID, err)
		return
	}

	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasPrefix(name, kvCacheFilePrefix) {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		if time.Since(info.ModTime()) > cutoff {
			path := filepath.Join(kvCacheDir, name)
			if err := os.Remove(path); err != nil {
				m.logger.Warnf("<%s> kvcache: failed to remove old cache file %s: %v", m.modelID, name, err)
			} else {
				m.logger.Debugf("<%s> kvcache: removed old cache file %s (age: %v, cutoff: %v)",
					m.modelID, name, time.Since(info.ModTime()), cutoff)
			}
		}
	}
}

// eraseAll erases all slot caches (used during shutdown to clean up in-memory state)
func (m *kvCacheManager) eraseAll() error {
	slots, err := m.listSlots()
	if err != nil {
		return fmt.Errorf("list slots before erase: %w", err)
	}

	for _, slotID := range slots {
		req, err := m.newSlotActionRequest(slotID.ID, "erase", nil)
		if err != nil {
			m.logger.Warnf("<%s> kvcache: failed to create erase request for slot %d: %v", m.modelID, slotID.ID, err)
			continue
		}

		resp, err := m.httpClient.Do(req)
		if err != nil {
			m.logger.Warnf("<%s> kvcache: failed to erase slot %d: %v", m.modelID, slotID.ID, err)
			continue
		}
		resp.Body.Close()

		m.logger.Debugf("<%s> kvcache: erased slot %d", m.modelID, slotID.ID)
	}

	return nil
}
