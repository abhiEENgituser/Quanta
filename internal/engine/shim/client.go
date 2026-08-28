// Package shim implements engine.Backend as a client for the C++ shim's unix
// socket protocol: [4-byte little-endian length][JSON body], strictly
// synchronous — one request, one response. See docs/protocol.md.
package shim

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"syscall"

	"github.com/abhiEENgituser/Quanta/internal/engine"
)

// MaxFrame mirrors the shim's limit. A length prefix beyond this means the
// stream is not trustworthy — same rule on both sides of the wire.
const MaxFrame = 8 << 20

// ErrConnClosed reports that the shim ended the connection. It deliberately
// covers both a clean EOF and ECONNRESET: which one arrives depends on whether
// the shim had unread bytes queued when it closed (FIN vs RST) — timing luck,
// not a meaningful distinction, so callers get one error to handle.
var ErrConnClosed = errors.New("shim: connection closed")

// OpError is a request the shim understood but refused (ok:false). The
// connection remains usable after one of these.
type OpError struct {
	Op       string
	Msg      string
	DecodeRC int // llama_decode's raw return when relevant: 1 = no KV slot
	// (recoverable by freeing memory), negative = fatal. 0 when not set.
}

func (e *OpError) Error() string {
	if e.DecodeRC != 0 {
		return fmt.Sprintf("shim: %s: %s (decode_rc=%d)", e.Op, e.Msg, e.DecodeRC)
	}
	return fmt.Sprintf("shim: %s: %s", e.Op, e.Msg)
}

// Client speaks the protocol over one connection. Safe for concurrent use: the
// protocol is strictly synchronous, so a mutex serialises callers — matching
// the shim, which serves exactly one request at a time anyway.
type Client struct {
	mu   sync.Mutex
	conn net.Conn
}

// interface conformance is checked at compile time, not discovered at runtime.
var _ engine.Backend = (*Client)(nil)

func Dial(socketPath string) (*Client, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return nil, fmt.Errorf("shim: dial %s: %w", socketPath, err)
	}
	return &Client{conn: conn}, nil
}

func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.conn.Close()
}

// classify wraps transport errors so callers can errors.Is(err, ErrConnClosed)
// without caring whether the close surfaced as EOF, unexpected EOF, or RST.
func classify(err error) error {
	if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) ||
		errors.Is(err, syscall.ECONNRESET) || errors.Is(err, net.ErrClosed) ||
		errors.Is(err, syscall.EPIPE) {
		return fmt.Errorf("%w: %v", ErrConnClosed, err)
	}
	return err
}

// call sends one request and decodes one response into resp.
func (c *Client) call(op string, req any, resp any) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("shim: %s: marshal: %w", op, err)
	}
	if len(body) > MaxFrame {
		return fmt.Errorf("shim: %s: request exceeds max frame (%d bytes)", op, len(body))
	}

	// One Write for prefix+body. Unlike the raw write(2) syscall, Go's
	// net.Conn.Write only returns short on error, so no manual loop is needed
	// on the write side. Reads are the opposite — see io.ReadFull below.
	frame := make([]byte, 4+len(body))
	binary.LittleEndian.PutUint32(frame[:4], uint32(len(body)))
	copy(frame[4:], body)
	if _, err := c.conn.Write(frame); err != nil {
		return classify(err)
	}

	var hdr [4]byte
	if _, err := io.ReadFull(c.conn, hdr[:]); err != nil {
		return classify(err)
	}
	n := binary.LittleEndian.Uint32(hdr[:])
	if n > MaxFrame {
		// The length itself is untrustworthy, so the frame boundary is lost.
		// Same rule as the shim: close, do not attempt to resynchronise.
		c.conn.Close()
		return fmt.Errorf("shim: %s: response frame too large (%d bytes): %w", op, n, ErrConnClosed)
	}

	buf := make([]byte, n)
	if _, err := io.ReadFull(c.conn, buf); err != nil {
		return classify(err)
	}

	// Every response carries ok/error; decode those first, then the payload.
	var env struct {
		Ok       bool   `json:"ok"`
		Error    string `json:"error"`
		DecodeRC int    `json:"decode_rc"`
	}
	if err := json.Unmarshal(buf, &env); err != nil {
		return fmt.Errorf("shim: %s: bad response json: %w", op, err)
	}
	if !env.Ok {
		return &OpError{Op: op, Msg: env.Error, DecodeRC: env.DecodeRC}
	}
	if resp != nil {
		if err := json.Unmarshal(buf, resp); err != nil {
			return fmt.Errorf("shim: %s: bad payload json: %w", op, err)
		}
	}
	return nil
}

// --- engine.Backend ----------------------------------------------------------

func (c *Client) Tokenize(text string, addSpecial bool) ([]int32, error) {
	req := struct {
		Op         string `json:"op"`
		Text       string `json:"text"`
		AddSpecial bool   `json:"add_special"`
	}{"tokenize", text, addSpecial}

	var resp struct {
		Tokens []int32 `json:"tokens"`
	}
	if err := c.call("tokenize", req, &resp); err != nil {
		return nil, err
	}
	return resp.Tokens, nil
}

func (c *Client) Prefill(seq engine.SeqID, tokens []int32, startPos int32) error {
	req := struct {
		Op       string  `json:"op"`
		Seq      int32   `json:"seq"`
		Tokens   []int32 `json:"tokens"`
		StartPos int32   `json:"start_pos"`
	}{"prefill", int32(seq), tokens, startPos}

	return c.call("prefill", req, nil)
}

func (c *Client) Step(active []engine.SeqID) (engine.StepResult, error) {
	ids := make([]int32, len(active))
	for i, s := range active {
		ids[i] = int32(s)
	}
	req := struct {
		Op     string  `json:"op"`
		Active []int32 `json:"active"`
	}{"step", ids}

	var resp struct {
		Tokens []struct {
			Seq      int32  `json:"seq"`
			ID       int32  `json:"id"`
			PieceB64 string `json:"piece_b64"`
			Finished bool   `json:"finished"`
		} `json:"tokens"`
	}
	if err := c.call("step", req, &resp); err != nil {
		return engine.StepResult{}, err
	}

	out := engine.StepResult{Tokens: make([]engine.Token, 0, len(resp.Tokens))}
	for _, t := range resp.Tokens {
		piece, err := base64.StdEncoding.DecodeString(t.PieceB64)
		if err != nil {
			return engine.StepResult{}, fmt.Errorf("shim: step: bad piece_b64 for seq %d: %w", t.Seq, err)
		}
		out.Tokens = append(out.Tokens, engine.Token{
			Seq:      engine.SeqID(t.Seq),
			ID:       t.ID,
			Piece:    piece,
			Finished: t.Finished,
		})
	}
	return out, nil
}

func (c *Client) Evict(seq engine.SeqID, p0, p1 int32) (bool, error) {
	req := struct {
		Op  string `json:"op"`
		Seq int32  `json:"seq"`
		P0  int32  `json:"p0"`
		P1  int32  `json:"p1"`
	}{"evict", int32(seq), p0, p1}

	var resp struct {
		Removed bool `json:"removed"`
	}
	if err := c.call("evict", req, &resp); err != nil {
		return false, err
	}
	return resp.Removed, nil
}

func (c *Client) PosRange(seq engine.SeqID) (min, max int32, err error) {
	req := struct {
		Op  string `json:"op"`
		Seq int32  `json:"seq"`
	}{"pos_range", int32(seq)}

	var resp struct {
		Min int32 `json:"min"`
		Max int32 `json:"max"`
	}
	if err := c.call("pos_range", req, &resp); err != nil {
		return 0, 0, err
	}
	return resp.Min, resp.Max, nil
}
