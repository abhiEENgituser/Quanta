package api

import "unicode/utf8"

// utf8Assembler turns a stream of raw token bytes into text safe to emit.
//
// A single token can be a fragment of a multi-byte UTF-8 character (byte-level
// BPE fallback — confirmed empirically with CJK and emoji). Emitting fragments
// would hand the client invalid UTF-8, so bytes accumulate here and only
// complete runes leave. An incomplete tail waits for the token that finishes it.
type utf8Assembler struct {
	buf []byte
}

// Push adds one token's bytes and returns all complete runes now available.
// The returned string may be empty (the token ended mid-character).
func (a *utf8Assembler) Push(piece []byte) string {
	a.buf = append(a.buf, piece...)

	n := 0
	for n < len(a.buf) {
		if !utf8.FullRune(a.buf[n:]) {
			break // incomplete tail — the next token should finish it
		}
		_, size := utf8.DecodeRune(a.buf[n:])
		n += size
	}

	out := string(a.buf[:n])
	// Copy the tail rather than reslicing: Push appends to buf, and an aliased
	// remainder would let a later append clobber bytes backing `out`.
	a.buf = append(a.buf[:0:0], a.buf[n:]...)
	return out
}

// Flush returns whatever remains at end of stream. A tail that never completed
// (generation stopped mid-character) is returned as-is; JSON marshalling
// replaces invalid bytes with U+FFFD, so the frame stays legal UTF-8.
func (a *utf8Assembler) Flush() string {
	out := string(a.buf)
	a.buf = nil
	return out
}
