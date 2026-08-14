#!/usr/bin/env python3
"""Conformance tests for quanta_shim.

Each test uses its own connection, because the shim serves one connection at a
time: it stays in its read loop until the client disconnects, and only then
returns to accept(). Opening a second socket while the first is still open just
parks it in the listen backlog, where nothing reads it.

Run the shim first:
    ./shim/build/quanta_shim models/qwen2.5-0.5b-q4km.gguf
"""

import json
import socket
import struct
import time

SOCK = "/tmp/quanta.sock"


def connect(timeout=5):
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(timeout)          # never let a test hang forever
    s.connect(SOCK)
    return s


def frame(obj):
    body = json.dumps(obj).encode()
    return struct.pack("<I", len(body)) + body


def read_frame(s):
    hdr = b""
    while len(hdr) < 4:
        chunk = s.recv(4 - len(hdr))
        if not chunk:
            raise EOFError("closed while reading header")
        hdr += chunk

    n = struct.unpack("<I", hdr)[0]

    body = b""
    while len(body) < n:
        chunk = s.recv(n - len(body))
        if not chunk:
            raise EOFError("closed while reading body")
        body += chunk

    return body


def rpc(s, obj):
    s.sendall(frame(obj))
    return json.loads(read_frame(s))


def check(name, ok, detail=""):
    print(f"{name:<22}: {'ok' if ok else 'FAIL'}{'  ' + str(detail) if detail else ''}")


# --- framing -----------------------------------------------------------------

# 1. normal round trip
with connect() as s:
    r = rpc(s, {"op": "echo", "v": 1})
    check("1 normal", r.get("ok") is True and r.get("v") == 1, r)

# 2. one frame split across two writes — fails if the server treats a single
#    read() as a whole message
with connect() as s:
    f = frame({"op": "echo", "payload": "x" * 500})
    s.sendall(f[:7])
    time.sleep(0.2)
    s.sendall(f[7:])
    r = json.loads(read_frame(s))
    check("2 split write", r.get("payload") == "x" * 500)

# 3. two frames in one write — fails if the server discards whatever followed
#    the first message in the same read
with connect() as s:
    s.sendall(frame({"op": "echo", "n": "a"}) + frame({"op": "echo", "n": "b"}))
    a = json.loads(read_frame(s))
    b = json.loads(read_frame(s))
    check("3 batched", (a.get("n"), b.get("n")) == ("a", "b"))

# 4. oversized length prefix — must reject and close before allocating.
#    Either outcome means "server closed": b"" is a clean FIN; ECONNRESET is an
#    RST, which is what close() sends when data is still queued unread.
with connect() as s:
    s.sendall(struct.pack("<I", 99_999_999) + b"junk")
    try:
        check("4 oversize", s.recv(4) == b"", "FIN")
    except ConnectionResetError:
        check("4 oversize", True, "RST")
    except socket.timeout:
        check("4 oversize", False, "server did not close")

# 5. zero-length frame — legal edge case
with connect() as s:
    s.sendall(struct.pack("<I", 0))
    r = json.loads(read_frame(s))
    check("5 empty frame", r.get("ok") is False, r.get("error"))

# --- error handling ----------------------------------------------------------

# 6. malformed JSON in a well-framed message. The length prefix already told the
#    server where the message ended, so this is recoverable: it must answer
#    ok:false and KEEP the connection usable.
with connect() as s:
    bad = b"{not json"
    s.sendall(struct.pack("<I", len(bad)) + bad)
    r = json.loads(read_frame(s))
    survived = rpc(s, {"op": "echo", "after": True})
    check("6 bad json recovers", r.get("ok") is False and survived.get("after") is True)

# 7. unknown op
with connect() as s:
    r = rpc(s, {"op": "nope"})
    check("7 unknown op", r.get("ok") is False, r.get("error"))

# --- tokenize ---------------------------------------------------------------

# 8. the probe's prompt — should be 5 tokens, matching "prefill: ... 5 tokens"
with connect() as s:
    r = rpc(s, {"op": "tokenize", "text": "The capital of France is", "add_special": True})
    toks = r.get("tokens", [])
    check("8 tokenize probe", r.get("ok") is True and len(toks) == 5, toks)

# 9. determinism — same input, same output
with connect() as s:
    a = rpc(s, {"op": "tokenize", "text": "hello world"})
    b = rpc(s, {"op": "tokenize", "text": "hello world"})
    check("9 deterministic", a.get("tokens") == b.get("tokens"), a.get("tokens"))

# 10. add_special changes the result (or is at least accepted)
with connect() as s:
    on  = rpc(s, {"op": "tokenize", "text": "hi", "add_special": True})
    off = rpc(s, {"op": "tokenize", "text": "hi", "add_special": False})
    check("10 add_special", on.get("ok") and off.get("ok"),
          f"on={on.get('tokens')} off={off.get('tokens')}")

# 11. empty text
with connect() as s:
    r = rpc(s, {"op": "tokenize", "text": ""})
    check("11 empty text", r.get("ok") is True, r.get("tokens"))

# 12. wrong type for 'text'
with connect() as s:
    r = rpc(s, {"op": "tokenize", "text": 42})
    check("12 text not string", r.get("ok") is False, r.get("error"))

# 13. multibyte input — the tokenizer must not choke, and the reply must be
#     valid JSON (token ids are ints, so no UTF-8 hazard here; that arrives with
#     piece_b64 in Stage 3)
with connect() as s:
    r = rpc(s, {"op": "tokenize", "text": "café — 東京 🙂"})
    check("13 multibyte", r.get("ok") is True, r.get("tokens"))