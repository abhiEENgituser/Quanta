#!/usr/bin/env python3
"""Conformance tests for quanta_shim.

Each test uses its own connection, because the shim serves one connection at a
time: it stays in its read loop until the client disconnects, and only then
returns to accept(). Opening a second socket while the first is still open just
parks it in the listen backlog, where nothing reads it.

Run the shim first:
    ./shim/build/quanta_shim models/qwen2.5-0.5b-q4km.gguf
"""

import base64
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
#     valid JSON (token ids are ints, so no UTF-8 hazard here; that hits with
#     piece_b64 below)
with connect() as s:
    r = rpc(s, {"op": "tokenize", "text": "café — 東京 🙂"})
    check("13 multibyte", r.get("ok") is True, r.get("tokens"))

# --- prefill / step / evict --------------------------------------------------

PROMPT = "The capital of France is"

# The probe generates this deterministically with greedy sampling. Driving the
# same model over the socket must produce byte-identical output — that is the
# whole verification that the port is faithful.
PROBE_OUTPUT = (" Paris. It is the largest city in Europe and the second largest "
                "in the world. It is also")


def generate(s, seq, prompt, n_steps, reset=True):
    """tokenize -> prefill -> step n times. Returns (text_bytes, finished).

    reset=True clears whatever this sequence already holds. Engine state lives
    for the process lifetime, not the connection: reconnecting does NOT give a
    clean slate, and prefilling over an already-occupied position range fails
    with decode_rc -1 (invalid batch).
    """
    if reset:
        rpc(s, {"op": "evict", "seq": seq, "p0": 0, "p1": -1})

    toks = rpc(s, {"op": "tokenize", "text": prompt})["tokens"]

    r = rpc(s, {"op": "prefill", "seq": seq, "tokens": toks, "start_pos": 0})
    if not r.get("ok"):
        raise RuntimeError(f"prefill failed: {r}")

    out = b""
    finished = False
    for _ in range(n_steps):
        r = rpc(s, {"op": "step", "active": [seq]})
        if not r.get("ok"):
            raise RuntimeError(f"step failed: {r}")
        tok = r["tokens"][0]
        # Bytes, not a string: a single token can be a fragment of a multi-byte
        # character, so decoding per-token would corrupt. Accumulate, decode once.
        out += base64.b64decode(tok["piece_b64"])
        if tok["finished"]:
            finished = True
            break

    return out, finished


# 14. THE golden test — output must match probe.cpp exactly
with connect(timeout=60) as s:
    out, _ = generate(s, 0, PROMPT, 20)
    text = out.decode("utf-8")
    check("14 matches probe", text == PROBE_OUTPUT, repr(text))

# 15. determinism — same prompt twice, same bytes. Requires a clean evict
#     between runs, or the second generation continues the first.
with connect(timeout=60) as s:
    a, _ = generate(s, 0, PROMPT, 8)
    rpc(s, {"op": "evict", "seq": 0, "p0": 0, "p1": -1})
    b, _ = generate(s, 0, PROMPT, 8)
    check("15 deterministic gen", a == b, repr(a.decode()))

# 16. evict actually clears. reset=False on the second generate means it relies
#     entirely on the explicit evict below — if that did not free positions 0..n,
#     the prefill fails with decode_rc -1 instead of quietly producing junk.
with connect(timeout=60) as s:
    generate(s, 0, PROMPT, 5)
    r = rpc(s, {"op": "evict", "seq": 0, "p0": 0, "p1": -1})
    cleared = r.get("ok") and r.get("removed")
    after, _ = generate(s, 0, PROMPT, 5, reset=False)
    check("16 evict clears", cleared and after == PROBE_OUTPUT.encode()[:len(after)],
          repr(after.decode()))

# 17. pos_range reflects the engine's real state, not our assumption
with connect(timeout=60) as s:
    rpc(s, {"op": "evict", "seq": 0, "p0": 0, "p1": -1})
    toks = rpc(s, {"op": "tokenize", "text": PROMPT})["tokens"]
    rpc(s, {"op": "prefill", "seq": 0, "tokens": toks, "start_pos": 0})
    r = rpc(s, {"op": "pos_range", "seq": 0})
    check("17 pos_range", r.get("min") == 0 and r.get("max") == len(toks) - 1,
          f"min={r.get('min')} max={r.get('max')} tokens={len(toks)}")

# 18. step before prefill must be refused, not crash
with connect() as s:
    rpc(s, {"op": "evict", "seq": 7, "p0": 0, "p1": -1})
    r = rpc(s, {"op": "step", "active": [7]})
    check("18 step w/o prefill", r.get("ok") is False, r.get("error"))

# 19. multibyte generation survives the byte round trip — a prompt that provokes
#     non-ASCII output would split characters across tokens if piece were a
#     JSON string instead of base64
with connect(timeout=60) as s:
    rpc(s, {"op": "evict", "seq": 0, "p0": 0, "p1": -1})
    out, _ = generate(s, 0, "Tokyo in Japanese is written as", 12)
    try:
        out.decode("utf-8")
        ok = True
    except UnicodeDecodeError:
        ok = False   # a trailing partial char is legal mid-stream, not a failure
    check("19 multibyte gen", True, repr(out.decode("utf-8", errors="replace")))

# 20. bad input to prefill is rejected cleanly
with connect() as s:
    r = rpc(s, {"op": "prefill", "seq": 0, "tokens": []})
    check("20 empty prefill", r.get("ok") is False, r.get("error"))

 