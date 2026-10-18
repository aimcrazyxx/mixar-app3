# SPDX-FileCopyrightText: 2026 aimcrazyxx
# SPDX-License-Identifier: GPL-3.0-or-later

"""Point Mixar's agents at any OpenAI-compatible API through the Local provider.

Mixar's agents run on Mixar's server. For cloud providers (OpenAI, Anthropic,
...) that server ignores a custom Base URL and checks the key against the
provider's own API, so a key from another endpoint is rejected. The Local
provider works differently: the server sends every model call back through
the app, which makes it - but only to a server on this computer or the local
network. This script is that server: it listens on 127.0.0.1 and forwards to
the API you give it.

    python mixar_forward.py https://api.example.com/v1 [--port 4000]

It asks for the API key without echoing it (Enter alone forwards the key
typed in Mixar's Local form instead) and keeps it in memory only. Then, in
Mixar's AI Provider dialog: Local, custom server, Base URL
http://127.0.0.1:4000/v1, model = the API's model ID.

Standard library only, so the Python inside Mixar runs it:
<Mixar folder>\\5.2\\python\\bin\\python.exe mixar_forward.py ...
"""

import argparse
import getpass
import http.server
import json
import sys
import urllib.error
import urllib.request

# Longer than the 270 s Mixar's relay waits for an answer.
UPSTREAM_TIMEOUT_S = 300
# Mixar's relay sends only these; anything else never reaches us.
FORWARDED_HEADERS = ("Content-Type", "Accept")


def make_handler(upstream, key):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self._forward("GET")

        def do_POST(self):
            self._forward("POST")

        def _forward(self, method):
            # Mixar calls <base>/chat/completions and, when saving, pings
            # <base>/models; with base http://127.0.0.1:<port>/v1 both start
            # with /v1/, which maps onto the upstream base.
            if not self.path.startswith("/v1/"):
                self._reply(404, b'{"error": {"message": "not found"}}')
                return
            body = None
            if method == "POST":
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            request = urllib.request.Request(upstream + self.path[len("/v1"):], data=body, method=method)
            authorization = f"Bearer {key}" if key else self.headers.get("Authorization")
            if authorization:
                request.add_header("Authorization", authorization)
            for name in FORWARDED_HEADERS:
                if self.headers.get(name):
                    request.add_header(name, self.headers[name])
            try:
                with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_S) as response:
                    self._reply(response.status, response.read(), response.headers.get("Content-Type"))
            except urllib.error.HTTPError as error:
                # Provider errors (401, 429, ...) go back as they are: the
                # agent shows them like any other provider error.
                self._reply(error.code, error.read(), error.headers.get("Content-Type"))
            except Exception as error:  # noqa: BLE001 - unreachable, TLS, timeout
                message = json.dumps({"error": {"message": f"upstream unreachable: {error}"}})
                self._reply(502, message.encode())

        def _reply(self, status, data, content_type=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type or "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_request(self, code="-", size="-"):
            # Method, path and status only: never bodies or keys.
            sys.stderr.write(f"{self.command} {self.path} -> {code}\n")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Forward Mixar's Local AI provider to an OpenAI-compatible API.")
    parser.add_argument("upstream", help="API base URL, e.g. https://api.example.com/v1")
    parser.add_argument("--port", type=int, default=4000, help="local port (default 4000)")
    args = parser.parse_args()
    upstream = args.upstream.rstrip("/")
    key = getpass.getpass("API key (Enter = use the key typed in Mixar): ").strip()

    # 127.0.0.1 only: nothing else on the network can use the key.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(upstream, key))
    print(f"Forwarding to {upstream}")
    print(f"In Mixar: AI Provider > Local > Base URL http://127.0.0.1:{args.port}/v1  (Ctrl+C stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
