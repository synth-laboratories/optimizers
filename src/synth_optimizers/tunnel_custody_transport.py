"""Bounded authority I/O for private custody IPC, with no redirect or retry."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .tunnels import TunnelError

MAX_AUTHORITY_BYTES = 262144


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


class CustodyApiClient:
    def __init__(self, backend_url: str, api_key: str, timeout_seconds: float):
        url = urlsplit(backend_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or url.username
                or url.password or url.query or url.fragment or url.path not in {"", "/"}):
            raise TunnelError("custody backend must be an HTTP origin")
        if not isinstance(api_key, str) or not 1 <= len(api_key) <= 4096 or any(
            not 33 <= ord(character) <= 126 for character in api_key
        ):
            raise TunnelError("invalid custody API credential")
        self.backend_url = backend_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.opener = build_opener(NoRedirects())

    def _json_request(self, method, path, payload=None):
        if not path.startswith("/api/v1/") or path.startswith("//"):
            raise TunnelError("invalid custody API path")
        headers = {"Authorization": "Bearer " + self.api_key, "Accept": "application/json"}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode()
            if len(body) > MAX_AUTHORITY_BYTES:
                raise TunnelError("custody authority request byte capacity exceeded")
            headers["Content-Type"] = "application/json"
        request = Request(self.backend_url + path, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout_seconds) as response:
                encoded = response.read(MAX_AUTHORITY_BYTES + 1)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError):
                error.close()
            raise TunnelError("custody authority request unresolved") from error
        if len(encoded) > MAX_AUTHORITY_BYTES:
            raise TunnelError("custody authority response byte capacity exceeded")
        result = json.loads(encoded)
        if not isinstance(result, dict):
            raise TunnelError("custody authority response must be an object")
        return result
