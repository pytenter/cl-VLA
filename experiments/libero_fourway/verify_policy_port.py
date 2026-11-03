"""Linux regression check for consecutive LIBERO policy server launches."""
from __future__ import annotations

import errno
import json
import socket

from run_experiment import check_policy_port


def main():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            check_policy_port(port)
        except RuntimeError:
            active_listener_rejected = True
        else:
            raise AssertionError("An active listener must remain protected")
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b""
            # Server closes first; its endpoint enters TIME_WAIT when the
            # client closes. This reproduces a just-terminated model server.
    with socket.socket() as original_probe:
        try:
            original_probe.bind(("127.0.0.1", port))
        except OSError as error:
            assert error.errno == errno.EADDRINUSE, error
            original_probe_reproduced_failure = True
        else:
            raise AssertionError("Expected original probe to reject TIME_WAIT")
    check_policy_port(port)
    print(json.dumps({"status": "passed", "active_listener_rejected": active_listener_rejected,
                      "original_probe_reproduced_failure": original_probe_reproduced_failure,
                      "time_wait_accepted": True}))


if __name__ == "__main__":
    main()
