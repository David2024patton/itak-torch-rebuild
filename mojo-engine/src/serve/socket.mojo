# socket.mojo - the serve layer's transport: POSIX TCP sockets
# in pure Mojo, through std.ffi external_call into libc
# (socket / setsockopt / bind / listen / accept / recv / send /
# close). ROUTE CHOSEN (2026-11-08): this FFI route, not a
# community library. Why: Mojo 1.2's stdlib has no HTTP server
# and no socket type, but its FFI reaches libc directly - the
# probe in the bench note bound, listened, accepted a curl
# request and answered it from a Mojo process, so no external
# dependency (Lightbug or otherwise) was needed, no Python, no
# Go, no second process. The whole transport is this file.
#
# IPv4 only in v1 (config refuses hostnames and non-IPv4 hosts
# by name). One request per connection; SO_REUSEADDR set so a
# restarted bench does not wait out TIME_WAIT. Sends use
# MSG_NOSIGNAL so a client that hangs up mid-answer cannot
# kill the engine with SIGPIPE.
#
# Named refusals: serve_socket_create / serve_socket_bind /
# serve_socket_listen / serve_socket_accept (transport
# failures), serve_body_too_large (Content-Length past the
# configured cap, or a request that outgrows the header
# allowance), serve_http_bad_request via http.mojo parsing.

from std.ffi import external_call
from std.memory import alloc

from serve.http import find_header_end, header_content_length
from tokenizer.byte_level import string_from_bytes_lossy

comptime _AF_INET = 2
comptime _SOCK_STREAM = 1
comptime _SOL_SOCKET = 1
comptime _SO_REUSEADDR = 2
comptime _MSG_NOSIGNAL = 0x4000
comptime _HEADER_ALLOWANCE_BYTES = 16384
comptime _RECV_CHUNK = 8192


struct TcpListener(Movable):
    var fd: Int32

    def __init__(out self, fd: Int32):
        self.fd = fd

    @staticmethod
    def bind_listen(octets: List[Int], port: Int) raises -> TcpListener:
        if len(octets) != 4:
            raise Error("serve_socket_bind: host is not a dotted quad")
        var fd = external_call["socket", Int32](
            Int32(_AF_INET), Int32(_SOCK_STREAM), Int32(0)
        )
        if fd < 0:
            raise Error("serve_socket_create: socket() failed")
        var opt = alloc[Int32](1)
        opt[0] = Int32(1)
        _ = external_call["setsockopt", Int32](
            fd, Int32(_SOL_SOCKET), Int32(_SO_REUSEADDR), opt, Int32(4)
        )
        opt.free()
        # sockaddr_in: family u16, port u16 (network order),
        # addr u32 (network order), 8 zero bytes.
        var addr = alloc[UInt8](16)
        for i in range(16):
            addr[i] = UInt8(0)
        addr[0] = UInt8(_AF_INET)
        addr[1] = UInt8(0)
        addr[2] = UInt8((port >> 8) & 0xFF)
        addr[3] = UInt8(port & 0xFF)
        for i in range(4):
            addr[4 + i] = UInt8(octets[i])
        var br = external_call["bind", Int32](fd, addr, Int32(16))
        addr.free()
        if br != 0:
            _ = external_call["close", Int32](fd)
            raise Error(
                "serve_socket_bind: bind() failed for port " + String(port)
            )
        var lr = external_call["listen", Int32](fd, Int32(16))
        if lr != 0:
            _ = external_call["close", Int32](fd)
            raise Error("serve_socket_listen: listen() failed")
        return TcpListener(fd)

    def accept_conn(self) raises -> Int32:
        # NULL sockaddr / length: the bench does not need the
        # peer address. (external_call passes 0 as the pointer.)
        var cfd = external_call["accept", Int32](self.fd, Int32(0), Int32(0))
        if cfd < 0:
            raise Error("serve_socket_accept: accept() failed")
        return cfd

    def close(mut self):
        if self.fd >= 0:
            _ = external_call["close", Int32](self.fd)
            self.fd = Int32(-1)


def close_conn(fd: Int32):
    _ = external_call["close", Int32](fd)


def send_all(fd: Int32, data: String) raises:
    """Send the whole string, looping over partial sends."""
    var bytes = data.as_bytes()
    var buf = alloc[UInt8](len(bytes) if len(bytes) > 0 else 1)
    for i in range(len(bytes)):
        buf[i] = bytes[i]
    var sent_total = 0
    while sent_total < len(bytes):
        var n = external_call["send", Int](
            fd,
            buf + sent_total,
            Int(len(bytes) - sent_total),
            Int32(_MSG_NOSIGNAL),
        )
        if n <= 0:
            buf.free()
            raise Error("serve_socket_send: send() failed")
        sent_total += n
    buf.free()


def _recv_chunk(fd: Int32, mut out: List[UInt8]) raises -> Int:
    var buf = alloc[UInt8](_RECV_CHUNK)
    var n = external_call["recv", Int](fd, buf, Int(_RECV_CHUNK), Int32(0))
    if n < 0:
        buf.free()
        raise Error("serve_socket_recv: recv() failed")
    for i in range(n):
        out.append(buf[i])
    buf.free()
    return n


def read_request(fd: Int32, max_body_bytes: Int) raises -> String:
    """Read one full HTTP request from a connection.

    Reads until the header block is complete, refuses
    serve_body_too_large when the declared Content-Length is
    past the cap, then reads until the body is complete or
    the peer closes. A request whose headers alone outgrow
    the allowance also refuses serve_body_too_large."""
    var bytes = List[UInt8]()
    var header_end = -1
    var content_length = 0
    while True:
        var n = _recv_chunk(fd, bytes)
        var text = string_from_bytes_lossy(bytes.copy())
        header_end = find_header_end(text)
        if header_end != -1:
            content_length = header_content_length(text)
            if content_length > max_body_bytes:
                raise Error(
                    "serve_body_too_large: Content-Length "
                    + String(content_length)
                    + " exceeds the configured cap "
                    + String(max_body_bytes)
                )
            if len(bytes) >= header_end + content_length:
                return text
            if n == 0:
                return text
        else:
            if len(bytes) > _HEADER_ALLOWANCE_BYTES:
                raise Error(
                    "serve_body_too_large: header block exceeds "
                    + String(_HEADER_ALLOWANCE_BYTES)
                    + " bytes"
                )
            if n == 0:
                return text
