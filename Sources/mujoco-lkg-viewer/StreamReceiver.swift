import Foundation

// 训练直播接收端（microduck_rl lkg_stream.py 的 UDP qpos 流）。
// 包格式（小端）：u32 magic('LKGQ'=0x4C4B4751) + u32 nq + f32 sim_t + f32 qpos[nq]。
// 只保留最新一包；包计数供遥测（pkt/s）。

final class UdpQposReceiver {
    static let magic: UInt32 = 0x4C4B4751

    private var fd: Int32 = -1
    private var source: DispatchSourceRead?
    private let lock = NSLock()
    private var latest: (simT: Float, qpos: [Float])?
    private var packets: UInt64 = 0

    deinit { source?.cancel(); if fd >= 0 { close(fd) } }

    func start(port: UInt16) -> String? {
        fd = socket(AF_INET, SOCK_DGRAM, 0)
        guard fd >= 0 else { return "socket() failed" }
        var yes: Int32 = 1
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, socklen_t(MemoryLayout<Int32>.size))
        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = port.bigEndian
        addr.sin_addr.s_addr = INADDR_ANY
        let bound = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(fd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bound == 0 else { close(fd); fd = -1; return "bind :\(port) failed (errno \(errno))" }
        let src = DispatchSource.makeReadSource(fileDescriptor: fd, queue: .global(qos: .userInteractive))
        src.setEventHandler { [weak self] in self?.drain() }
        src.resume()
        source = src
        return nil
    }

    private func drain() {
        var buf = [UInt8](repeating: 0, count: 2048)
        while true {
            let n = recv(fd, &buf, buf.count, MSG_DONTWAIT)
            if n <= 0 { break }
            guard n >= 12 else { continue }
            let (magic, nq, simT) = buf.withUnsafeBytes {
                ($0.load(as: UInt32.self),
                 Int($0.load(fromByteOffset: 4, as: UInt32.self)),
                 $0.load(fromByteOffset: 8, as: Float.self))
            }
            guard magic == Self.magic, nq > 0, nq <= 256, n == 12 + nq * 4 else { continue }
            var q = [Float](repeating: 0, count: nq)
            q.withUnsafeMutableBytes { dst in
                buf.withUnsafeBytes { src in
                    memcpy(dst.baseAddress!, src.baseAddress! + 12, nq * 4)
                }
            }
            lock.lock()
            latest = (simT, q)
            packets &+= 1
            lock.unlock()
        }
    }

    func takeLatest() -> (simT: Float, qpos: [Float])? {
        lock.lock(); defer { lock.unlock() }
        return latest
    }

    var packetCount: UInt64 {
        lock.lock(); defer { lock.unlock() }
        return packets
    }
}
