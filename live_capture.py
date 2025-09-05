import threading
import time
import json
import random
from collections import defaultdict, deque
from queue import Queue, Empty

try:
    from scapy.all import sniff, TCP, UDP, IP
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

# Map common ports to NSL-KDD service names
PORT_TO_SERVICE = {
    80: 'http', 443: 'http_443', 21: 'ftp', 20: 'ftp_data', 25: 'smtp', 110: 'pop_3',
    53: 'domain_u', 23: 'telnet', 22: 'ssh',  finger_port := 79: 'finger',  
}
# Fallback service if unknown
DEFAULT_SERVICE = 'other'


def _service_from_ports(sport, dport):
    return PORT_TO_SERVICE.get(sport) or PORT_TO_SERVICE.get(dport) or DEFAULT_SERVICE


class LiveCapture:
    def __init__(self):
        self.events = Queue(maxsize=1000)
        self._stop = threading.Event()
        self._thread = None
        # 2-second sliding windows for count/srv_count keyed by dst host and service
        self.window = deque()
        self.host_counts = defaultdict(int)
        self.service_counts = defaultdict(int)
        self.window_span_sec = 2.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        target = self._run_scapy if SCAPY_AVAILABLE else self._run_simulated
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _emit(self, record):
        try:
            self.events.put_nowait(record)
        except Exception:
            # Drop if queue is full
            pass

    def _gc_window(self, now_ts):
        # Remove old entries from sliding window and decrement counters
        while self.window and now_ts - self.window[0][0] > self.window_span_sec:
            _, host_key, svc_key = self.window.popleft()
            self.host_counts[host_key] -= 1
            self.service_counts[svc_key] -= 1
            if self.host_counts[host_key] <= 0:
                self.host_counts.pop(host_key, None)
            if self.service_counts[svc_key] <= 0:
                self.service_counts.pop(svc_key, None)

    def _handle_flow(self, proto, service, flag, src_bytes, dst_bytes, duration, dst_host):
        now_ts = time.time()
        host_key = dst_host
        svc_key = service
        # Add this connection to sliding window
        self.window.append((now_ts, host_key, svc_key))
        self.host_counts[host_key] += 1
        self.service_counts[svc_key] += 1
        self._gc_window(now_ts)

        record = {
            'protocol_type': proto,
            'service': service,
            'flag': flag,
            'src_bytes': float(max(0, src_bytes)),
            'dst_bytes': float(max(0, dst_bytes)),
            'duration': float(max(0, duration)),
            'count': int(self.host_counts.get(host_key, 0)),
            'srv_count': int(self.service_counts.get(svc_key, 0))
        }
        self._emit(record)

    def _run_scapy(self):
        # Track simple flows by 5-tuple to approximate duration and bytes
        flows = {}
        def on_packet(pkt):
            if self._stop.is_set():
                return False
            if not pkt.haslayer(IP):
                return
            ip = pkt[IP]
            proto = 'tcp' if pkt.haslayer(TCP) else ('udp' if pkt.haslayer(UDP) else 'icmp')
            sport = int(pkt[TCP].sport) if pkt.haslayer(TCP) else (int(pkt[UDP].sport) if pkt.haslayer(UDP) else 0)
            dport = int(pkt[TCP].dport) if pkt.haslayer(TCP) else (int(pkt[UDP].dport) if pkt.haslayer(UDP) else 0)
            service = _service_from_ports(sport, dport)
            flag = 'SF'
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                # Map TCP flags to NSL-KDD-ish
                if tcp.flags & 0x04:  # RST
                    flag = 'REJ'
                elif tcp.flags & 0x02 and not tcp.flags & 0x10:  # SYN without ACK
                    flag = 'S0'
                elif tcp.flags & 0x12:  # SYN+ACK
                    flag = 'S1'
                elif tcp.flags & 0x10:  # ACK
                    flag = 'SF'
            key = (ip.src, ip.dst, sport, dport, proto)
            rev = (ip.dst, ip.src, dport, sport, proto)
            length = int(len(pkt))
            now = time.time()
            if key in flows or rev in flows:
                k = key if key in flows else rev
                f = flows[k]
                if k == key:
                    f['src_bytes'] += length
                else:
                    f['dst_bytes'] += length
                f['last'] = now
                f['flag'] = flag
            else:
                flows[key] = {
                    'start': now,
                    'last': now,
                    'src_bytes': length,
                    'dst_bytes': 0,
                    'flag': flag,
                    'service': service,
                    'proto': proto,
                    'dst_host': ip.dst
                }
            # Emit completed/idle flows
            to_delete = []
            for fk, fv in flows.items():
                if now - fv['last'] > 1.0:
                    duration = fv['last'] - fv['start']
                    self._handle_flow(fv['proto'], fv['service'], fv['flag'], fv['src_bytes'], fv['dst_bytes'], duration, fv['dst_host'])
                    to_delete.append(fk)
            for fk in to_delete:
                flows.pop(fk, None)

        try:
            sniff(prn=on_packet, store=False, stop_filter=lambda p: self._stop.is_set())
        except Exception:
            # Fall back to simulated if scapy cannot run (permissions/driver)
            self._run_simulated()

    def _run_simulated(self):
        protos = ['tcp', 'udp', 'icmp']
        flags = ['SF', 'S0', 'REJ']
        services = ['http', 'ftp', 'smtp', 'domain_u', 'other']
        while not self._stop.is_set():
            proto = random.choice(protos)
            service = random.choice(services)
            flag = random.choice(flags)
            src_b = random.randint(0, 4000)
            dst_b = random.randint(0, 5000)
            duration = random.random() * 3.0
            dst_host = f"192.168.1.{random.randint(1,254)}"
            self._handle_flow(proto, service, flag, src_b, dst_b, duration, dst_host)
            time.sleep(0.3)

    def get_event_nowait(self):
        try:
            return self.events.get_nowait()
        except Empty:
            return None

    def iter_events(self):
        while not self._stop.is_set():
            try:
                evt = self.events.get(timeout=1.0)
                yield evt
            except Empty:
                continue 