"""Pull every DNS query out of the CIC PCAPs into CSVs, one per capture (same relative path).

Usage: python extract_names.py PCAP_DIR OUT_DIR

The captures were recorded with a 96-byte snapshot length, so long queries are cut off.
For those rows qname is the part that survived, qtype is empty, truncated is 1, and
qname_len is still the full length of the name, worked out from the UDP length field.
"""
import csv
import sys
from pathlib import Path

import dpkt

COLUMNS = ["time", "qname", "qtype", "qname_len", "truncated"]
FIXED_BYTES = 26  # UDP header 8 + DNS header 12 + qtype and qclass 4 + the name's own framing 2


def dns_queries(path):
    """Yield (timestamp, UDP segment) for every packet sent to port 53."""
    with open(path, "rb") as fh:
        for ts, buf in dpkt.pcap.Reader(fh):
            try:
                udp = dpkt.ethernet.Ethernet(buf).data.data  # Ethernet -> IP -> UDP
            except (dpkt.UnpackError, AttributeError):
                continue
            if isinstance(udp, dpkt.udp.UDP) and udp.dport == 53:
                yield ts, udp


def surviving_name(message):
    """The part of the query name that was captured: pad the cut message with zeros so the
    parser finds an end of name, then strip the zeros off the last label."""
    return dpkt.dns.DNS(message + bytes(512)).qd[0].name.rstrip("\x00")


def query_row(ts, udp):
    try:
        question = dpkt.dns.DNS(udp.data).qd[0]
        return [ts, question.name, question.type, len(question.name), 0]
    except dpkt.UnpackError:  # the message was cut short by the snapshot length
        name, full_length = surviving_name(udp.data), udp.ulen - FIXED_BYTES
        return [ts, name, "", full_length, int(len(name) < full_length)]


def main(pcap_dir, out_dir):
    pcap_dir, out_dir = Path(pcap_dir), Path(out_dir)
    for path in sorted(pcap_dir.rglob("*.pcap")):
        rows = [query_row(ts, udp) for ts, udp in dns_queries(path)]
        target = out_dir / path.relative_to(pcap_dir).with_suffix(".csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", newline="") as fh:
            csv.writer(fh).writerows([COLUMNS, *rows])
        print(f"{path.relative_to(pcap_dir)}: {len(rows)} queries, {sum(row[4] for row in rows)} truncated")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
