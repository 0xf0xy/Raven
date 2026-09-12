import argparse
from importlib.metadata import version
import os

from raven.core.models import ScanStatus
from raven.core.scanner import Raven
from raven.parsers.ports import PortParser


def red(text: str) -> str:
    return f"\033[1;31m{text}\033[0m"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Raven: Network reconnaissance and port scanner.",
        epilog="Root privileges are required for packet-based scans.",
        add_help=False,
    )

    target = parser.add_argument_group("Target Settings")

    target.add_argument(
        "host",
        help="Target host or IP address",
    )

    target.add_argument(
        "-p",
        "--ports",
        help="Ports to scan (comma-separated or range)",
    )

    scan = parser.add_argument_group("Scan Settings")

    scan_type = scan.add_mutually_exclusive_group()

    scan_type.add_argument(
        "-s",
        "--syn",
        action="store_true",
        help="Perform TCP SYN scan",
    )

    scan_type.add_argument(
        "-f",
        "--fin",
        action="store_true",
        help="Perform TCP FIN scan",
    )

    scan_type.add_argument(
        "-n",
        "--null",
        action="store_true",
        help="Perform TCP NULL scan",
    )

    scan_type.add_argument(
        "-x",
        "--xmas",
        action="store_true",
        help="Perform TCP XMAS scan",
    )

    scan_type.add_argument(
        "-u",
        "--udp",
        action="store_true",
        help="Perform UDP scan",
    )

    scan_type.add_argument(
        "-i",
        "--icmp",
        action="store_true",
        help="Perform ICMP host discovery",
    )

    scan_type.add_argument(
        "-b",
        "--banner",
        action="store_true",
        help="Grab service banners and versions",
    )

    scan.add_argument(
        "-t",
        "--ttl",
        type=int,
        default=64,
        help="Custom IP TTL value (default: 64)",
    )

    scan.add_argument(
        "-w",
        "--workers",
        type=int,
        default=50,
        help="Number of concurrent workers (default: 50)",
    )

    scan.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="Connection timeout in seconds (default: 1.0)",
    )

    meta = parser.add_argument_group("Information")

    meta.add_argument(
        "-V",
        "--verbose",
        action="store_true",
        help="Show closed and filtered ports",
    )

    meta.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Raven v{version('raven')}",
        help="Show program version",
    )

    meta.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help menu",
    )

    return parser


def get_scan_type(args) -> str:
    if args.syn:
        return "SYN"

    if args.fin:
        return "FIN"

    if args.null:
        return "NULL"

    if args.xmas:
        return "XMAS"

    if args.udp:
        return "UDP"

    if args.icmp:
        return "ICMP"

    if args.banner:
        return "BANNER"

    return "SYN"


def print_header(target: str, scan_type: str, ports: str | None) -> None:
    print(red("Raven - Network reconnaissance and port scanner"))
    print("─" * 50)

    print("\nScan Settings")
    print("─" * 35)
    print(f"Target    : {red(target)}")
    print(f"Scan Type : {red(scan_type)}")

    if ports:
        print(f"Ports     : {red(ports)}")

    print()


def print_tcp_udp_results(result, verbose: bool) -> None:
    found = False

    print(f"{'PORT':<10}" f"{'PROTO':<10}" f"STATUS")
    print("─" * 40)

    for port in result.ports:
        if port.status == ScanStatus.OPEN or verbose:
            found = True

            status = port.status.value.upper()

            print(f"{port.port:<10}" f"{port.protocol.upper():<10}" f"{red(status)}")

    if not found:
        print(f"{red('[x]')} No open ports found.")


def print_banner_results(result) -> None:
    found = False

    print(f"{'PORT':<8}" f"{'SERVICE':<10}" f"VERSION")
    print("─" * 35)

    for port in result.ports:
        if not port.banner:
            continue

        found = True

        print(
            f"{port.port:<8}"
            f"{port.service.upper() or "-":<10}"
            f"{red(port.version) or "-"}"
        )

    if not found:
        print(f"{red('[x]')} No banners found.")


def print_icmp_result(status: ScanStatus) -> None:
    print(f"Host Status : " f"{red(status.value.upper())}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if os.geteuid() != 0:
        parser.error("you must run this tool with root privileges.")

    if args.ttl <= 0 or args.ttl > 255:
        parser.error("TTL must be between 1 and 255.")

    if args.workers <= 0:
        parser.error("workers must be greater than zero.")

    if args.timeout <= 0:
        parser.error("timeout must be greater than zero.")

    scan_type = get_scan_type(args)

    if scan_type == "ICMP":
        ports = None

    elif args.ports:
        try:
            ports = PortParser.parse(args.ports, [])

        except ValueError as exc:
            parser.error(str(exc))

    else:
        ports = None

    print_header(target=args.host, scan_type=scan_type, ports=args.ports)

    scanner = Raven(timeout=args.timeout, workers=args.workers)

    try:
        if scan_type == "ICMP":
            status = scanner.scan_icmp(args.host)

            print_icmp_result(status)

            return

        if scan_type == "UDP":
            if ports is None:
                ports = scanner.common_ports

            result = scanner.scan_udp(target=args.host, ports=ports, ttl=args.ttl)

            print_tcp_udp_results(result, args.verbose)

            return

        if scan_type == "BANNER":
            if ports is None:
                ports = scanner.common_ports

            result = scanner.grab_banners(target=args.host, ports=ports)

            print_banner_results(result)

            return

        if ports is None:
            ports = scanner.common_ports

        result = scanner.scan_tcp(
            target=args.host, ports=ports, flag=scan_type, ttl=args.ttl
        )

        print_tcp_udp_results(result, args.verbose)

    except KeyboardInterrupt:
        print(f"\n{red('[!]')} Scan stopped by user.")
        raise SystemExit(130)

    except ValueError as exc:
        print(f"{red('[x]')} {exc}")
        raise SystemExit(1)

    except PermissionError:
        print(f"{red('[x]')} " "Root privileges are required.")
        raise SystemExit(1)

    except Exception as exc:
        print(f"{red('[x]')} {exc}")
        raise SystemExit(1)
