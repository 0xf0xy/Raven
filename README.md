<h1 align="center">RAVEN</h1>

<p align="center">
  <em>network reconnaissance and port scanner</em>
</p>

<p align="center">
  <img src="https://img.shields.io/github/release/0xf0xy/Raven?color=AAAAAA&style=for-the-badge&labelColor=111111"/>
  <img src="https://img.shields.io/badge/python-3.10+-AAAAAA?style=for-the-badge&logo=python&logoColor=FFFFFF&labelColor=111111"/>
  <img src="https://img.shields.io/github/license/0xf0xy/Raven?color=AAAAAA&style=for-the-badge&labelColor=111111"/>
</p>

<br>

> [!WARNING]
>
> **Raven is intended for educational, research, and authorized security testing purposes only.**
>
> Scans should only be performed against systems and networks where you have explicit permission to perform security testing.
>
> The author is not responsible for misuse of this software.

<br>

## > Overview

**Raven** is a modular network reconnaissance scanner designed to identify exposed ports and services through multiple scanning techniques.

Instead of relying on a single scan method, Raven separates the scanning core from the CLI and supports different protocols through independent modules.

Raven supports:

* TCP port scanning
* UDP port scanning
* ICMP host discovery
* TCP flag manipulation
* Service banner grabbing
* Configurable port ranges
* Concurrent scanning

Raven was built primarily as a learning and security research project around:

* Network reconnaissance research
* Service exposure analysis
* Network security experiments
* CLI application design
* Modular Python architecture

<br>

## > How It Works

Raven starts by resolving the target hostname to an IPv4 address. It then
selects a probe based on the requested scan type and applies it to every target
port using a pool of worker threads.

For example:

```bash
raven 192.168.1.10 -p 22,80,443 -s
```

For a TCP scan, Raven:

1. Resolves the target host.
2. Sends a TCP probe with the selected flag to each port.
3. Interprets the responses as open, closed, or filtered states.
4. Collects and sorts the results by port.
5. Displays the selected results in the terminal.

The same workflow is adapted for the other scan modes:

* UDP sends datagrams and evaluates replies or ICMP errors.
* ICMP checks whether the target responds to an echo request.
* Banner mode connects to open TCP ports and reads service information.

When no port list is provided, Raven loads its common ports from
`raven/data/common.txt`.

<br>

## > Scan Types

| Scan Type | Description |
| --------- | ----------- |
| `SYN`     | Sends a TCP SYN probe to identify port state. |
| `FIN`     | Sends a TCP FIN probe using a custom TCP flag. |
| `NULL`    | Sends a TCP probe without flags. |
| `XMAS`    | Sends a TCP probe with FIN, PSH and URG flags. |
| `UDP`     | Sends UDP probes to identify UDP services. |
| `ICMP`    | Checks whether the target responds to ICMP requests. |
| `BANNER`  | Attempts to identify services and versions on TCP ports. |

<br>

## > Installation

### Requirements

* Python 3.10+
* `pip`
* root privileges

Raven has no external runtime dependencies.

Clone the repository:

```bash
git clone https://github.com/0xf0xy/Raven.git
cd Raven
```

Install Raven:

```bash
pip install .
```

Verify the installation:

```bash
raven -h
```

<br>

## > Usage

Provide the host that should be scanned and the scanning options:

```bash
raven <host> [options]
```

For available commands and options:

```bash
raven -h
```

<br>

## > Project Status

Raven is an experimental project focused on security research, experimentation and learning.

The project is under active development and its architecture and behavior may change between releases.

<br>

---

<p align="center">
  <a href="https://github.com/0xf0xy"><b>0xf0xy</b></a> •
  <a href="./LICENSE"><b>MIT License</b></a>
</p>
