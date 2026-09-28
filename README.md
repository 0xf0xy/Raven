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
> Raven is intended for educational, research, and authorized security testing purposes only.

<br>

## > About

Raven is a tool for network reconnaissance and port scanning.

It supports TCP, UDP and ICMP scanning, TCP flag manipulation, service banner grabbing, configurable port ranges and concurrent scanning.

Raven includes the following scan types:

* SYN
* FIN
* NULL
* XMAS
* UDP
* ICMP
* BANNER

<br>

## > Installation

```bash
git clone https://github.com/0xf0xy/Raven.git
cd Raven
pip install .
```

Check the installation:

```bash
raven -h
```

Maybe you need to install as root.

<br>

## > Usage

Basic scan:

```bash
sudo raven 192.168.1.10
```

Scan specific ports:

```bash
sudo raven 192.168.1.10 -p 22,80,443
```

Use a specific scan type:

```bash
sudo raven 192.168.1.10 -p 22,80,443 -s
```

For all available options:

```bash
raven -h
```

---

<p align="center">
  <a href="https://github.com/0xf0xy"><b>0xf0xy</b></a> •
  <a href="./LICENSE"><b>MIT License</b></a>
</p>
