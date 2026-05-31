#!/bin/bash
# Copyright 2026 Marcus Müller
source "$(dirname "$(realpath "$0")")/common.bash"
source /etc/os-release
export DEBIAN_FRONTEND=noninteractive

check_sccache_s3() {
  type -p sccache >/dev/null && (sccache -h | grep -q ' *S3: *true')
}
install_deb() {
  apt-get install -y sccache
}

if check_sccache_s3; then
  echo "installed $(sccache --version) has S3 support"
  exit 0
fi

if [[ "${ID}" = "ubuntu" && -n "${VERSION_ID}" ]]; then
  ver_major="$(echo "${VERSION_ID}" | cut -f1 -d.)"
  if [[ "${ver_major}" -ge 26 ]]; then
    install_deb && exit 0
  fi
fi

if [[ "${ID}" = "debian" ]]; then
  ver_major=0
  if [[ -n "${VERSION_ID}" ]]; then
    ver_major="$(echo "${VERSION_ID}" | cut -f1 -d.)"
  else
    # non-release, um??? we know that this works for forky=14, but uuum
    # try to install, and if that succeeds, check availability of S3,
    # if missing, uninstall.
    if install_deb; then
      gh_message "speculative sccache installation" "Don't know debian ${PRETTY_NAME}, trying to install deb"
      if check_sccache_s3; then
        echo "using debian packaged $(sccache --version), which has S3 support"
        exit 0
      else
        gh_message "insufficient debian" "available sccache $(sccache --version) has no S3 support."
        apt-get purge -y sccache
      fi
    fi
  fi
  if [[ "${ver_major}" -ge 14 ]]; then
    install_deb && exit 0
  fi
fi

gh_message "sccache external download" "need to download from external source"

if [[ -z "${ARCH}" ]] && type -p dpkg >/dev/null; then
  ARCH="$(dpkg --print-architecture)"
fi
if [[ -z "${ARCH}" ]] && type -p rpm >/dev/null; then
  ARCH="$(rpm --eval '%{_arch}')"
fi
if [[ -z "${ARCH}" ]] && type -p arch >/dev/null; then
  ARCH="$(arch)"
fi

sccache_release="0.15.0"
sccache_arch="${ARCH}"
abi_suffix=""
case "${ARCH}" in
"x86_64" | "amd64")
  sccache_arch="x86_64"
  ;;
"arm64" | "aarch64")
  sccache_arch="aarch64"
  ;;
"arm" | "armv7")
  sccache_arch="armv7"
  abi_suffix="eabi"
  ;;
"x86" | "i386" | "i686" | "pentium")
  sccache_arch="i686"
  ;;
s390*)
  sccache_arch="s390x"
  ;;
risc64*)
  sccache_arch="riscv64gc"
  ;;
esac
URL="https://github.com/mozilla/sccache/releases/download/v${sccache_release}/sccache-v${sccache_release}-${sccache_arch}-unknown-linux-musl${abi_suffix}.tar.gz"
echo "Getting '${URL}'"
(
  set -e
  cd /tmp/
  curl -s -L "${URL}" | tar xz
  cd sccache-*
  cp sccache /usr/bin
  check_sccache_s3
) || fail_with_message "no sccache installable" "couldn't install a suitable sccache"
