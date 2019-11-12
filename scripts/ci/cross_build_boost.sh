#!/bin/bash
# fail script immediately on any errors in external commands
set -e

set -x

## arguments
##   for NEONv7: "arm"     "linux-gnueabihf-g++"
##   for NEONv8: "aarch64" "linux-gnu-g++"

##source travis_retry.sh

## determine if the build exists and is ok

mkdir -p cache
cd cache

if ! [ -f boost_1_66_0.tar.bz2 ]; then
    wget --no-check-certificate https://sourceforge.net/projects/boost/files/boost/1.66.0/boost_1_66_0.tar.bz2
fi

if ! [ -f boost_1_66_0/stage/lib/libboost_system.so ] || ! [ -f boost_1_66_0/stage/lib/libboost_filesystem.so ]; then
    tar xvf boost_1_66_0.tar.bz2 2>&1 | tail -10
    cd boost_1_66_0
    ./bootstrap.sh
    echo "using gcc" ":" "$1" ":" "$1-$2 ;" > user-config.jam
    BOOST_BUILD_PATH=./ ./b2 toolset=gcc-$1 --with-system --with-filesystem
fi
