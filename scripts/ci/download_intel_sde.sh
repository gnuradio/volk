#!/bin/bash

set -e
set -x

function test_sde
{
    if ! [ -f ${SDE} ]; then
        echo "1"
    else
        ${SDE} -- ls > /dev/null
        echo $?
    fi
}

mkdir -p cache
cd cache

[ -z "${SDE_VERSION}" ] && SDE_VERSION=sde-external-8.50.0-2020-03-26-lin
[ -z "${SDE_URL}" ] && SDE_URL=http://software.intel.com/content/dam/develop/external/us/en/protected/
[ -z "${SDE}" ] && SDE=${SDE_VERSION}/sde64


if [ _$(test_sde) == _0 ]; then
    MSG="found working version: ${SDE_VERSION}"
else
    MSG="downloading: ${SDE_VERSION}"
    wget ${SDE_URL}/${SDE_VERSION}.tar.bz2
    tar xvf ${SDE_VERSION}.tar.bz2
fi

echo $SDE
