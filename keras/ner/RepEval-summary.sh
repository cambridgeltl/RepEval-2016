#!/bin/bash

set -e
set -u

egrep -v '^[0-9][[:space:]]' | perl -pe 's/(\S+) win (\d+).*\n/$1\t$2\t/' | perl -pe 's/^(\S+\s+\S+\s+).*?(\S+\s+\S+)$/$1$2/'