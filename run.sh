#!/bin/bash
if [ -z "$VIRTUAL_ENV" ]; then
	echo 'need opt virtualenv'
	exit 1
fi
LD_LIBRARY_PATH=$VIRTUAL_ENV/opt/lib
python tvauction/processor_pulp.py
