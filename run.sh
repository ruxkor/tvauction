#!/bin/bash
if [ -z "$VIRTUAL_ENV" ]; then
	echo 'need opt virtualenv'
	exit 1
fi
export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib
python tvauction/processor_pulp.py
