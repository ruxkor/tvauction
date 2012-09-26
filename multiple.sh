#!/bin/bash -e

SLOT_QTY=$(echo "168 * 2" | bc)
BIDDER_QTY=50

INITIAL_VECTOR="vcg zero"
CORE_ALGORITHM="trim switch reuse"
RANDOM_SEED="1 2 3 4 5"

mkdir -p logs
pushd logs &>/dev/null

DFORMAT="%F_%H-%M-%S"
echo -n "running"
for iv in $INITIAL_VECTOR; do
	for algo in $CORE_ALGORITHM; do
		for rs in $RANDOM_SEED; do
			log_suffix="$(date +$DFORMAT)-$SLOT_QTY-$BIDDER_QTY-$iv-$algo-$rs.log"
			log_file="log-$log_suffix"
			debug_file="debug-$log_suffix"
			echo -n "."
			echo -e "$(date) $SLOT_QTY $BIDDER_QTY $iv $algo $rs\n-------------\n" >$log_file
			LOG_LEVEL=20 python ../simulation/algorithmic_simulator.py --slot-qty=$SLOT_QTY --bidder-qty=$BIDDER_QTY --initial-vector=$iv --core-algorithm=$algo --random-seed=$rs 2>$debug_file >>$log_file
		done
	done
done
popd >/dev/null
echo "done"
