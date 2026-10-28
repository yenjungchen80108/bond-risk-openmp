#!/bin/bash
# Run ./portfolio for several thread counts and OpenMP schedules, REPS times each,
# and append every run to scaling.csv (raw data; medians are computed by the plot script).
#
# Usage:   ./run_scaling.sh                 # default thread list
#          THREADS="1 2 4 8" ./run_scaling.sh
# Needs:   bonds.csv (make data N=1000000) and ./portfolio (make)

THREADS=${THREADS:-"1 2 4 6 8 10 12"}
SCHEDULES=${SCHEDULES:-"dynamic static"}
REPS=${REPS:-5}
OUT=${OUT:-scaling.csv}

# APPEND=1 adds new rows to an existing $OUT instead of starting a new file.
if [ -z "$APPEND" ]; then
  echo "schedule,threads,compute_s,total_s,read_s,write_s" > "$OUT"
fi

for sched in $SCHEDULES; do
  for p in $THREADS; do
    for k in $(seq 1 "$REPS"); do
      out=$(OMP_SCHEDULE=$sched OMP_NUM_THREADS=$p ./portfolio)
      compute=$(echo "$out" | awk '/time:/ {print $6}')
      # line format: e2e total: T s (read R, compute C, write W)
      total=$(echo "$out" | awk '/e2e total:/ {print $3}')
      read_s=$(echo "$out" | awk '/e2e total:/ {gsub(",","",$6); print $6}')
      write_s=$(echo "$out" | awk '/e2e total:/ {gsub(/[,)]/,"",$10); print $10}')
      # "dynamic,1000" contains a comma, which would break the CSV: store it as dynamic_1000
      label=${sched//,/_}
      echo "$label,$p,$compute,$total,$read_s,$write_s" >> "$OUT"
    done
    echo "done: schedule=$sched threads=$p"
  done
done
echo "wrote $OUT"