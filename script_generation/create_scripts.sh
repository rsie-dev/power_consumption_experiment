#!/usr/bin/env bash

MULTIMETER="377D11895400"
SENSOR="/dev/ttyUSB0"


OPTIONS="--multimeter $MULTIMETER"
OPTIONS+=" --with-sensor $SENSOR --mon-temp 1"
OPTIONS+=" --runs 30"
OPTIONS+=" --use-time-timings"
OPTIONS+=" --use-dut-timings"

DELAY=3
RUN_OPTIONS=$OPTIONS
RUN_OPTIONS+=" -t host"
RUN_OPTIONS+=" --head-delay $DELAY --head-delay-max 60 --tail-delay $DELAY"
RUN_OPTIONS+=" --warmup 30"
RUN_OPTIONS+=" --data-folder /data"

venv/bin/generator --script data_gen.py --host visionfive2 --ip 192.168.5.104 --multimeter $MULTIMETER --use-time-timings -t datagen --data-folder data


venv/bin/generator --script raspi5_baseline.py --host raspi5 --ip 192.168.5.102 $OPTIONS -t baseline
venv/bin/generator --script radxax4_baseline.py --host radxax4 --ip 192.168.5.103 $OPTIONS -t baseline
venv/bin/generator --script visionfive2_baseline.py --host visionfive2 --ip 192.168.5.104 $OPTIONS -t baseline


venv/bin/generator --script raspi5_1.py --host raaspi5 --ip 192.168.5.102 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set sensor imagelarge
venv/bin/generator --script raspi5_2.py --host raaspi5 --ip 192.168.5.102 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set webster textlarge
venv/bin/generator --script raspi5_3.py --host raaspi5 --ip 192.168.5.102 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set sensor imagelarge
venv/bin/generator --script raspi5_4.py --host raaspi5 --ip 192.168.5.102 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set webster textlarge

venv/bin/generator --script radxax4_1.py --host radxax4 --ip 192.168.5.103 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set sensor imagelarge
venv/bin/generator --script radxax4_2.py --host radxax4 --ip 192.168.5.103 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set webster textlarge
venv/bin/generator --script radxax4_3.py --host radxax4 --ip 192.168.5.103 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set sensor imagelarge
venv/bin/generator --script radxax4_4.py --host radxax4 --ip 192.168.5.103 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set webster textlarge

venv/bin/generator --script visionfive2_1.py --host visionfive2 --ip 192.168.5.104 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set sensor imagelarge
venv/bin/generator --script visionfive2_2.py --host visionfive2 --ip 192.168.5.104 $RUN_OPTIONS --tool gzip pigz bzip2 lbzip2 bzip3 --data-set webster textlarge
venv/bin/generator --script visionfive2_3.py --host visionfive2 --ip 192.168.5.104 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set sensor imagelarge
venv/bin/generator --script visionfive2_4.py --host visionfive2 --ip 192.168.5.104 $RUN_OPTIONS --tool xz lz4 lzop zstd --data-set webster textlarge
