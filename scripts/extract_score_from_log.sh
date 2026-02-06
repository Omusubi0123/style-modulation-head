#!/bin/bash

INPUT_LOG_FILE=${INPUT_LOG_FILE:-"logs/input_log_file.log"}
OUTPUT_LOG_FILE=${OUTPUT_LOG_FILE:-"logs/output_log_file.log"}

TRAIT=${TRAIT:-"evil"}

perl -ne '
    push @lines, $_;
    shift @lines if @lines > 3;
    if (@lines == 3 &&
        ($lines[0] =~ /^data\/) &&
        $lines[2] =~ /^coherence/) {
        print join("", @lines), "\n";
    }
' $INPUT_LOG_FILE >> $OUTPUT_LOG_FILE
