#!/bin/bash

# Copyright 2016 Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Usage: bazel run //tools:visualizer -- \
#  --image_path="`pwd`/data/traffic_frames/*.bmp" \
#  --output_path="`pwd`/visualized"
#
# Output: ../visualized/visualized_0001.bmp, ../visualized/visualized_0002.bmp
#
# Assumes that /data/traffic_frames/0001.bmp.txt exists and each line is in the
# following format:
#  label: score [y1 x1 y2 x2]

. "$0.runfiles/shFlags/shflags" || exit -1

DEFINE_string image_path 'test.bmp' 'The image file. '
DEFINE_string output_path '/tmp' 'Folder to output visualized images to.'
DEFINE_float score_threshold '0.4' 'Cutoff threshold.'
DEFINE_boolean view_now false 'Whether or not launching eog to visualize the result.'

function visualize() {
  image_fn=$1
  result_fn=$2
  vis_fn=$3
  encoder=$4

  if [ -f "$image_fn" ] && [ -f "$result_fn" ]; then

    col=$(identify  "$image_fn"| cut -d ' ' -f 3 | cut -d 'x' -f 1)
    row=$(identify  "$image_fn"| cut -d ' ' -f 3 | cut -d 'x' -f 2)
    box=$(cut -d \[ -f 2- "${result_fn}"| cut -d \] -f 1)
    xmins=($(echo "${box}" | cut -d , -f 2))
    ymins=($(echo "${box}" | cut -d , -f 1))
    xmaxs=($(echo "${box}" | cut -d , -f 4))
    ymaxs=($(echo "${box}" | cut -d , -f 3))
    labels=($(cut -d : -f 1 "${result_fn}"))
    scores=($(cut -d : -f 2 "${result_fn}" | cut -d \[ -f 1))

    convert "${image_fn}" "${encoder}":"${vis_fn}"
    for (( i=0;i<${#scores[@]};i++ )); do

    bx=${xmins[$i]}
    by=${ymins[$i]}
    ex=${xmaxs[$i]}
    ey=${ymaxs[$i]}
    label=${labels[$i]}
    score=${scores[$i]}

    meets_threshold=$(echo "$score > $FLAGS_score_threshold" | bc -l)
    if [[ meets_threshold -eq 1 ]]; then
      x1=$(echo - | awk "{print $bx*$col}"  )
      y1=$(echo - | awk "{print $by*$row}" )
      x2=$(echo - | awk "{print $ex*$col}" )
      y2=$(echo - | awk "{print $ey*$row}" )
      y11=$(echo - | awk  "{print $y1-20}" )
      convert "${vis_fn}" -strokewidth 1 -draw "fill none stroke red  rectangle $x1,$y1,$x2,$y2" -strokewidth 1 -pointsize 10 -draw "stroke green fill green text $x1,$y11 '$label:$score' "  "${encoder}":"${vis_fn}"
    fi
    done
    if [[ ${FLAGS_view_now} -eq ${FLAGS_true} ]]; then
      eog "$vis_fn"
    fi

  else

    echo "Both ${image_fn} and ${result_fn} must exist."

  fi
}

function main() {
  if [ ! -d "${FLAGS_output_path}" ]
  then
    mkdir "${FLAGS_output_path}"
  fi

  for path in ${FLAGS_image_path}
  do
    basename=`basename ${path}`
    basename="${basename%.*}"
    image_fn="${path}"
    result_fn="${path}.txt"
    vis_fn="${FLAGS_output_path}/${basename}_visualized.bmp"
    encoder="bmp"

    visualize \
      ${image_fn} \
      ${result_fn} \
      ${vis_fn} \
      ${encoder}
  done
}

FLAGS "$@" || exit 1
eval set -- "${FLAGS_ARGV}"
main
