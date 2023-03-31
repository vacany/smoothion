#!/bin/bash

function escape_slashes {
    sed 's/\//\\\//g'
}


function change_line {
    local OLD_LINE_PATTERN=$1; shift
    local NEW_LINE=$1; shift
    local FILE=$1

    local NEW=$(echo "${NEW_LINE}" | escape_slashes)
    # FIX: No space after the option i.
    sed -i.bak '/'"${OLD_LINE_PATTERN}"'/s/.*/'"${NEW}"'/' "${FILE}"
    mv "${FILE}.bak" /tmp/
}



for i in {0..21};
 do change_line "python -u" "python -u motion_segmentation/ego_supervision/common.py $i" run_clustering.sh && sbatch run_clustering.sh;
  done
