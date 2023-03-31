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
 do change_line "python -u" "python -u motion_segmentation/removert/transform_data.py $i" run_postprocess.sh && sbatch run_postprocess.sh;
  done
