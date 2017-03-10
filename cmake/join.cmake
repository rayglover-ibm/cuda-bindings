# rejoin semi-colon (;) delimited elements of a string with
# the given GLUE string
function (join values GLUE output)
    string (REGEX REPLACE "([^\\]|^);" "\\1${GLUE}" _tmp_str "${values}")
    # fixes escaping
    string (REGEX REPLACE "[\\](.)" "\\1" _tmp_str "${_tmp_str}")
    set (${output} "${_tmp_str}" PARENT_SCOPE)
endfunction ()

# rejoin a semi-colon (;) delimited paths with
# the platform path separator
function (join_paths values OUT)
    if (WIN32)
        join ("${values}" ";" result)
    else ()
        join ("${values}" ":" result)
    endif ()
    set (${OUT} "${result}" PARENT_SCOPE)
endfunction ()