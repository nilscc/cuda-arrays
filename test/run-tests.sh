#!/bin/sh

TESTS=$@

for TEST in $TESTS
do

    echo -n "Running \"${TEST}\"..."

    OUTPUT=$( { ./$TEST; } 2>&1 )

    RESULT=$?

    if [ $RESULT -eq 0 ]
    then
        echo " OK"
    else
        echo " FAIL:"
        echo
        echo "$OUTPUT"
        echo
        exit $RESULT
    fi

done
