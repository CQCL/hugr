#!/bin/bash

case $1 in
    P-low)
        echo Low;;
    P-medium)
        echo Medium;;
    P-high)
        echo High;;
    P-critical)
        echo Highest;;
    *)
        exit 1;;
esac

exit 0
