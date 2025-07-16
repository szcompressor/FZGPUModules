#!/bin/bash -
#title           :download-sdrb-data.sh
#description     :This script will download sample dataset from SDRBench.
#author          :Cody Rivera
#copyright       :(C) 2020 by Washington State University, The University of Alabama, Argonne National Laboratory
#license         :See LICENSE in top-level directory
#date            :2020-09-28
#version         :0.1
#usage           :./download-sdrb-data.sh data-dir
#==============================================================================

banner()
{
    echo "+------------------------------------------+"
    printf "| %-40s |\n" "`date`"
    echo "|                                          |"
    printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
    echo "+------------------------------------------+"
}

sbanner()
{
    echo "+------------------------------------------+"
    printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
    echo "+------------------------------------------+"
}

pwd; hostname; date;

if [ "$#" -ne 1 ]; then
    echo "usage: $0 data-dir";
    exit 2;
fi

if ! mkdir -p $1; then
    echo "Cannot download data to $1";
    exit 2;
fi

DATA_DIR=$1

banner "Downloading Data";

banner "CESM";
if [ ! -f "$DATA_DIR/SDRBENCH-CESM-ATM-1800x3600.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-1800x3600.tar.gz
fi

if [ ! -d "$DATA_DIR/CESM_1800x3600" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-CESM-ATM-1800x3600.tar.gz
    # Rename extracted directory if it doesn't match the desired format
    if [ -d "$DATA_DIR/1800x3600" ]; then
        mv "$DATA_DIR/1800x3600" "$DATA_DIR/CESM_1800x3600"
    fi
fi

banner "EXAALT";
if [ ! -f "$DATA_DIR/SDRBENCH-EXAALT-2869440.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXAALT/SDRBENCH-EXAALT-2869440.tar.gz 
fi
  
if [ ! -d "$DATA_DIR/EXAALT_2869440" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-EXAALT-2869440.tar.gz
    # Rename extracted directory if it doesn't match the desired format
    if [ -d "$DATA_DIR/2869440" ]; then
        mv "$DATA_DIR/2869440" "$DATA_DIR/EXAALT_2869440"
    fi
fi

banner "Hurricane Isabel";
if [ ! -f "$DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/Hurricane-ISABEL/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
fi

if [ ! -d "$DATA_DIR/HURR_100x500x500" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-Hurricane-ISABEL-100x500x500.tar.gz
    # Rename extracted directory if it doesn't match the desired format
    if [ -d "$DATA_DIR/100x500x500" ]; then
        mv "$DATA_DIR/100x500x500" "$DATA_DIR/HURR_100x500x500"
    fi
fi

# !!! LARGE DATASETS !!! -- test script will not fail if these aren't present
banner "HACC 1GB";
if [ ! -f "$DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-medium-size.tar.gz
fi

if [ ! -d "$DATA_DIR/HACC_280953867" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/EXASKY-HACC-data-medium-size.tar.gz
    # Rename extracted directory if it doesn't match the desired format
    if [ -d "$DATA_DIR/HACC_M_280953867" ]; then
        mv "$DATA_DIR/HACC_M_280953867" "$DATA_DIR/HACC_280953867"
    elif [ -d "$DATA_DIR/280953867" ]; then
        mv "$DATA_DIR/280953867" "$DATA_DIR/HACC_280953867"
    fi
fi

# banner "HACC 4GB";
# if [ ! -f "$DATA_DIR/EXASKY-HACC-data-big-size.tar.gz" ]; then
#     wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/HACC/EXASKY-HACC-data-big-size.tar.gz
# fi

# if [ ! -d "$DATA_DIR/HACC_1billion" ]; then
#     tar -C $DATA_DIR -xvf $DATA_DIR/EXASKY-HACC-data-big-size.tar.gz
#     # Rename extracted directory if it doesn't match the desired format
#     if [ -d "$DATA_DIR/1billionparticles_onesnapshot" ]; then
#         mv "$DATA_DIR/1billionparticles_onesnapshot" "$DATA_DIR/HACC_1billion"
#     fi
# fi

banner "NYX"
if [ ! -f "$DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz" ]; then
    wget -c -P $DATA_DIR https://g-8d6b0.fd635.8443.data.globus.org/ds131.2/Data-Reduction-Repo/raw-data/EXASKY/NYX/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
fi

if [ ! -d "$DATA_DIR/NYX_512x512x512" ]; then
    tar -C $DATA_DIR -xvf $DATA_DIR/SDRBENCH-EXASKY-NYX-512x512x512.tar.gz
    # Rename extracted directory if it doesn't match the desired format
    if [ -d "$DATA_DIR/512x512x512" ]; then
        mv "$DATA_DIR/512x512x512" "$DATA_DIR/NYX_512x512x512"
    fi
fi