#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENVIRONMENT_FILE=environment.yml
echo "Using ${ENVIRONMENT_FILE} ..."

# Make sure mamba is installed
installed=$(conda list ^mamba)

if [[ "$installed" == *"mamba"* ]]
then
    echo "Mamba installation found..."
else
    echo "Mamba installation not found, installing..."
    conda install -c conda-forge mamba
fi

# Create library environment.
mamba env create -f "${SCRIPT_DIR}/${ENVIRONMENT_FILE}" \
&& eval "$(conda shell.bash hook)" \
&& conda activate cs3630_proj5 \
&& echo "Setup complete. Activate later with: conda activate cs3630_proj5"
