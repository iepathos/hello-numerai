#!/bin/bash

arch=$(uname)

if [ "$arch" = "Darwin" ]; then
	# libomp dependency on macos for lightgbm
	brew install poetry libomp
fi


poetry env use 3.10.13
poetry install
