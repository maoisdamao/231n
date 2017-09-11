#!/bin/bash

# accept 1st param as assignment order
cd ./assignment$1
source .env/bin/activate
jupyter-notebook --no-browser --port=7000

