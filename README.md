# mcs-nyu-baselines

## Install

 git clone https://github.com/NextCenturyCorporation/mcs-nyu-baselines.git
 cd mcs-nyu-baselines
 git checkout caci-eval-4
 conda create -n "baselines" -y python=3.8
 conda activate baselines
 python -m pip install -r requirements.txt

Edit run_baselines.py to set the MCS Unity app to the appropriate
location for your machine.

## Run

When you run it, you need to give it the name of the scene file. To do
all in a directory:

  for file in `ls ~/eval4scenes/*.json`; do echo $file; python run_baselines.py $file; done

run_baselines.py will use info in the json file to run the appropriate
baseline algorithm, so there will not be an error if you give it a
json file it cannot use.
