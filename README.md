# COLA
## Install
. To create a conda environment:
```
conda create -n your_env_name python=3.8
```
Activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda activate your_env_name
pip install -r requirements.txt
```
Download Carla 0.9.4, and git clone macad_gym from its github Repository.

- Fork/Clone the repository to your workspace:
  `git clone && cd macad-gym`
- Create a new conda env named "macad-gym" and install the required packages:
`conda env create -f conda_env.yml`
- Activate the `macad-gym` conda python env:
`source activate macad-gym`
- Install the `macad-gym` package:
`pip install -e .`
- Install CARLA PythonAPI: `pip install carla==0.9.4`
- Copy three files in `~/COLA/macad_gym` to `~/macad-gym/src/macad_gym/carla` and replace original files in it.

## Usage

### A3C Training

```sh
python async_a2c_agent.py --env Carla-v0 --model-dir ./trained_models/YOUR_MODEL/ --gpu-id 0
```
### A3C Testing

```sh
python async_a2c_agent.py --env Carla-v0 --model-dir ./trained_models/YOUR_MODEL/ --test
```
### Classifier Training
```sh
python COLA_rl_agent.py --env Carla-v0 --gpu-id 0
```

### Gradient Buffer Collecting
```sh
python gradient_COLA_rl_agent.py --env Carla-v0 --gpu-id 0
```

### COLA Executing
```sh
python COLA_gradient_agent.py --env Carla-v0 --test --gpu-id 0
```

The gradient buffer direction could be modified by line 94 in `gradient_COLA_rl_agent.py`.