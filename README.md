# COLA

## Source code for *Self-Adaptive Driving in Nonstationary Environments through  Conjectural Online Lookahead Adaptation*
A3C model is original implementated by [Palanisamy](https://github.com/PacktPublishing/Hands-On-Intelligent-Agents-with-OpenAI-Gym) in Chapter 8, the structure of classifer is based on [Rashi Sharma](https://medium.com/swlh/natural-image-classification-using-resnet9-model-6f9dc924cd6d). Both models are deployed in [`deep.py`](function_approximator/deep.py).

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-2-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
	- [Contributors](#contributors)
- [License](#license)

## Background
<img src="https://github.com/Panshark/COLA/blob/main/COLA.png"/>

Powered by deep representation learning, reinforcement learning (RL) provides an end-to-end learning framework capable of solving self-driving (SD) tasks without manual designs. However, time-varying nonstationary environments cause proficient but specialized RL policies to fail at execution time. For example, an RL-based SD policy trained under sunny days does not generalize well to the rainy weather. Even though meta learning enables the RL agent to adapt to new tasks/environments in a sample-efficient way, its offline operation fails to equip the agent with online adaptation ability when facing nonstationary environments. This work proposes an online meta reinforcement learning algorithm based on the **conjectural online lookahead adaptation** (COLA). COLA determines the online adaptation at every step by maximizing the agent's conjecture of the future performance in a lookahead horizon.  Experimental results demonstrate that under dynamically changing weather and lighting conditions, the COLA-based self-adaptive driving outperforms the baseline policies in terms of online adaptability.

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

The gradient buffer directory could be modified by line 94 in `gradient_COLA_rl_agent.py`. Modify the `environment/carla_gym/config.json` to set `"dynamic_on": false`. And modify line 151 in `~/macad-gym/src/macad_gym/carla/scenarios.py` for collecting gradients from cloudy (1) and rainy (4). Then change back the `dynamic_on` botton. You can do the COLA Executing now.

## Maintainers

[@Haozhe Lei](https://github.com/Panshark).

## Contributing

Feel free to dive in! [Open an issue](https://github.com/Panshark/Attack_metaRL/issues/new) or submit PRs.

Standard Readme follows the [Contributor Covenant](http://contributor-covenant.org/version/1/3/0/) Code of Conduct.

### Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Panshark"><img src="https://avatars.githubusercontent.com/u/71244619?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Haozhe Lei</b></sub></a><br /><a href="https://github.com/Panshark/Attack_metaRL/commits?author=Panshark" title="Code">💻</a> <a href="#data-Panshark" title="Data">🔣</a> <a href="https://github.com/Panshark/Attack_metaRL/commits?author=Panshark" title="Documentation">📖</a> <a href="#ideas-Panshark" title="Ideas, Planning, & Feedback">🤔</a> <a href="#maintenance-Panshark" title="Maintenance">🚧</a> <a href="#projectManagement-Panshark" title="Project Management">📆</a> <a href="#question-Panshark" title="Answering Questions">💬</a> <a href="https://github.com/Panshark/Attack_metaRL/pulls?q=is%3Apr+reviewed-by%3APanshark" title="Reviewed Pull Requests">👀</a> <a href="#design-Panshark" title="Design">🎨</a></td>
    <td align="center"><a href="https://engineering.nyu.edu/student/tao-li-0"><img src="https://avatars.githubusercontent.com/u/46550706?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Tao Li</b></sub></a><br /><a href="#design-TaoLi-NYU" title="Design">🎨</a> <a href="#eventOrganizing-TaoLi-NYU" title="Event Organizing">📋</a> <a href="#ideas-TaoLi-NYU" title="Ideas, Planning, & Feedback">🤔</a> <a href="#data-TaoLi-NYU" title="Data">🔣</a> <a href="#content-TaoLi-NYU" title="Content">🖋</a> <a href="#question-TaoLi-NYU" title="Answering Questions">💬</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Panshark/COLA/blob/main/LICENCE) [MIT](https://github.com/Panshark/COLA/blob/main/LICENCE) © Haozhe Lei
