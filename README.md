# FuRL

## Environment Setup

Install the conda env via:

```shell
conda create --name furl python==3.11
conda activate furl
conda install pytorch==2.3.1 torchvision==0.18.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
git clone https://github.com/carlosferrazza/humanoid-bench.git
cd humanoid-bench/ 
pip install -e .
pip install "jax[cuda12]"
cd ..
pip uninstall torch torchvision torchaudio
pip uninstall jax jaxlib
pip install torch torchvision torchaudio
pip install --upgrade flax jax jaxlib
```

## Training

```bash
python main.py --config.env_name=h1hand-run-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-run-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-run-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-walk-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-walk-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-walk-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-sit_hard-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-sit_hard-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-sit_hard-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-stair-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-stair-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-stair-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-stand-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-stand-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-stand-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-balance_simple-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-balance_simple-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-balance_simple-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-sit_simple-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-sit_simple-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-sit_simple-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-slide-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-slide-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-slide-customized-v0 --config.exp_name="furl" --config.seed=2

python main.py --config.env_name=h1hand-balance_hard-customized-v0 --config.exp_name="furl" --config.seed=0
python main.py --config.env_name=h1hand-balance_hard-customized-v0 --config.exp_name="furl" --config.seed=1
python main.py --config.env_name=h1hand-balance_hard-customized-v0 --config.exp_name="furl" --config.seed=2
```

## Paper

[**FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning**](https://arxiv.org/pdf/2406.00645)

Yuwei Fu, Haichao Zhang, Di Wu, Wei Xu, Benoit Boulet

*International Conference on Machine Learning* (ICML), 2024

## Cite

Please cite our work if you find it useful:

```txt
@InProceedings{fu2024,
  title = {FuRL: Visual-Language Models as Fuzzy Rewards for Reinforcement Learning},
  author = {Yuwei Fu and Haichao Zhang and Di Wu and Wei Xu and Benoit Boulet},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  year = {2024}
}
```
