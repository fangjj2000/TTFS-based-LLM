# TCLLM
Official PyTorch implement for **ACHIEVE LATENCY-EFFICIENT TEMPORA-CODING
SPIKING LLMS VIA DISCRETIZATION-AWARE CONVERSION**. 



## Highlights	
Support discrete conversion for Llama models, into Spiking LLMs.

## Contents
- [Installation](#Installation)
- [Quantization](#quantization)
- [Inference](#Inference)
- [Citation](#citation)


## Installation
1.  Install miniconda3, and create virtual environment with python version equals 3.10.18
2. Activate the created virtual environment, and install poetry with `pip install poetry`.
3. Then `cd` into the root of this project and install project requirements with `poetry install --no-root`.
4. clone the repository [**Fast Hadamard Transform** ](https://github.com/Dao-AILab/fast-hadamard-transform.git), then place the toml file "pyproject for fast-hadamard.toml" into the root of fast-hadamard-transform, rename this toml file into "pyproject.toml". Finally cd into the root directory of fast-hadamard-transform, and install Fast Hadamard Transform with command  `pip install -e . --no-build-isolation`.

## Quantization

Prepare quantization model with guides form repository [**PrefixQuant**](https://github.com/ChenMnZ/PrefixQuant.git):



## Evaluation

Configure and run the `my_eval*.sh` scripts, like:
```
gpu=0,1 bash my_eval*.sh
```



## Shared Logs

We share the logs produced by our experiments, in this shared google drive [**link**](https://drive.google.com/file/d/1dty4Pah4vpD9w9N-Obxtrhwhwxsds44D/view?usp=sharing).



## Citation

Please cite this work if it help with your project or research.
```
To be Added.
```
