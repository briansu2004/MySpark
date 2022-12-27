# PySpark + Python + Jypyter Notebook + VSCode

VSCode has built-in support on Jypyter Notebook.

## No Conda

### MacOS

- Python

```bash
me ~ % brew install python
Warning: python@3.10 3.10.9 is already installed and up-to-date.
To reinstall 3.10.9, run:
  brew reinstall python@3.10
me ~ % brew link python@3.10
Warning: Already linked: /usr/local/Cellar/python@3.10/3.10.9
To relink, run:
  brew unlink python@3.10 && brew link python@3.10
me ~ % brew link --overwrite python@3.10
Warning: Already linked: /usr/local/Cellar/python@3.10/3.10.9
To relink, run:
  brew unlink python@3.10 && brew link python@3.10
me ~ % ls -l /usr/local/bin/2*
lrwxr-xr-x  1 x239757  admin  37 26 Dec 19:55 /usr/local/bin/2to3 -> ../Cellar/python@3.10/3.10.9/bin/2to3
lrwxr-xr-x  1 x239757  admin  42 26 Dec 19:55 /usr/local/bin/2to3-3.10 -> ../Cellar/python@3.10/3.10.9/bin/2to3-3.10
lrwxr-xr-x  1 root     wheel  70 11 Aug 15:26 /usr/local/bin/2to3-3.7 -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/2to3-3.7
me ~ % 
```

- VSCode

- VSCode Python extension

- VSCode "Python: Select Interpreter"

- VSCode "Create: New Jupyter Notebook"

### Windows

...

## PySpark + Python + Jypyter Notebook + VSCode + Conda

Not needws anymore.

Sometimes not working!

### A

- Anaconda

- Anaconda Prompt

```dos
conda install ipykernel
conda install jupyter
cd\<MyFolder>
jupyter-notebook
```

![1672097489350](image/PySparkNotebook/1672097489350.png)

### B

- Anaconda

- Anaconda Prompt

```dos
conda install ipykernel
```

![1672086486743](image/PySparkNotebook/1672086486743.png)

- VSCode
- Install Python extension
- Python: Select Interpreter

![1672086607385](image/PySparkNotebook/1672086607385.png)

- Trust
- Create: New Jupyter Notebook
- Firewall
- Notebook

### C

1. Anaconda

```dos
conda config --set channel_priority strict
conda env create -f golden_scenario_env.yml
conda info --envs
conda activate golden_scenario_env
conda install python=3.7
```

```dos
python -m ipykernel install --user --name golden_scenario_env --display-name "Golden Scenario Env"
```

- VSCode

## PySpark + Python + Jypyter Notebook + Docker

```dos
docker pull jupyter/datascience-notebook
docker images
docker run --rm -t -d --name=jupyter -p 8888:8888 <imageId>
docker run --rm -t -d --name=jupyter -p 8888:8888 --mount src="$(pwd)",target=/app,type=bind jupter_v2:latest
```

`docker run --rm -t -d --name=jupyter -p 8888:8888 --mount src=C:\Code\MySpark\PySpark-Notebook-simple,target=/app,type=bind jupter_v2:latest`
