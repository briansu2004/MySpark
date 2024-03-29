# PySpark + Python + Jypyter Notebook + VSCode

VSCode has built-in support on Jypyter Notebook.

## No Conda

### PySpark + Python + Jypyter Notebook + VSCode: MacOS

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

Note:

```bash
me % cd /usr/local/bin
me % ls -la | grep pip
lrwxr-xr-x    1 x239757  admin       37 26 Dec 19:55 pip3 -> ../Cellar/python@3.10/3.10.9/bin/pip3
lrwxr-xr-x    1 x239757  admin       40 26 Dec 19:55 pip3.10 -> ../Cellar/python@3.10/3.10.9/bin/pip3.10
lrwxrwxr-x    1 root     admin       68 11 Aug 15:27 pip3.7 -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/pip3.7
me % ls -la | grep python
lrwxr-xr-x    1 x239757  admin       37 26 Dec 19:55 2to3 -> ../Cellar/python@3.10/3.10.9/bin/2to3
lrwxr-xr-x    1 x239757  admin       42 26 Dec 19:55 2to3-3.10 -> ../Cellar/python@3.10/3.10.9/bin/2to3-3.10
lrwxr-xr-x    1 x239757  admin       38 26 Dec 19:55 idle3 -> ../Cellar/python@3.10/3.10.9/bin/idle3
lrwxr-xr-x    1 x239757  admin       41 26 Dec 19:55 idle3.10 -> ../Cellar/python@3.10/3.10.9/bin/idle3.10
lrwxr-xr-x    1 x239757  admin       37 26 Dec 19:55 pip3 -> ../Cellar/python@3.10/3.10.9/bin/pip3
lrwxr-xr-x    1 x239757  admin       40 26 Dec 19:55 pip3.10 -> ../Cellar/python@3.10/3.10.9/bin/pip3.10
lrwxr-xr-x    1 x239757  admin       39 26 Dec 19:55 pydoc3 -> ../Cellar/python@3.10/3.10.9/bin/pydoc3
lrwxr-xr-x    1 x239757  admin       42 26 Dec 19:55 pydoc3.10 -> ../Cellar/python@3.10/3.10.9/bin/pydoc3.10
lrwxr-xr-x    1 x239757  admin       40 26 Dec 19:55 python3 -> ../Cellar/python@3.10/3.10.9/bin/python3
lrwxr-xr-x    1 x239757  admin       47 26 Dec 19:55 python3-config -> ../Cellar/python@3.10/3.10.9/bin/python3-config
lrwxr-xr-x    1 x239757  admin       43 26 Dec 19:55 python3.10 -> ../Cellar/python@3.10/3.10.9/bin/python3.10
lrwxr-xr-x    1 x239757  admin       50 26 Dec 19:55 python3.10-config -> ../Cellar/python@3.10/3.10.9/bin/python3.10-config
lrwxr-xr-x    1 root     wheel       71 11 Aug 15:26 python3.7 -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7
lrwxr-xr-x    1 root     wheel       78 11 Aug 15:26 python3.7-config -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7-config
lrwxr-xr-x    1 root     wheel       72 11 Aug 15:26 python3.7m -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7m
lrwxr-xr-x    1 root     wheel       79 11 Aug 15:26 python3.7m-config -> ../../../Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7m-config
lrwxr-xr-x    1 x239757  admin       39 26 Dec 19:55 wheel3 -> ../Cellar/python@3.10/3.10.9/bin/wheel3
lrwxr-xr-x    1 x239757  admin       42 26 Dec 19:55 wheel3.10 -> ../Cellar/python@3.10/3.10.9/bin/wheel3.10
```

- VSCode

- VSCode Python extension

- VSCode "Python: Select Interpreter"

- VSCode "Create: New Jupyter Notebook"

### PySpark + Python + Jypyter Notebook + VSCode: Windows

#### New way

```dos
pip install jupyterlab
python -m jupyterlab
```

![1672160305152](image/PySparkNotebook/1672160305152.png)

Considering make an alias (e.g. `j`) for `python -m jupyterlab`

Can do it with venv together

```dos
python -m venv venv
venv\scripts\activate.bat
pip install ipykernel jupyter jupyterlab
python -m jupyterlab
venv\scripts\deactivate.bat
```

#### Old way

```dos
python -m pip install --upgrade pip
pip install ipykernel jupyter
cd <MyFolder>
jupyter-notebook
```

## PySpark + Python + Jypyter Notebook + VSCode + Conda

### PySpark + Python + Jypyter Notebook + VSCode + Conda : MacOS

...

### PySpark + Python + Jypyter Notebook + VSCode + Conda : Windows

May not be needed.

Sometimes not working!

#### A

- Anaconda

- Anaconda Prompt

```dos
conda install ipykernel jupyter
cd <MyFolder>
jupyter-notebook
```

![1672097489350](image/PySparkNotebook/1672097489350.png)

#### B

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

#### C

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

## PySpark installation

### PySpark installation :  MacOS

...

### PySpark installation : Windows

```dos
pip install pyspark
pyspark
```

```dos
C:\Code\MySpark>pip install pyspark
Requirement already satisfied: pyspark in c:\apps\python37\lib\site-packages (3.3.1)
Requirement already satisfied: py4j==0.10.9.5 in c:\apps\python37\lib\site-packages (from pyspark) (0.10.9.5)

C:\Code\MySpark>pyspark
Python 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019, 00:11:34) [MSC v.1916 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
22/12/27 20:03:21 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.3.1
      /_/

Using Python version 3.7.5 (tags/v3.7.5:5c02a39a0b, Oct 15 2019 00:11:34)
Spark context Web UI available at http://L113220.mshome.net:4040
Spark context available as 'sc' (master = local[*], app id = local-1672189403113).
SparkSession available as 'spark'.
>>>
```
