#!/usr/bin/env python
# coding: utf-8

# (How_to_Use_This_Book)=
# # How to use this book
# 
# 
# This book is a [Jupyter Book](https://jupyterbook.org/intro.html), which allows you to interactively execute, reproduce, or modify the sections of this book that contain code. This chapter provides an overview of what Jupyter books are, and how they allow their content to be reproducible. 
# 
# 
# ## Jupyter book project
# 
# The Jupyter Book project is an open-source project. It is developed by the [Executable Book Project](https://executablebooks.org/en/latest/), which is an [international collaboration between several universities and other open-source projects](https://executablebooks.org/en/latest/about.html). 
# 
# Under the hood, a Jupyter Book is essentially a collection of markdown files and Jupyter notebooks. These files are compiled together according to the configuration and table of content specifications. You can follow the Github icon at the top of this page to see the source code for this book. 
# 
# 
# ## Book layout
# 
# The book layout provides five main zones:
# 
# 1. The search tool (top left): It allows finding all sections of the book that contains a particular term.
# 2. The table of content (left panel): It allows navigating between the different chapters and main sections of the book.
# 3. The external links (top): It contains icons for launching notebooks, switching to full-screen mode, accessing the Github repository, or downloading the content of the current section.
# 4. The main panel (center): The content of the current section.
# 5. The section content (Right panel): It allows navigating in the subsections of the current section. 
# 
# 
# 
# 
# ## Reproducibility
# 
# All sections that contain code (Jupyter notebooks), such as this section, will have a shuttle icon in the external links, allowing to run the notebook on the cloud. Two services are provided: Google Colab, and Binder. We briefly describe these two services below, with their pros and cons. 
# 
# ### Google Colab
# 
# 
# [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb#) is a service provided by Google Research and requires to have a Google account. It allows to run Python notebooks for free on Google cloud. 
# 
# The computational resources provided by Google Colab can fluctuate. They depend on available Google resources, as well as your usage of the service (intense use may lead to lower resources). You may however expect to have one core, and a few gigabytes of RAM. Google Colab also provides a free GPU, whose model may vary depending on available resources. Provided GPU models often include Nvidia K80s, T4s, P4s, or P100s. See this [link](https://research.google.com/colaboratory/faq.html?hl=ko-KRUbuntu) for more details on the resources that Google can provide. 
# 
# ### Binder
# 
# [Binder](https://mybinder.readthedocs.io/en/latest/index.html) {cite}`beg2021using` allows to create, use and share custom computing environments. It is powered by [BinderHub](https://github.com/jupyterhub/binderhub), which is an open-source tool that deploys the Binder service in the cloud. 
# 
# Notebooks may be launched and run for free, thanks to the [BinderHub Federation](https://mybinder.readthedocs.io/en/latest/about/federation.html), currently supported by Google Cloud Platform, OVH, GESIS notebooks, and the Alan Turing Institute. BinderHub relies on [Repo2Docker](https://github.com/jupyterhub/repo2docker) to generate docker images from a Git repository, and [JupyterHub](https://github.com/jupyterhub/jupyterhub) to run and share notebooks. 
# 
# It takes usually around one or two minutes for Binder to create a Docker container and start serving a notebook. The `environment.yml` file, present at the root of the Github book repository, is used to specify which Python libraries must be installed in the docker container. 
# 
# As with Google Colab, the resources provided for free by the BinderHub Federation fluctuate. See the [Binder user guide](https://mybinder.readthedocs.io/en/latest/about/about.html) and [Binder user guidelines](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html) for more details on the resources that the BinderHub Federation can provide. 
# 
# 
# ### Pros and cons
# 
# A summary of the features of Google Colab and Binder is provided below. 
# 
# 
# || Google Colab | Binder|
# :----- | :---- | :-----
# |Registration   | Requires a Google login | None|
# |CPU resources | Fluctuate  | Fluctuate|
# |RAM resources | Fluctuate  | Fluctuate|
# |GPU resources | Fluctuate. Often includes Nvidia K80s, T4s, P4s, or P100s  | None|
# |Session duration  | Up to 12 hours  | Up to 12 hours|
# |Environment  | Python 3.6 | Customizable with environment.yml file|
# 
# In our experience, Google Colab execution times are about twice as fast as Binder. 
# 
# This summary reflects the status of the two services at the time of the writing of this section (January 2021). The features of these two services are however expected to evolve in the short future.  
# 
# 
# 
# ### Try it!
# 
# Open this notebook using either Google Colab or Binder to execute the following cell:

# In[2]:


print('Hello world!')


# ## Local execution and book compilation
# 
# This is an open-source book, and the code is made [available on GitHub](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook) (or by following the GitHub icon in the external links at the top of this page). You may run the book notebooks on your computer by cloning the GitHub repository. 
# 
# You may also recompile the book. Compiling the book will first require you to [install Jupyter Book](https://jupyterbook.org/intro.html#install-jupyter-book).
# 
# Once done, this is a two-step process:
# 
# 1. Clone the book repository:
# 
# ```
# git clone https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook
# ```
# 
# 2. Compile the book
# 
# ```
# jupyter-book build fraud-detection-handbook
# ```
# 
# The book will be available locally at `fraud-detection-handbook/_build/html/index.html`.
# 
# 

# In[ ]:




