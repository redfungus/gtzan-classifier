# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.154.0/containers/python-3/.devcontainer/base.Dockerfile

# Set Docker arguments
ARG VARIANT

# [Choice] Python version: 3, 3.9, 3.8, 3.7, 3.6
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}

# Global git ignore (temp files & OS X shit)
RUN echo '*~' >> ~/.gitignore_global && echo .DS_Store >> ~/.gitignore_global && git config --global core.excludesfile ~/.gitignore_global

RUN apt-get update

RUN apt-get -y install libsndfile1

# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# [Optional] If your pip requirements rarely change, uncomment this section to add them to the image.
COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
   && rm -rf /tmp/pip-tmp

# Install pycodestyle
RUN pip3 --disable-pip-version-check --no-cache-dir install pycodestyle
