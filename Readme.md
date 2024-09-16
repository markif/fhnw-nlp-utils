This library provides utilities for NLP courses taught at FHNW.

# Install 

This package builds on Python 3.

```bash
sudo pip install fhnw-nlp-utils
```

# Build

- Increase the version number in [setup.py](setup.py).

- Start a Docker container with the build/deploy tools:

```bash
docker run -e TZ=Europe/Zurich --name datascience-notebook --net=host -p 8888:8888 -v "$(pwd)":/home/jovyan/work -v ${HOME}/.pypirc:/home/jovyan/.pypirc -it --rm i4ds/datascience-notebook start-notebook.sh --NotebookApp.token=''
```

- Build and deploy the library to pypi:


```bash
docker exec -it datascience-notebook /bin/bash

cd work
rm -rf dist/*
python3 setup.py bdist_wheel
python3 -m twine upload dist/*
```

- Upload the latest changes to the git repository:

```bash
git commit -am "message"
git push
```
