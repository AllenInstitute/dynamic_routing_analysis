# **dynamic_routing_analysis**

## installation
1. clone the dynamic_routing_analysis repo
2. create a new environment with python=3.11.5
   ```bash
   conda create -n dr_analysis python=3.11.5
   ```
4. navigate to the folder where you cloned the repo
5. activate your new environment
   ```bash
   conda activate dr_analysis
   ```
6. Optional: if you want to plot on the CCF, install allensdk (not necessary for other analyses; workaround to override some of allensdk's deps with dynamic_routing_analysis deps)
   ```bash
   pip install allensdk
   ```
7. install remaining dependencies from requirements.txt file
   ```bash
   pip install -r requirements.txt
   ```
8. install dynamic_routing_analysis in editable mode
   ```bash
   pip install -e .
   ```


## contributing

### **first-time capsule use**

1. duplicate the dev capsule https://codeocean.allenneuraldynamics.org/capsule/3127916
2. attach AWS and GitHub credentials (requires an access token)
3. fire up VSCode
4. pull from origin/main, activate and update the dev venv outside of conda:
    ```bash
    conda deactivate
    source install.sh
    ```
4. verify that the venv is activated in VSCode

    - `Ctrl+Shift+P` and start typing any part of `Python: Select Interpreter`
    - the interpreter should be set to `Python 3.9.12 ('.venv': venv) ./.venv/bin/python`
    - if that's not an option, hit the refresh button 
    - if it still doesn't appear, hit `Enter interpreter path...` and enter `./.venv/bin/python`
5. verify that the venv is activated in a new terminal ```[Ctrl+Shift+`]```

    the folder name and Python version should be indicated on the command line (e.g. `dra-3.9`):
    ```shell
    (dra-3.9) root@c5876abdc7b5:/code/dra# |
    ```
    
### **capsule re-use**
1. Make sure to check for updates in the source control tab in VSCode `[Ctrl+Shift+G]` and pull where appropriate
2. Update the venv in a terminal with 
    ```shell
    pdm update
    ```

***
### **adding/removing dependencies**

* when adding, this method will find a compatible version of the dependency, based on the package's Python version requirement and other existing dependencies

* the dependency will be added/removed from:
    * [pyproject.toml](pyproject.toml), which specifies required dependencies
    * the currently-activated dev venv
        * if removed, all of its sub-dependencies will also be removed
    * [pdm.lock](pdm.lock), which specifies the dev venv
        - commit any changes to the `pdm.lock` to signal updates to the common dev venv
        - if you're unsure if your lock file is correct, run `pdm update` - this will add add the most up-to-date versions of dependencies specifed in [pyproject.toml](pyproject.toml) 
        - if your venv is broken, delete the `.venv` folder and [re-install](scripts/install.sh)

For dependencies of the package itself (ie. needed for code within `dynamic_routing_analysis`):

```shell
pdm add numpy pandas
pdm remove numpy pandas
```

For dependencies needed for development of the package (ie. testing, linting, formatting, static type-analysis):

```shell
pdm add -G dev mypy pytest
pdm remove -G dev mypy pytest
```

### **updating dependencies**

If we specify only a lower bound on a dependency (e.g. `pandas >= 2.0`), any new install of `dynamic_routing_analysis` will also install the latest version of `pandas`.

To make sure that the latest versions of dependencies don't introduce breaking changes, we should update the dev venv periodically by running `pdm update` and running any tests, then committing [pdm.lock](pdm.lock).

More info on `pdm` and the lock file: https://pdm.fming.dev/latest/usage/dependency/#install-the-packages-pinned-in-lock-file
