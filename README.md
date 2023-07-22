# **dynamic_routing_analysis**


## **quickstart**

### **first-time capsule use**

1. clone the dev capsule https://codeocean.allenneuraldynamics.org/capsule/3127916
2. fire up VSCode
3. pull the latest changes and update the dev venv:

    `linux`
    ```bash
    source scripts/update.sh
    ```

    `windows`
    ```cmd
    scripts\update.bat
    ```
4. verify that the venv is activated in VSCode

    - `Ctrl+Shift+P` and start typing `Select Interpreter`, then select `Python: Select Interpreter`
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
## Add/remove dependencies

* when adding, this method will find a compatible version of the dependency, based on the package's Python version requirement and other existing dependencies

* the dependency will be added/removed from dependencies in [pyproject.toml](pyproject.toml) 

* the dependency will be added/removed from the currently-activated venv

* if removed, all of the sub-dependencies will also be removed from the venv

* multiple dependencies can be specified together

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
