# Introduction

## Special directories

We've identified 9 special directories where the project contents should go. They are all placed in the root of the project.

1. *Applications* - contains applications
2. *Configuration* - contains configuration
3. *Data* - contains non-configuration data
4. *Documentation* - contains documentation
5. *Examples* - contains examples of library usage
6. *Libraries* - contains headers and sources of libraries
7. *Scripts* - contains scripts
8. *Templates* - contains file templates
9. *Tests* - contains tests

### Project Structure

Additionally, a project needs to bind itself to other projects and it needs to be built and packaged. The following directories are also placed in the root of the project:

- *Dependencies* - place where we link to project dependencies, usually with git submodule
- *Build* - place for the development build
- *Release* - place for the release build
- *Outputs* - this directory a result of installation, a complete package

### Project creation with hey.py

You can create a project with `hey.py` script. Do the following:

```
sudo ln -s /path/to/script/hey.py /usr/bin/local/hey
mkdir MyProject
cd MyProject
git init
git remote add origin git@github.com:MyName/MyProject.git 
hey project create --all
```

This will create: 

1. *Applications* - empty directory
2. *Configuration*:
    + *MyProject*
        * *setup.yml* - configures a setup.py script
        * *project.yml* - describes the project
    + *ProjectCMake* - configuration of cmake for the project
    + *ProjectDoxygen* - configuration of doxygen
3. *Data* - empty directory
4. *Documentation* - empty directory
5. *Examples* - empty directory
6. *Libraries* - empty directory
7. *Scripts*:
    + *MyProject*
        * *setup.py* - a script for setting up the project and for development
        * *run.py* - a script for finding and running other scripts
8. *Templates* - empty directory
9. *Tests* - empty directory
10. CMakeLists.txt - sets up ProjectCMake to work in your project
11. Dockerfile - build your project with `docker build -t my_project .`
12. LICENSE.md - current default license
13. README.md - current default readme
14. .clang-format - defautl clang-format config
15. .clang-tidy - defautl clang-tidy config
16. .dockerignore - default dockerignore
17. .gitignore - default gitignore
18. .gitmodules - git uses this file to manage submodules, should contain ProjectTools
19. .travis.yml - generated travis configuration

### Outputs' structure

Is a special folder where the project installation gets deployed to.

A Continuous Integration tool pushes this folder as a standalone branch 'outputs' into the project repository.

The additional directories are:

- *Includes* - Stores declarations for the libraries. It's required for packaging, but in the Project itself, we store the declarations next to the source files.

## Configuration / ProjectCMake

**Configuration / ProjectCMake** is another directory where the project structure is adopted.
A set of defaults is defined for each concept, but those can be overridden with your own files.

You can add a `Project.cmake` file, which is considered to be the top level cmake file that gets included first.

Then, you can add the project structure cmake files, which get included later. The convention is:
- `Applications.cmake` on ENABLE_APPLICATIONS
- `Configuration.cmake` on ENABLE_CONFIGURATION
- ...
- `Tests.cmake` on ENABLE_TESTS

See [**Project structure**](#user-content-project-structure) to check all the concept names.
