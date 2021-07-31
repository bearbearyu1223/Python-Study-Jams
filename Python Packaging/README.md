### Study Notes

#### July 31, 2021

##### Reference

* [How to Build Your Very First Python Package](https://www.freecodecamp.org/news/build-your-first-python-package/)

    * In directory *Python Packaging*, a simple python project is provided to show the basic structure and metadata to
      package a python application.
      ```
      |-- base_verysimplemodule
      |   |-- verysimplemodule
      |       |-- __init__.py
      |       |-- extras
      |          |-- __init__.py
      |          |-- multiply.py
      |          |-- divide.py 
      |       |-- add.py
      |       |-- substract.py
      |-- setup.py
      |-- .gitignore
      ```
    * In the same directory *base-verysimpleodule*, run the following command:
      ```shell
      python setup.py sdist bdist_wheel
      ```    
      This will build all the necessary packages that Python will require. The *sdist* and *bdist_wheel* commands will
      create a source distribution and a wheel that you can later upload to *PyPi*.
* *PyPi* is the official Python repository where all Python packages are stored. You can think of it as the Github for
  Python Packages. But before upload packages to *PyPi*, you need to install twine if you don’t already have it
  installed. It’s as simple as `pip3 install twine`.

* Assuming you have twine installed, go ahead and run `twine upload dist/*`, this will upload the contents of the dist
  folder that was automatically generated when we ran `python setup.py`.

* Check out the uploaded package at: https://pypi.org/project/verysimplemodule-0731-2021/0.0.1/.
  