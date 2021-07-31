### Study Notes

#### July 31, 2021
##### Reference 
* [How to Build Your Very First Python Package](https://www.freecodecamp.org/news/build-your-first-python-package/)

  * In directory *Python Packaging*, a simple python project is provided to show the basic structure and metadata to package a python application.
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
    This will build all the necessary packages that Python will require. The *sdist* and *bdist_wheel* commands will create a source distribution and a wheel that you can later upload to *PyPi*.