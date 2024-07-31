# Installing Dependencies
```pip install -r requirements.txt```
or
```python3 -m pip install -r requirements.txt```

# Example (Using Virtual Environment):

```
python -m venv myenv    # Create a virtual environment
source myenv/bin/activate  # Activate virtual environment
# source myenv/Scripts/activate if using windows
pip install -r requirements.txt  # Install packages
```
# Update pip if necessary
```python -m pip install --upgrade pip setuptools wheel```

# if issues with PyYaml
Windows
```pip install build ```
Linux
``` 
sudo apt-get install libyaml-dev
pip install build
```
Then
```pip install PyYAML --no-binary :all:```

# for windows, you need C++
Download and install the Microsoft C++ Build Tools from the link in the error message: https://visualstudio.microsoft.com/visual-cpp-build-tools/
During installation, ensure you select the "Desktop development with C++" workload and include the "MSVC v140 - VS 2015 C++ build tools" component.

# bash
```pip install Cython --no-binary :all:```




