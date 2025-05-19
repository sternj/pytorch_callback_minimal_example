from setuptools import setup
from setuptools.command.build_ext import build_ext
from pathlib import Path

from setuptools import Extension, setup
import os
import subprocess
import shutil
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())
class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension):
        # print("NAME", ext.name)
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        # print("FULLPATH", ext_fullpath)
        extdir = ext_fullpath.parent.resolve()
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPython_EXECUTABLE={self.get_executable()}",
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DCMAKE_BUILD_TYPE=DEBUG",
            "-GNinja",
            '-DCMAKE_CXX_COMPILER=/usr/bin/clang++-18',
            '-DCMAKE_C_COMPILER=/usr/bin/clang-18'
        ]
        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)
        subprocess.check_call(["cmake",  ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(["cmake", "--build", "."], cwd=build_temp)
        compile_commands = os.path.join(build_temp, "compile_commands.json")
        if os.path.exists(compile_commands):
            shutil.copy(compile_commands, os.path.join(os.getcwd(), '.vscode', "compile_commands.json"))

        # print("TEMPTEMP", os.listdir(Path(self.build_temp) / ext.name))
    # def get_source_dir(self):
    #     return os.path.abspath(os.path.dirname(__file__))

    def get_executable(self):
        import sys
        return sys.executable

setup(
    name="cb_test",
    version="0.1.0",
    packages=["cb_test"],
    package_dir={"cb_test": "src/python/cb_test"},
    package_data={"cb_test._C": ["*.so"]},
    ext_modules=[CMakeExtension("cb_test._C", sourcedir="src/cpp")],
    cmdclass={"build_ext": CMakeBuild},
)