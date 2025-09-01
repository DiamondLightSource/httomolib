from setuptools import Extension, setup
import numpy
import platform

if platform.system() == "Windows":
    extra_compile_args = ["/DWIN32", "/EHsc", "/DBOOST_ALL_NO_LIB", "/openmp"]
else:
    extra_compile_args = ["-fopenmp", "-O2", "-funsigned-char", "-Wall"]

extra_include_dirs = [numpy.get_include()]

ext_modules = [
    Extension(
        "httomolib.core.modules",
        sources=[
            "httomolib/core/modules.pyx",
            "httomolib/core/rescale_to_int.c",
        ],
        include_dirs=extra_include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lgomp"],
    ),
]


if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(
        name="httomolib",
        ext_modules=cythonize(ext_modules, language_level="3"),
    )
