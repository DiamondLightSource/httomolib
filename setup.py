from setuptools import Extension, setup, find_packages
from Cython.Build import cythonize
from distutils.command.build import build as build_orig

import numpy
import platform

if platform.system() == "Windows":
    extra_compile_args = ["/DWIN32", "/EHsc", "/DBOOST_ALL_NO_LIB", "/openmp"]
else:
    extra_compile_args = ["-fopenmp", "-O2", "-funsigned-char", "-Wall"]


exts = [
    Extension(
        name="httomolib.core.modules",
        sources=[
            "httomolib/core/rescale_to_int.c",
            "httomolib/core/modules.pyx",
        ],
        include_dirs=["httomolib/core"],
        extra_compile_args=extra_compile_args,
        extra_link_args=["-lgomp"],
    ),
]


class build(build_orig):

    def finalize_options(self):
        super().finalize_options()
        for extension in self.distribution.ext_modules:
            extension.include_dirs.append(numpy.get_include())
        self.distribution.ext_modules = cythonize(
            self.distribution.ext_modules, language_level=3
        )


setup(
    name="httomolib",
    ext_modules=exts,
    packages=find_packages(),
    setup_requires=["cython", "numpy"],
    install_requires=["numpy"],
    zip_safe=False,
    include_package_data=True,
    cmdclass={"build": build},
)
