from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="snhlib",
    version="0.0.1-alpha.43",
    author="Shidiq Nur Hidayat",
    author_email="s.hidayat@nanosense-id.com",
    description="The SNH's Misc. Tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shidiq/snhlib.git",
    project_urls={"Bug Tracker": "https://github.com/Shidiq/snhlib/issues"},
    license="MIT",
    packages=["snhlib"],
    install_requires=[],
)
