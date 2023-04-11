import nox


@nox.session(name="test")
def run_tests(session):
    """Run Pytests."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8==4.0.1")
    session.install(
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.run("flake8", "src", "tests", "noxfile.py")


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "tests", "noxfile.py")
    session.run("black", "src", "tests", "noxfile.py")
