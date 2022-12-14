import nox


@nox.session
def tests(session):
    session.install("pip")
    session.run("pip", "install", "numpy")  # required to install matscipy
    session.run("python", "setup.py", "develop")
    session.run("python", "-m", "mace_jax.run_train", "configs/aspirin_small.gin")
