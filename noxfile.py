import nox


@nox.session
def tests(session):
    session.install("pip")
    session.run("pip", "install", ".", "-v")
    session.run("python", "scripts/run_train.py", "configs/aspirin_small.gin")
