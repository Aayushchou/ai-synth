import yaml


def read_yaml(filename: str):
    with open(filename, "r") as stream:
        return yaml.safe_load(stream)
