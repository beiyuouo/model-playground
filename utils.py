import ezkfg as ez


def load_config(config_path: str = "config.yaml"):
    cfg = ez.load(config_path)
    return cfg
