from src.config import load_config, ensure_dirs
from src.demo import launch


def main():
    cfg = load_config()
    ensure_dirs(cfg)
    launch(cfg)


if __name__ == '__main__':
    main()
