import fire

from .play_terminal import play_against_bot


fire.Fire({"play": play_against_bot})
