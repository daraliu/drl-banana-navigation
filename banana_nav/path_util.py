import pathlib

from banana_nav import config as cfg


def mk_path_weights(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_WEIGHTS)


def mk_path_scores(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_SCORES)


def mk_path_metadata(dir_output):
    return pathlib.Path(dir_output).joinpath(cfg.FILENAME_METADATA)
