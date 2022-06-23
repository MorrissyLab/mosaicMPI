import os
import shutil
import click
import cnmf
import warnings
from collections import OrderedDict
from typing import Optional, Mapping
from cnmfsns.containers import CnmfResult
from cnmfsns.config import Config

class OrderedGroup(click.Group):
    """
    Overwrites Groups in click to allow ordered commands.
    """
    def __init__(self, name: Optional[str] = None, commands: Optional[Mapping[str, click.Command]] = None, **kwargs):
        super(OrderedGroup, self).__init__(name, commands, **kwargs)
        #: the registered subcommands by their exported names.
        self.commands = commands or OrderedDict()

    def list_commands(self, ctx: click.Context) -> Mapping[str, click.Command]:
        return self.commands


@click.group(cls=OrderedGroup)
def cli():
    pass

@click.command()
def inspect_inputs():
    print("Inspect_inputs")

@click.command()
def select_genes():
    pass

@click.command()
def factorize():
    pass

@click.command()
def postprocess():
    pass

@click.command()
def annotate_usages():
    pass

@click.command()
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="cNMF result directory")
@click.option('-c', '--config_file', type=click.Path(exists=True, dir_okay=False), help="TOML config file")
@click.option('-i', '--input_h5mu', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5mu input file")
def initialize(output_dir, config_file, input_h5mu):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5mu files directly, it is recommended to use a .toml file which allows for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    if config_file is not None and input_h5mu:
        raise RuntimeError("A TOML config file can be specified, or 1 or more h5mu files can be specified, but not both.")
    if not all(fn.endswith(".h5mu") for fn in input_h5mu):
        raise RuntimeError("Input files must be h5mu mudata files.")
    
    
    # create directory structure, warn if overwriting
    output_dir = os.path.normpath(output_dir)
    if os.path.isdir(output_dir):
        warnings.warn(f"Integration directory {output_dir} already exists. Files may be overwritten.")
    os.makedirs(os.path.join(output_dir, "input", "datasets"), exist_ok=True)
    
    # create config
    if config_file is not None:
        config = Config.from_toml(config_file)
    elif input_h5mu:
        config = Config.from_h5mu_files(input_h5mu)

    # # copy files to output directory
    # for dataset in config.datasets:
    #     shutil.copy(dataset["filename"], os.path.join(output_dir, "input", "datasets", dataset["alias"] + ".h5mu"))
    #     shutil.copy(dataset["metadata"], os.path.join(output_dir, "input", "datasets", dataset["alias"] + ".metadata.txt"))

    # add missing colors to config
    config.add_missing_colors()

    # test if variables already exist for SNS steps, otherwise create defaults that can be edited
    pass

    # save to output directory

    config.to_toml(os.path.join(output_dir, "config.toml"))

    # UpSet plot of OD Genes
    

    # Create plot of correlation distributions for pearson and spearman



@click.command()
def create_sns():
    pass

@click.command()
def annotate_sns():
    pass

@click.command()
@click.option('-d', '--cnmf_result_dir', type=click.Path(exists=True, file_okay=False), required=True, help="cNMF result directory")
@click.option('-l', '--local_density_threshold', default=None, type=float, show_default=True,
              help='Choose this local density threshold from those that were used to run cNMF. If unspecified, it is inferred from filenames.')
@click.option('-o', '--output_file', type=click.Path(exists=False, dir_okay=False), default=None, help="Path to output file ending with .h5mu")
@click.option('--delete', is_flag=True, help='Delete cNMF result directory after completion.')
def create_h5mu(cnmf_result_dir, local_density_threshold, output_file, delete):
    cnmf_result_dir = os.path.normpath(cnmf_result_dir)
    if output_file is None:
        output_file = os.path.normpath(cnmf_result_dir) + ".h5mu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CnmfResult.from_dir(cnmf_result_dir, local_density_threshold).to_mudata().write(output_file)
    if delete:
        shutil.rmtree(cnmf_result_dir)

cli.add_command(inspect_inputs)
cli.add_command(select_genes)
cli.add_command(factorize)
cli.add_command(postprocess)
cli.add_command(annotate_usages)
cli.add_command(initialize)
cli.add_command(create_sns)
cli.add_command(annotate_sns)
cli.add_command(create_h5mu)


if __name__ == "__main__":
    cli()