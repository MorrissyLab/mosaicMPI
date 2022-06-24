import os
import logging
import shutil
import click
import cnmf
import sys
import warnings
from collections import OrderedDict
from typing import Optional, Mapping
from cnmfsns.containers import CnmfResult, Integration
from cnmfsns.config import Config

def start_logging(output_dir):
    logging.captureWarnings(True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "logfile.txt"), mode="a"),
            logging.StreamHandler()
        ]
    )
    return

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
    pass

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
@click.option('-o', '--output_dir', type=click.Path(file_okay=False), required=True, help="Output directory for cNMF-SNS")
@click.option('-c', '--config_file', type=click.Path(exists=True, dir_okay=False), help="TOML config file")
@click.option('-i', '--input_h5mu', type=click.Path(exists=True, dir_okay=False), multiple=True, help="h5mu input file")
def initialize(output_dir, config_file, input_h5mu):
    """
    Initiate a new integration by creating a working directory with plots to assist with parameter selection.
    Although -i can be used multiple times to add .h5mu files directly, it is recommended to use a .toml file which allows for full customization.
    Using the .toml configuration file, datasets can be giving aliases and colors for use in downstream plots.
    """
    start_logging(output_dir)
    logging.info("cnmfsns initialize")
    
    if config_file is not None and input_h5mu:
        raise ValueError("A TOML config file can be specified, or 1 or more h5mu files can be specified, but not both.")
    if not all(fn.endswith(".h5mu") for fn in input_h5mu):
        raise ValueError("Input files must be h5mu mudata files.")
    
    # create directory structure, warn if overwriting
    output_dir = os.path.normpath(output_dir)
    if os.path.isdir(output_dir):
        warnings.warn(f"Integration directory {output_dir} already exists. Files may be overwritten.")
    os.makedirs(os.path.join(output_dir, "input", "datasets"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "correlation_distributions"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "output", "overdispersed_genes"), exist_ok=True)
    
    # create config
    if config_file is not None:
        config = Config.from_toml(config_file)
    elif input_h5mu:
        config = Config.from_h5mu_files(input_h5mu)
    logging.info("Copying data to output directory...")
    # copy files to output directory
    for name, d in config.datasets.items():
        shutil.copy(d["filename"], os.path.join(output_dir, "input", "datasets", name + ".h5mu"))
        shutil.copy(d["metadata"], os.path.join(output_dir, "input", "datasets", name + ".metadata.txt"))

    # add missing colors to config
    config.add_missing_colors()

    # test if variables already exist for SNS steps, otherwise create defaults that can be edited
    pass

    # save config file to output directory
    config.to_toml(os.path.join(output_dir, "config.toml"))

    # Import data from h5mu files
    data_pearson = Integration(config, output_dir, corr_method="pearson", min_corr=-1)
    data_spearman = Integration(config, output_dir, corr_method="spearman", min_corr=-1)

    # overdispersed gene UpSet plots
    data_pearson.plot_genelist_upset()

    # Create plot of correlation distributions for pearson and spearman
    data_pearson.plot_pairwise_corr(show_threshold=False)
    data_spearman.plot_pairwise_corr(show_threshold=False)

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