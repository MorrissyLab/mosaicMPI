import os
import click
import cnmf
from collections import OrderedDict
from typing import Optional, Mapping
from cnmfsns.containers import CnmfResult

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
def explore_thresholds():
    pass

@click.command()
def integrate():
    pass

@click.command()
def annotate_sns():
    pass

@click.command()
@click.option('-o', '--cnmf_result_dir', type=click.Path(exists=True), required=True, help="cNMF result directory")
@click.option('-l', '--local_density_threshold', default=None, type=float, show_default=True, help='local density threshold for consensus')
def import_from_cnmf(cnmf_result_dir, local_density_threshold):
    cnmf_result_dir = os.path.normpath(cnmf_result_dir)
    if not os.path.isdir(cnmf_result_dir):
        raise IOError(f"{cnmf_result_dir} is not a valid directory.")
    outfile = os.path.join(os.path.join(os.path.normpath(cnmf_result_dir), ".h5ad"))
    CnmfResult.from_dir(cnmf_result_dir, local_density_threshold).to_h5ad(outfile)

cli.add_command(inspect_inputs)
cli.add_command(select_genes)
cli.add_command(factorize)
cli.add_command(postprocess)
cli.add_command(annotate_usages)
cli.add_command(explore_thresholds)
cli.add_command(integrate)
cli.add_command(annotate_sns)
cli.add_command(import_from_cnmf)


if __name__ == "__main__":
    cli()