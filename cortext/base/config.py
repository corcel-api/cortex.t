import bittensor as bt
from argparse import ArgumentParser


def add_common_config(parser: ArgumentParser):
    parser.add_argument("--netuid", type=int, default=1, help="The chain subnet uid.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    return parser
