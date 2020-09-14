#!/usr/bin/python3
from argparse import ArgumentParser
from wsl.run import train, wild, ood


def main():
    '''This is the main entrypoint for the entire project.

    This function is called whenever the wsl command is run after the package
    has been installed. This behavior is configured in the repository's setup.py.

    It sets up parsers to parse the command line arguments and then delegates to
    the appropriate function based on the provided subcommand.

    '''
    # Create the top level argument parser for the cspine_detect script
    top_parser = ArgumentParser(
        prog='wsl',
        description='Main entrypoint for the wsl project'
    )
    subparsers = top_parser.add_subparsers(help='Sub-command')

    # Add parsers for sub-commands one by one

    # Train - the main training routine
    train_parser = subparsers.add_parser('train', help='Train a model')

    # Type of dataset
    train_parser.add_argument('--debug', action='store_true',
                              help='In debugging mode, runs for just 10 sample images.')
    train_parser.add_argument('--data', type=str, default='rsna',
                              help='Type of dataset')
    train_parser.add_argument('--col_name', type=str, default='Pneumonia',
                              help='Name of the column that contains ground truth in info.csv')
    train_parser.add_argument('--extension', type=str, default='dcm')
    train_parser.add_argument('--classes', type=int, default=1)

    # Type of model
    train_parser.add_argument('--network', type=str, default='densenet',
                              help='Choose - densenet/resnet/vgg')
    train_parser.add_argument('--depth', type=int, default=121,
                              help='Model depth')
    train_parser.add_argument('--wildcat', action='store_true',
                              help='Add wildcat layers to network')
    train_parser.add_argument('--pretrained', action='store_true',
                              help='Use pretrianed network')
    train_parser.add_argument('--optim', type=str, default='adam',
                              help='Choose - sgd/adam')

    # For resuming model
    train_parser.add_argument('--resume', action='store_true',
                              help='Resume network')
    train_parser.add_argument('--name', type=str,
                              help='Model name to resume')

    # General parameters
    train_parser.add_argument('--lr', type=float, default=1e-6)
    train_parser.add_argument('--batchsize', type=int, default=64)
    train_parser.add_argument('--workers', type=int, default=4)
    train_parser.add_argument('--patience', type=int, default=10)
    train_parser.add_argument('--balanced', action='store_true')

    # Wildcat parameters
    train_parser.add_argument('--maps', default=1, type=int,
                              help='maps per class')
    train_parser.add_argument('--alpha', default=0.0, type=float,
                              help='Global Average Pooling layer weight')
    train_parser.add_argument('--k', default=1, type=float,
                              help='local pixels choosen')

    # Regression Parameters
    train_parser.add_argument('--regression', action='store_true')
    train_parser.add_argument('--error_range', default=4, type=int,
                              help='absolute error allowed')

    train_parser.set_defaults(func=train.main)

    # Wild - the main wild map calculating routine
    wild_parser = subparsers.add_parser('wild', help='Summarize all wild maps results for models')
    wild_parser.add_argument('--store', action='store_true')
    wild_parser.set_defaults(func=wild.main)

    # OOD - the main out of order distribution testing routine
    ood_parser = subparsers.add_parser('ood', help='Out of order distribution')

    # Arguments
    ood_parser.add_argument('--out_data', type=str, default='chexpert',
                            help='Name of the out distribution')
    ood_parser.set_defaults(func=ood.main)

    # Run the parsers
    args = top_parser.parse_args()
    if hasattr(args, 'func'):
        # Need to remove the function itself from the arguments
        # to prevent it being passed to itself
        func = args.func
        dict_args = vars(args)
        del dict_args['func']

        # Call to the relevant sub function
        func(**dict_args)
    else:
        # If no sub-command was specified, print the top level help
        top_parser.print_help()
