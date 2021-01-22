#!/usr/bin/python3
from argparse import ArgumentParser
from wsl.run import medinet, retinanet, saliency, ood


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

    # medinet - the main medinet routine
    medinet_parser = subparsers.add_parser('medinet', help='medinet a model')

    # Type of dataset
    medinet_parser.add_argument('--debug', action='store_true', help='In debugging mode, runs for just 10 sample images.')
    medinet_parser.add_argument('--data', type=str, default='rsna', help='Type of dataset')
    medinet_parser.add_argument('--column', type=str, default='Pneumonia', help='Name of the column that contains ground truth in info.csv')
    medinet_parser.add_argument('--extension', type=str, default='dcm')
    medinet_parser.add_argument('--classes', type=int, default=1)
    medinet_parser.add_argument('--augmentation', action='store_true', help='Add augmentation to data')
    # Type of model
    medinet_parser.add_argument('--network', type=str, default='densenet', help='Choose - densenet/resnet/vgg')
    medinet_parser.add_argument('--depth', type=int, default=121, help='Model depth')
    medinet_parser.add_argument('--wildcat', action='store_true', help='Add wildcat layers to network')
    medinet_parser.add_argument('--pretrained', action='store_true', help='Use pretrianed network')
    medinet_parser.add_argument('--optim', type=str, default='adam', help='Choose - sgd/adam')
    # For resuming model
    medinet_parser.add_argument('--resume', action='store_true', help='Resume network')
    medinet_parser.add_argument('--name', type=str, help='Model name to resume')
    # General parameters
    medinet_parser.add_argument('--lr', type=float, default=1e-5)
    medinet_parser.add_argument('--batchsize', type=int, default=32)
    medinet_parser.add_argument('--workers', type=int, default=4)
    medinet_parser.add_argument('--patience', type=int, default=5)
    medinet_parser.add_argument('--balanced', action='store_true')
    # Wildcat parameters
    medinet_parser.add_argument('--maps', default=1, type=int, help='maps per class')
    medinet_parser.add_argument('--alpha', default=0.0, type=float, help='Global Average Pooling layer weight')
    medinet_parser.add_argument('--k', default=1, type=float, help='local pixels choosen')
    # Regression parameters
    medinet_parser.add_argument('--regression', action='store_true')
    medinet_parser.add_argument('--error_range', default=4, type=int, help='absolute error allowed')
    # Identification parameter
    medinet_parser.add_argument('--ID', type=str, default='placeholder', help='Special ID to identify a set of models')
    medinet_parser.set_defaults(func=medinet.main)

    # retinanet - the main retinanet routine
    retinanet_parser = subparsers.add_parser('retinanet', help='retinanet a model')
    # Type of dataset
    retinanet_parser.add_argument('--debug', action='store_true', help='In debugging mode, runs for just 10 sample images.')
    retinanet_parser.add_argument('--data', type=str, default='rsna', help='Type of dataset')
    retinanet_parser.add_argument('--column', type=str, default='Pneumonia', help='Name of the column that contains ground truth in info.csv')
    retinanet_parser.add_argument('--extension', type=str, default='dcm')
    retinanet_parser.add_argument('--classes', type=int, default=1)
    # Type of model
    retinanet_parser.add_argument('--depth', type=int, default=101, help='Model depth')
    retinanet_parser.add_argument('--pretrained', action='store_true', help='Use pretrained network')
    retinanet_parser.add_argument('--optim', type=str, default='adam', help='Choose - sgd/adam')
    # For resuming model
    retinanet_parser.add_argument('--resume', action='store_true', help='Resume network')
    retinanet_parser.add_argument('--results', action='store_true', help='Compute box results csv')
    retinanet_parser.add_argument('--name', type=str, help='Model name to resume or compute final box results')
    # General parameters
    retinanet_parser.add_argument('--lr', type=float, default=1e-6)
    retinanet_parser.add_argument('--batchsize', type=int, default=8)
    retinanet_parser.add_argument('--workers', type=int, default=4)
    retinanet_parser.add_argument('--patience', type=int, default=10)
    # Identification parameter
    retinanet_parser.add_argument('--ID', type=str, default='placeholder', help='Special ID to identify a set of models')
    retinanet_parser.set_defaults(func=retinanet.main)

    # Wild - the main wild map calculating routine
    saliency_parser = subparsers.add_parser('saliency', help='Summarize all saliency maps results for models')
    # Parameters
    saliency_parser.add_argument('--name', type=str, default='all', help='all or specific model string')
    saliency_parser.add_argument('--start', type=int, default=0)
    saliency_parser.add_argument('--plot', action='store_true', help='Plot the maps')
    saliency_parser.set_defaults(func=saliency.main)

    # OOD - the main out of order distribution testing routine
    ood_parser = subparsers.add_parser('ood', help='Out of order distribution')
    # Parameters
    ood_parser.add_argument('--model', type=str, help='Model name to run for')
    ood_parser.add_argument('--debug', action='store_true', help='In debugging mode, runs for just 10 sample images.')
    ood_parser.add_argument('--data', type=str, default='siim', help='Name of the out distribution')
    ood_parser.add_argument('--column', type=str, default='Pneumothorax', help='Name of the column that contains ground truth in info.csv')
    ood_parser.add_argument('--extension', type=str, default='dcm')
    ood_parser.add_argument('--classes', type=int, default=1)
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
