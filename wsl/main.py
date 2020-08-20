from argparse import ArgumentParser
import wsl.train.main as train

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
    train_parser.add_argument('--check',
                              action='store_true',
                              help='Prints \'works\' on successful install')
    train_parser.set_defaults(func=train.main)

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
