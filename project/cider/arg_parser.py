"""Define argument parser class."""
import argparse

class ArgParser(object):
    """Argument parser for label.py"""
    def __init__(self):
        """Initialize argument parser."""
        parser = argparse.ArgumentParser()

        # Input report parameters.
        parser.add_argument('--data',
                            required=True,
                            help='Path to (train/test)_reports.txt')

        parser.add_argument('--output_path',
                            required=True,
                            help='Output path to write tf to.')

        self.parser = parser

    def parse_args(self):
        """Parse and validate the supplied arguments."""
        args = self.parser.parse_args()

        return args
