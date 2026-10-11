"""Refit coupled-hydro policies and recompute references in a new directory."""
import argparse
from . import ROOT
from synthetic_hydro_certificate import scaling_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reservoirs',type=int,nargs='+',default=[3])
    parser.add_argument('--stages',type=int,nargs='+',default=[18])
    parser.add_argument('--output',default='outputs/refitted_certificate')
    args=parser.parse_args()
    scaling_run.OUT=ROOT/args.output
    for reservoirs in args.reservoirs:
        for stages in args.stages:scaling_run.run_cell(reservoirs,stages)


if __name__=='__main__':main()
