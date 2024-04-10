import argparse

from CLIP_AC import ClipAC


parser = argparse.ArgumentParser(prog='CLIPAC')
parser.add_argument('-i', '--image', type=str)
parser.add_argument('-l', '--label', type=str)

args = parser.parse_args()

clip_ac = ClipAC(args.label)

print(clip_ac(args.image))
