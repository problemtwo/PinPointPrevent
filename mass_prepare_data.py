# mass_prepare_data.py
# This python program takes an input directory and in output directory (in that order) from the command line, and crops out faces in subdirectories.

import os
from prepare_data import main

for d in os.listdir(sys.argv[1]):
	if not os.path.exists(os.path.join('training',d)):
		os.makedir(os.path.join('training',d))
	main(os.path.join(sys.argv[1],d),d)
