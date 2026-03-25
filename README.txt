This project consists of a series of python files and folders.

Files explanation:
- "general_solver.py" is arguably the most important file. It contains a function to solve the generalized isentropic flow relation imposing the    conservation of mass as a constraint.
- "Main_v1.py" utilizes the function contained in the solver to get results for a series of conditions.
    Those results are stored in folder that are automatically created from the file (e.g. BOS_DEC_CORR_2025)
- "Main_gammapv.py" utilizes the generalized isentropic equation from Nederstigt 2023 in combination with the Mach area relation to solve the quasi 1D problem.
    SINCE THE MACH-AREA RELATION IS DERIVED FOR A PERFECT GAS, THE CONSERVATION OF MASS IS NOT RESPECTED (DISCOVERED LATER)
- The remaining files are only used to plot or analyze the results, they are commented relatively well.

Folders explanation:
- "coord" contains the nominal dimensions of the ORCHID nozzle
- Any other folder is generated while solving the quasi 1D problem wither with Main_v1 or Main_gammapv