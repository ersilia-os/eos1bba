import csv
import sys
from gem import get_gem_pred


# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]
# run model
outputs = get_gem_pred(smiles_list)


# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
           'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'])  # header
    for o in outputs:
        writer.writerow(o)
