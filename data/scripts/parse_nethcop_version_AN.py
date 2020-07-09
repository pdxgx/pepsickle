from optparse import OptionParser

# define command line parameters
parser = OptionParser()
parser.add_option("-i", "--in_file", dest="in_file",
                  help="")
parser.add_option("-o", "--out_file", dest="out_file",
                  help="")
(options, args) = parser.parse_args()

# handle = "/Users/weeder/Downloads/viral_proteins_cleavage_preds.txt"

with open(options.in_file, 'r') as netchop_file:
        line = netchop_file.readline()
        table_start = False
        table_list = []
        while line:
            if line == ' pos  AA  C      score      Ident\n':
                table_start = True
                # handles break line right under table headers
                line = netchop_file.readline()
                line = netchop_file.readline()
                table_lines = []
            while table_start:
                if line == '--------------------------------------\n':
                    table_start = False
                    table_list.append(table_lines)
                    break
                else:
                    table_lines.append(line)
                    line = netchop_file.readline()
            line = netchop_file.readline()


out = open(options.out_file, "w")
out.write(" pos aa  cleave_pred prob    ident\n")

for t in table_list:
    out.writelines(t)
    out.write(" NA  NA  NA  NA  NA\n")

out.close()
