import glob

p = glob.glob('../videos/**/*.mp4')

with open('file_mappings.tsv', 'w') as f:
    line = 'mediaID'+'\t'+'file'+'\n'
    f.write(line)
    for i, x in enumerate(p):
        line = str(i)+'\t'+x[10:]+'\n'
        f.write(line)