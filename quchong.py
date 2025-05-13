import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--result_file", "-r", type=str)
args = parser.parse_args()

fp=open(args.result_file,'r')
ofp=open('new_'+args.result_file, "a", encoding="utf-8")

lines=fp.readlines()

already = set()

for line in lines:
    json_line = json.loads(line)
    index = json_line['index']
    if index not in already:
        json.dump(json_line, ofp, ensure_ascii=False)
        ofp.write('\n')
        already.add(index)
