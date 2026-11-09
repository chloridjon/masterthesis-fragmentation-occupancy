"""Portable compatibility aliases for scientific subprocesses."""
import json,os,runpy,sys
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'workflow.json').is_file())
ALIASES=json.loads((ROOT/'tools/job_aliases.json').read_text())
def main():
    if len(sys.argv)<2 or sys.argv[1]=='--list':
        print('\n'.join(sorted(ALIASES)));return
    name=sys.argv[1]
    if name not in ALIASES:raise SystemExit('Unknown scientific job: '+name)
    path=ROOT/ALIASES[name];sys.argv=[str(path),*sys.argv[2:]]
    os.chdir(ROOT);runpy.run_path(str(path),run_name='__main__')
if __name__=='__main__':main()
