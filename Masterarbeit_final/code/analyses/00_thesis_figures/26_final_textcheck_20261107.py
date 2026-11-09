"""Final wording corrections in two remaining Chapter3 vector figures.
Every non-text data artist is retained exactly; original exports unchanged.
"""
from pathlib import Path
import importlib.util,json,re,hashlib,xml.etree.ElementTree as ET
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
p=Path(__file__).with_name('25_chapter5_labels_20261107.py');spec=importlib.util.spec_from_file_location('labels26',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=ROOT/'results/00_thesis_figures/26_final_textcheck_20261107';m.FIG=OUT/'figures';m.FIG.mkdir(parents=True,exist_ok=True);m.register_fonts()
for name,key in [('distance_metric_agreement','3.5'),('species_parameter_domain_heatmap','3.10')]:
 source=ROOT/'results/00_thesis_figures/03_method_decisions/figures'/f'{name}.svg';m.revise(source,key)
 target=m.FIG/source.name;tree=ET.parse(target);root=tree.getroot();before=m.data_signature(root);parents={c:n for n in root.iter() for c in n};changes=[]
 for g in list(root.iter(m.Q+'g')):
  if not re.fullmatch(r'text_\d+',g.get('id','')):continue
  text=m.group_text(g);pos=m.position(g)
  if not pos:continue
  label=next((new for old,new in [('Mean NN distance','MNN'),('Mean inter-patch','MIP'),('Complete-MST','D50')] if old in text),None)
  if key=='3.5' and label:
   axis=parents[g]
   while axis is not None and not axis.get('id','').startswith('axes_'):axis=parents.get(axis)
   box=m.bounds(axis);ys=[float(t.get('y')) for t in g.iter(m.Q+'text') if t.get('y')]
   y=sum(ys)/len(ys) if ys else pos[1]
   m.replace(g,[label],box[0]-7,y-3,size=8.5,anchor='end');changes.append({'before':text,'after':label})
  for old,new in [('Near-one responses','Occupancy near 1'),('Near zero','Occupancy near 0'),('Selected main domain','Selected range')]:
   if text==old:
    for t in g.iter(m.Q+'text'):
     if t.text==old:
      if new=='Occupancy near 0':
       x,y,_=m.position(g);m.replace(g,['Occupancy','near 0'],x,y-3,size=8.5)
       for node in g.iter(m.Q+'text'):node.set('style',"font-family:'LM Roman 8';font-size:8.5px;fill:white")
      else:t.text=new
      changes.append({'before':old,'after':new})
      break
 assert before==m.data_signature(root)
 tree.write(target,encoding='utf-8',xml_declaration=True);m.pdf(target)
 m.AUDIT[key]['additional_text_changes']=changes
(OUT/'revision_audit.json').write_text(json.dumps({'scope':'Two final Chapter3 terminology corrections only','new_analysis':False,'data_artists_unchanged':True,'figures':m.AUDIT},indent=2)+'\n')
