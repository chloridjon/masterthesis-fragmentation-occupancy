"""Revise Chapter5 labels in existing native SVGs, retaining every data artist.
Only text groups change. No plotting data, interpolation or model calculations.
"""
from pathlib import Path
import sys,re,json,hashlib,ctypes,xml.etree.ElementTree as ET,copy,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/chapter5-label-mpl")
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='Masterarbeit_final')
sys.path.insert(0,str(ROOT/'code/base'))
from src.tools.final_plot_style import roman_font_files
OUT=ROOT/'results/00_thesis_figures/25_chapter5_labels_20261107';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
NS='http://www.w3.org/2000/svg';Q='{'+NS+'}'
ET.register_namespace('',NS);ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
AUDIT={}

def register_fonts():
 fc=ctypes.CDLL('libfontconfig.so.1');fc.FcConfigGetCurrent.restype=ctypes.c_void_p;fc.FcConfigAppFontAddFile.argtypes=[ctypes.c_void_p,ctypes.c_char_p];fc.FcConfigAppFontAddFile.restype=ctypes.c_int
 cfg=fc.FcConfigGetCurrent()
 for p in [*roman_font_files().values(),Path('/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf')]:assert p.exists() and fc.FcConfigAppFontAddFile(cfg,str(p).encode())

def pdf(svgpath):
 root=ET.parse(svgpath).getroot();w=float(root.get('width').removesuffix('pt'));h=float(root.get('height').removesuffix('pt'))
 svg=ctypes.CDLL('librsvg-2.so.2');c=ctypes.CDLL('libcairo.so.2');g=ctypes.CDLL('libgobject-2.0.so.0')
 svg.rsvg_handle_new_from_file.argtypes=[ctypes.c_char_p,ctypes.c_void_p];svg.rsvg_handle_new_from_file.restype=ctypes.c_void_p
 c.cairo_pdf_surface_create.argtypes=[ctypes.c_char_p,ctypes.c_double,ctypes.c_double];c.cairo_pdf_surface_create.restype=ctypes.c_void_p;c.cairo_create.argtypes=[ctypes.c_void_p];c.cairo_create.restype=ctypes.c_void_p;c.cairo_scale.argtypes=[ctypes.c_void_p,ctypes.c_double,ctypes.c_double]
 svg.rsvg_handle_render_cairo.argtypes=[ctypes.c_void_p,ctypes.c_void_p];svg.rsvg_handle_render_cairo.restype=ctypes.c_int
 for n in ['cairo_destroy','cairo_surface_finish','cairo_surface_destroy']:getattr(c,n).argtypes=[ctypes.c_void_p]
 g.g_object_unref.argtypes=[ctypes.c_void_p]
 handle=svg.rsvg_handle_new_from_file(str(svgpath).encode(),None);assert handle
 surface=c.cairo_pdf_surface_create(str(svgpath.with_suffix('.pdf')).encode(),w,h);cr=c.cairo_create(surface);c.cairo_scale(cr,.75,.75);assert svg.rsvg_handle_render_cairo(handle,cr)
 c.cairo_destroy(cr);c.cairo_surface_finish(surface);c.cairo_surface_destroy(surface);g.g_object_unref(handle)

def group_text(g):
 # Glyph order in mathtext SVG follows font runs rather than reading order.
 # Plain prose remains contiguous and is enough to identify the labels.
 return ''.join(n.text or '' for n in g.iter() if n.tag in [Q+'text',Q+'tspan'] and (n.tag==Q+'tspan' or not list(n))).replace('\xa0',' ').strip()

def bounds(g):
 for child in g:
  if child.get('id','').startswith('patch_'):
   p=child.find(Q+'path')
   if p is not None:
    vals=[float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',p.get('d',''))]
    if len(vals)>=8:
     x=vals[::2];y=vals[1::2];return (min(x),min(y),max(x),max(y))
 return None

def position(g):
 for n in g.iter():
  transform=n.get('transform','');m=re.search(r'translate\(([-\d.]+)[ ,]+([-\d.]+)\)',transform)
  if m:return float(m[1]),float(m[2]),'rotate(-90' in transform
  m=re.search(r'rotate\(-90 ([-\d.]+) ([-\d.]+)\)',transform)
  if m:return float(m[1]),float(m[2]),True
 for t in g.iter(Q+'text'):
  if 'x' in t.attrib and 'y' in t.attrib:return float(t.get('x')),float(t.get('y')),False
 return None

def rich(text,line):
 # Keep variable symbols italic; use real subscript spans for established notation.
 parts=re.split(r'(ell|r_s|ρ|(?<![A-Za-z])H(?![A-Za-z])|(?<![A-Za-z])r(?![A-Za-z]))',line)
 for part in parts:
  if not part:continue
  if part=='ell':
   ET.SubElement(text,Q+'tspan',{'font-family':'Latin Modern Math','font-style':'italic'}).text='ℓ'
   ET.SubElement(text,Q+'tspan',{'baseline-shift':'sub','font-size':'7'}).text='1/2'
  elif part=='r_s':
   ET.SubElement(text,Q+'tspan',{'font-style':'italic'}).text='r';ET.SubElement(text,Q+'tspan',{'baseline-shift':'sub','font-size':'7'}).text='s'
  else:ET.SubElement(text,Q+'tspan',{'font-style':'italic' if part in ['H','r','ρ'] else 'normal','font-family':'Latin Modern Math' if part=='ρ' else 'LM Roman 8'}).text=part

def replace(g,lines,x,y,vertical=False,italic=False,size=9.5,anchor='middle'):
 for n in list(g):g.remove(n)
 transform=f'translate({x:.6f} {y:.6f})'+(' rotate(-90)' if vertical else '')
 block=ET.SubElement(g,Q+'g',{'transform':transform,'data-revised-label':'true'})
 for i,line in enumerate(lines):
  t=ET.SubElement(block,Q+'text',{'x':'0','y':str((i-(len(lines)-1)/2)*11.7+3.0),'text-anchor':anchor,'font-family':'LM Roman 8','font-size':str(size),'font-style':'italic' if italic else 'normal','{http://www.w3.org/XML/1998/namespace}space':'preserve'})
  if italic:t.text=line
  else:rich(t,line)

def data_signature(root):
 # All non-text data shapes, image payloads and clipping geometry must be exact.
 items=[]
 def visit(n):
  if n.get('id','').startswith('text_') or n.tag in [Q+'text',Q+'tspan',Q+'style',Q+'metadata']:return
  if n.tag in [Q+'path',Q+'image',Q+'use',Q+'rect',Q+'polygon',Q+'polyline',Q+'circle']:
   items.append((n.tag,tuple(sorted(n.attrib.items()))))
  for c in n:visit(c)
 visit(root);return hashlib.sha256(repr(items).encode()).hexdigest()

def revise(path,key):
 tree=ET.parse(path);root=tree.getroot();before=data_signature(root);parents={c:n for n in root.iter() for c in n};axes={n:bounds(n) for n in root.iter(Q+'g') if n.get('id','').startswith('axes_')};axes={n:b for n,b in axes.items() if b};data=[b for b in axes.values() if b[2]-b[0]>25]
 assert data,path
 frame=(min(b[0] for b in data),min(b[1] for b in data),max(b[2] for b in data),max(b[3] for b in data));changed=[]
 for g in list(root.iter(Q+'g')):
  if not re.fullmatch(r'text_\d+',g.get('id','')):continue
  s=group_text(g);flat=re.sub(r'[\s-]+','',s).lower();pos=position(g)
  if not pos:continue
  x,y,vertical=pos;parent=parents.get(g);axis=parent
  while axis is not None and axis not in axes:axis=parents.get(axis)
  box=axes.get(axis,frame);yc=(box[1]+box[3])/2;xc=(box[0]+box[2])/2;lines=None
  if any(term in flat for term in ['habitatproportion','forestproportion','forestcoverproportion']) or (flat.startswith('forestcover') and parent is not None and parent.get('id','')=='figure_1'):
   lines=['Forest cover H'];x=xc if axis is not None else (frame[0]+frame[2])/2
  elif vertical:
   if 'extinction-to-coloni' in s or 'Extinction-to-coloni' in s:lines=['Extinction-to-colonisation','ratio r (log scale)']
   elif 'Dispersal' in s and 'half-distance' in s:lines=['Dispersal half-distance ell','(cells; log scale)']
   elif key=='5.1' and 'occupancy' in s.lower():lines=['Mean equilibrium patch','occupancy']
   elif 'Median equilibrium patch occupancy' in s:lines=['Mean equilibrium patch occupancy','(median across runs)']
   elif 'occupancy difference' in s.lower():lines=['Mean equilibrium patch','occupancy difference']
   elif s.startswith('Occupancy:') or s.startswith('Occupancy difference:'):lines=['Change in mean equilibrium','patch occupancy']
   elif 'Spearman' in s:lines=['Spearman rank','correlation r_s'] if 'Median' not in s else ['Median Spearman r_s']
   elif 'Reconstructed forest-site occupancy' in s:lines=['Reconstructed forest-site','occupancy']
   elif s=='Occupancy (0–1)':lines=['Occupancy']
   elif 'Sum of seven curve RMSEs' in s:lines=['Sum of seven','curve RMSEs']
   elif 'Signed area' in s:lines=['Signed area','(occupancy × H)']
   # Even labels retaining their wording must centre on the data axes, not the canvas.
   if lines is None and len(list(g.iter(Q+'text')))==1 and not list(next(g.iter(Q+'text'))):lines=[s]
   if lines:replace(g,lines,x,yc,True);changed.append({'id':g.get('id'),'before':s,'after':lines,'axis_center_y':yc});continue
  if lines:
   replace(g,lines,x,y,False);changed.append({'id':g.get('id'),'before':s,'after':lines});continue
  # Retain the empirical quantity; only model-specific map annotations use patch terminology.
  if key=='5.5' and ('Mean patch occupancy' in s or 'mean occupancy' in s):
   value=re.search(r'(?:[+=−-]\s*)?[0-9]+\.[0-9]+',s)
   if 'Mean patch occupancy' in s and value:
    replace(g,['Mean equilibrium patch','occupancy: '+value[0]],x,y-2,size=8.5,anchor='start');changed.append({'id':g.get('id'),'before':s,'after':'Mean equilibrium patch occupancy'});continue
   if value and ('Increase' in s or 'Decrease' in s):
    sign='+' if 'Increase' in s else '−';verb='Increase' if 'Increase' in s else 'Decrease'
    replace(g,[verb+' in mean equilibrium patch occupancy: '+sign+value[0].lstrip('+=−- ')],x,y-3,size=8.5,anchor='start')
    g.set('fill','#377A9F' if verb=='Increase' else '#C47723');changed.append({'id':g.get('id'),'before':s,'after':'Mean equilibrium patch occupancy change'});continue
  if key=='5.5' and re.match(r'\([a-d]\) H=',s):
   replace(g,[s.replace(' H=',' Forest cover H = ')],x,y-3,size=9.5,anchor='start');changed.append({'id':g.get('id'),'before':s,'after':'Forest cover H'});continue
  if s=='Non-habitat':
   replace(g,['Non-forest'],x,y,size=8.5,anchor='start');changed.append({'id':g.get('id'),'before':s,'after':'Non-forest'});continue
  # Scenario headings and legend keys, including old mathtext font-run encodings.
  scenario=next((name for token,name in [('onlyrandom','only random'),('edgerandom','edge random'),('accessbased','access based')] if token in flat),None)
  if scenario:
   prefix=re.search(r'\([a-z]\)',s);rho=re.search(r'ρ\s*=\s*([0-9.]+)',s)
   if parent is not None and parent.get('id','').startswith('axes_') and prefix:
    txt=prefix[0]+' '+scenario+(f', ρ = {rho[1]}' if rho else '');replace(g,[txt],x,y,size=9.5,italic=True,anchor='start')
   elif ('Only-Random' in s or 'Edge-Random' in s or 'Access-based' in s) and not rho:
    txt=s.replace('H=','Forest cover H = ').replace('Only-Random','only random').replace('Edge-Random','edge random').replace('Access-based','access based').replace('seeds','runs');replace(g,[txt],x,y,size=8.5,anchor='start',italic=True)
   else:
    # Existing scenario keys already use correct lowercase names. Italicise all glyph runs.
    for t in g.iter():
     if t.tag in [Q+'text',Q+'tspan']:t.set('font-style','italic');t.set('style',t.get('style','')+';font-style:italic')
   changed.append({'id':g.get('id'),'scenario':scenario})
 # Register actual optical-size8 family for the SVG renderer; preserve mathematical faces.
 for node in root.iter():
  if node.get('font-family'):
   style=node.get('style','')+";font-family:'"+node.get('font-family')+"', 'Latin Modern Math';"
   if node.get('font-style'):style+='font-style:'+node.get('font-style')+';'
   if node.get('font-size'):style+='font-size:'+node.get('font-size')+'px;'
   node.set('style',style)
  if 'style' in node.attrib:
   style=node.get('style').replace("'Latin Modern Roman 8'","'LM Roman 8'").replace("'Latin Modern Roman'","'LM Roman 8'")
   node.set('style',style)
 assert before==data_signature(root),'Non-text data artists changed'
 target=FIG/path.name;tree.write(target,encoding='utf-8',xml_declaration=True);pdf(target)
 AUDIT[key]={'source':str(path.relative_to(ROOT)),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'data_artist_signature':before,'data_artists_unchanged':True,'labels':changed,'target':str(target.with_suffix('.pdf').relative_to(ROOT))}
 print('Revised',key,flush=True)

if __name__=='__main__':
 register_fonts()
 sources=json.loads((ROOT/'config/00_thesis_figures/final_label_sources.json').read_text())
 for key,relative in sources.items():revise(ROOT/relative,key)
 assert len(AUDIT)==21
 (OUT/'revision_audit.json').write_text(json.dumps({'scope':'Final thesis text/layout only','new_analysis':False,'figures':AUDIT},indent=2,ensure_ascii=False)+'\n')
