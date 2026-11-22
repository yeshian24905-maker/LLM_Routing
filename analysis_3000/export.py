"""Small local HTML, Word and Excel exporters; standard library only."""

import html
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as E

from PIL import Image


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}
for prefix, uri in NS.items():
    E.register_namespace(prefix, uri)


def node(tag, attrs=None, text=None):
    prefix, local = tag.split(":")
    attrs = {f"{{{NS[k.split(':')[0]]}}}{k.split(':')[1]}" if ":" in k else k: str(v)
             for k, v in (attrs or {}).items()}
    e = E.Element(f"{{{NS[prefix]}}}{local}", attrs)
    e.text = text
    return e


def sub(parent, tag, attrs=None, text=None):
    e = node(tag, attrs, text)
    parent.append(e)
    return e


def xml(e):
    return E.tostring(e, encoding="utf-8", xml_declaration=True)


def paragraph(parent, text="", style=None, size=None, bold=False):
    p = sub(parent, "w:p")
    if style or size:
        props = sub(p, "w:pPr")
        if style:
            sub(props, "w:pStyle", {"w:val": style})
    for i, line in enumerate(str(text).split("\n")):
        r = sub(p, "w:r")
        if size or bold:
            props = sub(r, "w:rPr")
            if size:
                sub(props, "w:sz", {"w:val": size})
            if bold:
                sub(props, "w:b")
        if i:
            sub(r, "w:br")
        t = sub(r, "w:t", text=line)
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return p


class Report:
    def __init__(self, title, subtitle):
        self.title, self.subtitle = title, subtitle
        self.blocks = [("title", title), ("subtitle", subtitle)]

    def h1(self, text): self.blocks.append(("h1", text))
    def h2(self, text): self.blocks.append(("h2", text))
    def p(self, text): self.blocks.append(("p", text))
    def note(self, text): self.blocks.append(("note", text))
    def page(self): self.blocks.append(("page",))
    def table(self, headers, rows, widths=None): self.blocks.append(("table", headers, rows, widths))
    def image(self, path, caption): self.blocks.append(("image", Path(path), caption))

    def save_html(self, path):
        esc = lambda s: html.escape(str(s)).replace("\n", "<br>")
        parts = []
        for b in self.blocks:
            kind = b[0]
            if kind in ("title", "subtitle", "h1", "h2", "p", "note"):
                tag = {"title":"h1", "subtitle":"p", "h1":"h2", "h2":"h3", "p":"p", "note":"aside"}[kind]
                parts.append(f'<{tag} class="{kind}">{esc(b[1])}</{tag}>')
            elif kind == "page":
                parts.append('<div class="pagebreak"></div>')
            elif kind == "image":
                rel = b[1].relative_to(path.parent).as_posix()
                parts.append(f'<figure><img src="{esc(rel)}" alt="{esc(b[2])}"><figcaption>{esc(b[2])}</figcaption></figure>')
            elif kind == "table":
                parts.append('<table><thead><tr>'+"".join(f'<th>{esc(c)}</th>' for c in b[1])+"</tr></thead><tbody>")
                for row in b[2]:
                    parts.append("<tr>"+"".join(f"<td>{esc(c)}</td>" for c in row)+"</tr>")
                parts.append("</tbody></table>")
        css = """
@page {size:A4; margin:16mm 14mm 17mm;}
* {box-sizing:border-box;} body {font:10pt/1.5 'Microsoft YaHei','Noto Sans CJK SC',sans-serif;color:#243746;max-width:1050px;margin:auto;background:white;}
h1,h2,h3 {color:#153e59;line-height:1.3;break-after:avoid;} h1.title {font-size:23pt;margin:6mm 0 3mm;} h2 {font-size:16pt;border-bottom:2px solid #278779;padding-bottom:2mm;margin-top:5mm;} h3 {font-size:11.5pt;margin-bottom:2mm;}
.subtitle {font-size:11pt;color:#5a7180;} p {margin:2mm 0;text-align:justify;} aside {padding:2.5mm 3.5mm;background:#edf5f5;border-left:3px solid #278779;font-size:9.5pt;margin:3mm 0;}
table {border-collapse:collapse;width:100%;font-size:8.4pt;line-height:1.35;margin:3mm 0;table-layout:auto;} th {background:#153e59;color:white;text-align:left;padding:1.5mm;} td {padding:1.4mm;border-bottom:1px solid #d9e3e8;vertical-align:top;} tr:nth-child(even) {background:#f3f7f9;} tr {break-inside:avoid;} thead {display:table-header-group;}
figure {margin:3mm 0;break-inside:avoid;text-align:center;} img {width:auto;max-width:100%;height:auto;max-height:218mm;object-fit:contain;} figcaption {font-size:8.2pt;color:#5a7180;margin-top:1mm;text-align:left;} .pagebreak {break-before:page;}
@media screen {body {padding:30px 45px;background:#fff;box-shadow:0 0 30px #d9e3e8;margin:25px auto;} html {background:#eef3f7;}.pagebreak {border-top:1px dashed #bbcbd4;margin-top:35px;padding-top:10px;}}
"""
        path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>'+esc(self.title)+
                        '</title><style>'+css+'</style><body>'+"\n".join(parts)+"</body></html>", encoding="utf-8")

    def save_docx(self, path):
        doc = node("w:document")
        body = sub(doc,"w:body")
        images = []
        rels = [("rIdStyles", "styles", "styles.xml"), ("rIdFooter", "footer", "footer1.xml")]
        for b in self.blocks:
            kind = b[0]
            if kind in ("title", "subtitle", "h1", "h2", "p", "note"):
                paragraph(body,b[1],{"title":"Title", "subtitle":"Subtitle", "h1":"Heading1",
                                     "h2":"Heading2", "p":"Normal", "note":"Quote"}[kind])
            elif kind == "page":
                p = sub(body,"w:p");r = sub(p,"w:r");sub(r,"w:br",{"w:type":"page"})
            elif kind == "table":
                tbl = sub(body,"w:tbl")
                props = sub(tbl,"w:tblPr")
                sub(props,"w:tblW",{"w:w":10000,"w:type":"dxa"})
                sub(props,"w:tblLayout",{"w:type":"fixed"})
                margins=sub(props,"w:tblCellMar")
                for side in ("top","left","bottom","right"):
                    sub(margins,f"w:{side}",{"w:w":65,"w:type":"dxa"})
                widths = b[3] or [10000//len(b[1])]*len(b[1])
                grid=sub(tbl,"w:tblGrid")
                for w in widths: sub(grid,"w:gridCol",{"w:w":w})
                for i,row in enumerate([b[1]]+b[2]):
                    tr = sub(tbl,"w:tr")
                    tp=sub(tr,"w:trPr");sub(tp,"w:cantSplit")
                    if i == 0:sub(tp,"w:tblHeader")
                    for j,value in enumerate(row):
                        tc=sub(tr,"w:tc");cp=sub(tc,"w:tcPr")
                        sub(cp,"w:tcW",{"w:w":widths[j],"w:type":"dxa"})
                        sub(cp,"w:shd",{"w:fill":"153E59" if i==0 else ("F0F5F7" if i%2==0 else "FFFFFF")})
                        p=paragraph(tc,value,"TableText",size=17,bold=i==0)
                        if i==0:
                            for run in p.findall("w:r",NS):
                                rp=run.find("w:rPr",NS);sub(rp,"w:color",{"w:val":"FFFFFF"})
                paragraph(body,"",size=4)
            elif kind == "image":
                idx=len(images)+1;name=f"image{idx}.png";rid=f"rIdImage{idx}"
                images.append((name,b[1]));rels.append((rid,"image",f"media/{name}"))
                with Image.open(b[1]) as im: iw,ih=im.size
                cx=min(6300000,round(7900000*iw/ih));cy=round(cx*ih/iw)
                p=sub(body,"w:p");pp=sub(p,"w:pPr");sub(pp,"w:jc",{"w:val":"center"});sub(pp,"w:keepNext")
                r=sub(p,"w:r");d=sub(r,"w:drawing");inline=sub(d,"wp:inline",{"distT":0,"distB":0,"distL":0,"distR":0})
                sub(inline,"wp:extent",{"cx":cx,"cy":cy});sub(inline,"wp:docPr",{"id":idx,"name":b[2]})
                graphic=sub(inline,"a:graphic");gd=sub(graphic,"a:graphicData",{"uri":NS["pic"]})
                pic=sub(gd,"pic:pic");nv=sub(pic,"pic:nvPicPr");sub(nv,"pic:cNvPr",{"id":0,"name":name});sub(nv,"pic:cNvPicPr")
                fill=sub(pic,"pic:blipFill");sub(fill,"a:blip",{"r:embed":rid});stretch=sub(fill,"a:stretch");sub(stretch,"a:fillRect")
                sp=sub(pic,"pic:spPr");xf=sub(sp,"a:xfrm");sub(xf,"a:off",{"x":0,"y":0});sub(xf,"a:ext",{"cx":cx,"cy":cy})
                geo=sub(sp,"a:prstGeom",{"prst":"rect"});sub(geo,"a:avLst")
                paragraph(body,b[2],"Caption")
        section=sub(body,"w:sectPr")
        sub(section,"w:footerReference",{"w:type":"default","r:id":"rIdFooter"})
        sub(section,"w:pgSz",{"w:w":11906,"w:h":16838})
        sub(section,"w:pgMar",{"w:top":900,"w:right":950,"w:bottom":900,"w:left":950,"w:header":350,"w:footer":400})
        styles=node("w:styles")
        defaults=sub(styles,"w:docDefaults");rp=sub(sub(defaults,"w:rPrDefault"),"w:rPr")
        sub(rp,"w:rFonts",{"w:ascii":"Calibri","w:hAnsi":"Calibri","w:eastAsia":"Microsoft YaHei"})
        sub(rp,"w:sz",{"w:val":21});sub(rp,"w:lang",{"w:val":"en-US","w:eastAsia":"zh-CN"})
        pp=sub(sub(defaults,"w:pPrDefault"),"w:pPr");sub(pp,"w:spacing",{"w:after":100,"w:line":310,"w:lineRule":"auto"})
        definitions=[("Normal",21,"243746",False), ("Title",42,"153E59",True),
                     ("Subtitle",23,"5A7180",False), ("Heading1",30,"153E59",True),
                     ("Heading2",24,"153E59",True), ("Caption",17,"5A7180",False),
                     ("Quote",20,"278779",False), ("TableText",17,"243746",False)]
        for sid,sz,color,bold in definitions:
            st=sub(styles,"w:style",{"w:type":"paragraph","w:styleId":sid})
            sub(st,"w:name",{"w:val":sid})
            if sid!="Normal":sub(st,"w:basedOn",{"w:val":"Normal"})
            pp=sub(st,"w:pPr")
            if sid in ("Title","Heading1","Heading2"):
                sub(pp,"w:keepNext");sub(pp,"w:keepLines")
                if sid.startswith("Heading"):sub(pp,"w:outlineLvl",{"w:val":int(sid[-1])-1})
            if sid=="TableText":sub(pp,"w:spacing",{"w:after":20,"w:line":260,"w:lineRule":"auto"})
            rp=sub(st,"w:rPr");sub(rp,"w:sz",{"w:val":sz});sub(rp,"w:color",{"w:val":color})
            if bold:sub(rp,"w:b")
        footer=node("w:ftr");p=paragraph(footer,"MMMU | 3,000 questions | 2026-11-22 | ","Caption")
        r=sub(p,"w:r");sub(r,"w:fldChar",{"w:fldCharType":"begin"})
        r=sub(p,"w:r");sub(r,"w:instrText",text="PAGE")
        r=sub(p,"w:r");sub(r,"w:fldChar",{"w:fldCharType":"end"})
        content_types='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'''
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml",content_types)
            z.writestr("_rels/.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
            z.writestr("word/document.xml",xml(doc));z.writestr("word/styles.xml",xml(styles));z.writestr("word/footer1.xml",xml(footer))
            z.writestr("word/_rels/document.xml.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+"".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/{typ}" Target="{target}"/>' for rid,typ,target in rels)+"</Relationships>")
            for name,p in images:z.write(p,f"word/media/{name}")


def render_pdf(source, target):
    edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
    if not edge.is_file():
        print("PDF skipped: local Microsoft Edge was not found; HTML and Word are available.")
        return
    # Use a separate temporary profile so existing interactive browser sessions are unaffected.
    profile = tempfile.mkdtemp(prefix="mmmu-report-")
    try:
        subprocess.run([str(edge), "--headless", "--disable-gpu", "--no-sandbox",
                        "--disable-background-networking", "--disable-component-update",
                        "--no-first-run", "--disable-sync", "--no-pdf-header-footer",
                        f"--user-data-dir={profile}", f"--print-to-pdf={target}", source.as_uri()],
                       check=True, capture_output=True, timeout=45,
                       creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if not target.is_file():
            raise RuntimeError("Browser exited without writing the PDF")
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("PDF export failed; the HTML and Word outputs are still available") from exc


def workbook(path, sheets):
    """Export typed table cells; all text is literal (never an Excel formula)."""
    ss="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    def sheet_xml(rows):
        root=E.Element("worksheet",xmlns=ss)
        views=E.SubElement(root,"sheetViews");view=E.SubElement(views,"sheetView",workbookViewId="0")
        E.SubElement(view,"pane",ySplit="1",topLeftCell="A2",activePane="bottomLeft",state="frozen")
        data=E.SubElement(root,"sheetData")
        keys=list(rows[0])
        for rn,values in enumerate([keys]+[[row[k] for k in keys] for row in rows],1):
            tr=E.SubElement(data,"row",r=str(rn))
            for cn,v in enumerate(values):
                n=cn+1; letters=""
                while n:n,k=divmod(n-1,26);letters=chr(65+k)+letters
                ref=f"{letters}{rn}"
                if v is None:continue
                if isinstance(v,(int,float)) and not isinstance(v,bool):
                    c=E.SubElement(tr,"c",r=ref);E.SubElement(c,"v").text=str(v)
                else:
                    c=E.SubElement(tr,"c",r=ref,t="inlineStr")
                    text=E.SubElement(E.SubElement(c,"is"),"t")
                    text.text=re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]","",str(v))[:32767]
        E.SubElement(root,"autoFilter",ref=f"A1:{letters}{len(rows)+1}")
        return xml(root)
    with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as z:
        types=['<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>','<Default Extension="xml" ContentType="application/xml"/>','<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>']
        wb=E.Element("workbook",xmlns=ss);sh=E.SubElement(wb,"sheets")
        relationships=[]
        for i,(name,rows) in enumerate(sheets.items(),1):
            E.SubElement(sh,"sheet",name=name[:31],sheetId=str(i),attrib={f"{{{NS['r']}}}id":f"rId{i}"})
            z.writestr(f"xl/worksheets/sheet{i}.xml",sheet_xml(rows))
            types.append(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
            relationships.append(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>')
        z.writestr("xl/workbook.xml",xml(wb))
        z.writestr("xl/_rels/workbook.xml.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+"".join(relationships)+"</Relationships>")
        z.writestr("_rels/.rels",'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("[Content_Types].xml",'<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'+"".join(types)+"</Types>")
