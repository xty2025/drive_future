"""Merge several visualization runs into one HTML gallery with a zoomable image viewer.

Every --section TITLE DIR INTRO adds a tab with that run's overview and scene figures, in the run's own order and with
its captions (read from the run's index.html), filterable by split and scene group. The grid shows JPEG thumbnails;
the viewer opens the full-resolution figure: mouse wheel or pinch to zoom at the cursor, drag to pan, double-click to
switch between fit and 1:1, arrow keys for the previous / next figure.

Without --embed the thumbnails are files next to the gallery and the viewer loads the run's PNGs by relative path,
so the gallery has to stay where it is written relative to the run folders. With --embed the page is self-contained:
thumbnails (JPEG) and full-resolution figures (WebP, decoded only when opened) are data URIs and every run's
summary.md is inlined.

With --pairs, columns of every run (models.json order) are compared in pairs, 1-2, 3-4, ... (--pairs adjacent) or
1-2, 2-3, 3-4, ... (--pairs chain), in a full-navtest table from their evaluation CSVs: PDMS difference B - A with a
95% interval over the test scenes, scenes that move by >= 0.5 in each direction, and the sub-metric differences. When
every section shows the same models (e.g. one run per split) the table is shown once above the tabs.
"""

import argparse
import base64
import csv
import html
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

SUB = (("NC", "no_at_fault_collisions"), ("DAC", "drivable_area_compliance"), ("TTC", "time_to_collision_within_bound"),
       ("C", "comfort"), ("EP", "ego_progress"))
GROUP_TEXT = {"ref_better": "{ref} 好、{other} 多数失败", "other_better": "{other} 好、{ref} 失败",
              "disagree": "模型之间不一致", "scorer_miss": "打分器漏掉好候选", "low_coverage": "有模型没有好候选",
              "all_good": "全部都好", "manual": "指定场景", "overview": "总览"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="gallery HTML file")
    parser.add_argument("--title", required=True)
    parser.add_argument("--intro", default="", help="paragraph under the title")
    parser.add_argument("--section", nargs=3, action="append", required=True, metavar=("TITLE", "DIR", "INTRO"))
    parser.add_argument("--pairs", choices=("adjacent", "chain"), help="add a full-navtest table of column pairs")
    parser.add_argument("--thumb-width", type=int, default=960)
    parser.add_argument("--embed", action="store_true", help="self-contained page: images and summaries inlined")
    parser.add_argument("--full-quality", type=int, default=90, help="WebP quality of the embedded full figures")
    return parser.parse_args()


def read_entries(run):
    """(image, caption) in the order of the run's index.html."""
    text = (run / "index.html").read_text(encoding="utf-8")
    return [(html.unescape(image), html.unescape(caption))
            for caption, image in re.findall(r'<section><p>(.*?)</p><a href="(.*?)">', text, flags=re.S)]


def group_of(image):
    """(split, group) from the file names the visualizers write: <split>/<n>_<group>_<token>.png, overview_<split>.png."""
    path = Path(image)
    if path.stem.startswith("overview_"):
        return path.stem.split("_", 1)[1], "overview"
    return path.parent.name, path.stem.split("_", 1)[1].rsplit("_", 1)[0]


def group_label(group, families):
    ref = families[0] if families else "A"
    other = next((family for family in families if family != ref), ref) if families else "B"
    base = group.removesuffix("_closest")
    text = GROUP_TEXT.get(base, base).format(ref=ref, other=other)
    return text + ("（无完全符合的场景，取最接近的）" if base != group else "")


def read_csv(eval_dir):
    """Per-token rows of the newest evaluation CSV of an experiment directory or one run directory."""
    eval_dir = Path(eval_dir)
    files = sorted([*eval_dir.glob("*.csv"), *eval_dir.glob("*/*.csv")], key=lambda path: path.stat().st_mtime)
    with open(files[-1]) as stream:
        return {row["token"]: row for row in csv.DictReader(stream)
                if row["token"] != "average" and row["valid"] == "True"}


def pair_rows(run, mode):
    """Full-navtest comparison of model columns of a run: 1-2, 3-4, ... (adjacent) or 1-2, 2-3, ... (chain)."""
    meta = json.loads((run / "models.json").read_text(encoding="utf-8"))
    labels = list(meta["titles"])
    rows = []
    for a, b in (zip(labels[:-1], labels[1:]) if mode == "chain" else zip(labels[0::2], labels[1::2])):
        csv_a, csv_b = (read_csv(meta["models"][label]["eval_dir"]) for label in (a, b))
        tokens = sorted(set(csv_a) & set(csv_b))
        score = lambda table, field: np.array([float(table[t][field]) for t in tokens])  # noqa: E731
        sa, sb = score(csv_a, "score"), score(csv_b, "score")
        diff = sb - sa
        rows.append({"a": meta["titles"][a], "b": meta["titles"][b], "n": len(tokens),
                     "pdms_a": 100 * sa.mean(), "pdms_b": 100 * sb.mean(), "delta": 100 * diff.mean(),
                     "ci": 196 * diff.std(ddof=1) / np.sqrt(len(diff)),
                     "better": int(np.sum(diff >= 0.5)), "worse": int(np.sum(diff <= -0.5)),
                     "sub": [100 * (score(csv_b, field).mean() - score(csv_a, field).mean()) for _, field in SUB]})
    return rows


def data_uri(image, fmt, **options):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **options)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buffer.getvalue()).decode()


def embedded(source, width, quality):
    """(thumbnail, full figure) of a PNG as data URIs."""
    with Image.open(source) as image:
        image = image.convert("RGB")
        small = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
        return data_uri(small, "JPEG", quality=84, optimize=True), data_uri(image, "WEBP", quality=quality, method=4)


def thumbnail(source, target, width):
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        height = round(image.height * width / image.width)
        image.resize((width, height), Image.LANCZOS).save(target, "JPEG", quality=84, optimize=True)


def pair_table(rows):
    head = ("<tr><th>A → B</th><th>A navtest</th><th>B navtest</th><th>B − A（±95%）</th>"
            "<th>B 明显更好 / 更差<br><span class=sub>|Δ| ≥ 0.5 的场景</span></th>"
            + "".join(f"<th>{name}</th>" for name, _ in SUB) + "</tr>")
    body = []
    for row in rows:
        beyond = abs(row["delta"]) > row["ci"]
        body.append(
            f"<tr><td>{html.escape(row['a'])}<br>→ {html.escape(row['b'])}</td><td>{row['pdms_a']:.2f}</td>"
            f"<td>{row['pdms_b']:.2f}</td><td class='{'beyond' if beyond else 'within'}'>{row['delta']:+.2f} ± "
            f"{row['ci']:.2f}</td><td>{row['better']} / {row['worse']}</td>"
            + "".join(f"<td class='{'neg' if v < -0.2 else 'pos' if v > 0.2 else ''}'>{v:+.2f}</td>" for v in row["sub"])
            + "</tr>")
    note = (f"全量 navtest {rows[0]['n']:,} 个场景，B − A 单位为 PDMS 点。<b>加粗</b>表示差值超出 ±95% 区间；"
            "该区间只含测试场景抽样误差，不含训练随机种子，且均为单次训练。各 run 的 summary.md 只用 80 个随机测试场景，"
            "会波动 2~4 分，结论以此表为准。")
    return f"<table class=pairs>{head}{''.join(body)}</table><p class=note>{note}</p>"


CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--accent:#2a78d6;--neg:#d03b3b;--pos:#006300}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",sans-serif;
background:var(--page);color:var(--ink)}
header{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--grid);padding:14px 24px 10px}
h1{font-size:20px;margin:0 0 6px}p.intro{margin:0 0 10px;color:var(--ink2);font-size:14px;max-width:1500px;line-height:1.55}
nav.tabs{display:flex;gap:6px;flex-wrap:wrap}nav.tabs button{font:inherit;font-size:14px;padding:7px 14px;border:1px solid var(--axis);
background:var(--surface);color:var(--ink);border-radius:999px;cursor:pointer}
nav.tabs button.on{background:var(--ink);color:var(--surface);border-color:var(--ink)}
main{padding:18px 24px 60px}section.tab{display:none}section.tab.on{display:block}
section.tab>h2{font-size:18px;margin:4px 0 6px}section.tab>p{color:var(--ink2);margin:0 0 12px;font-size:14px;line-height:1.55}
.links a{color:var(--accent);margin-right:14px;font-size:13px}
table.pairs{border-collapse:collapse;background:var(--surface);font-size:13px;margin:10px 0 4px}
table.pairs th,table.pairs td{border-bottom:1px solid var(--grid);padding:6px 10px;text-align:right;white-space:nowrap}
table.pairs th:first-child,table.pairs td:first-child{text-align:left}table.pairs th{color:var(--ink2);font-weight:600}
td.beyond{font-weight:700}td.neg{color:var(--neg)}td.pos{color:var(--pos)}span.sub{font-weight:400;color:var(--muted);font-size:12px}
p.note{font-size:12.5px;color:var(--ink2);max-width:1400px;line-height:1.5}
details.summary{margin:4px 0 8px;font-size:13px}details.summary summary{cursor:pointer;color:var(--accent)}
details.summary pre{background:var(--surface);border:1px solid var(--grid);padding:10px 12px;overflow:auto;font-size:12px;max-height:420px}
.filters{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin:14px 0 12px;font-size:13px;color:var(--ink2)}
.filters .label{margin:0 4px 0 10px}.filters .label:first-child{margin-left:0}
.filters button{font:inherit;font-size:13px;padding:4px 11px;border:1px solid var(--axis);background:var(--surface);border-radius:999px;cursor:pointer;color:var(--ink)}
.filters button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(620px,1fr));gap:16px}
figure.card{margin:0;background:var(--surface);border:1px solid var(--grid);border-radius:8px;overflow:hidden}
figure.card.wide{grid-column:1/-1}figure.card a{display:block;cursor:zoom-in}figure.card img{display:block;width:100%;height:auto}
figcaption{padding:8px 12px 10px;font-size:12.5px;color:var(--ink2);line-height:1.5;word-break:break-word}
.chip{display:inline-block;padding:1px 8px;margin:0 6px 3px 0;border-radius:999px;background:#eceae4;color:var(--ink);font-size:12px}
#viewer{position:fixed;inset:0;z-index:50;background:rgba(11,11,11,.94);display:none}#viewer.on{display:block}
#stage{position:absolute;inset:0 0 64px 0;overflow:hidden;cursor:grab;touch-action:none}#stage.drag{cursor:grabbing}
#vimg{position:absolute;left:0;top:0;transform-origin:0 0;max-width:none;user-select:none;-webkit-user-drag:none}
#vbar{position:absolute;left:0;right:0;bottom:0;height:64px;display:flex;align-items:center;gap:8px;padding:0 16px;
background:rgba(11,11,11,.85);color:#fff;font-size:13px}
#vbar button,#vbar a{font:inherit;color:#fff;background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.25);
border-radius:6px;padding:6px 11px;cursor:pointer;text-decoration:none}#vbar button:hover,#vbar a:hover{background:rgba(255,255,255,.25)}
#vzoom{min-width:52px;text-align:center}#vpos{min-width:64px;text-align:center}
#vcap{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#e1e0d9;margin:0 10px}
#vhelp{color:#c3c2b7;font-size:12px;white-space:nowrap}
"""

JS = r"""
(function(){
const tabs=[...document.querySelectorAll('nav.tabs button')], sections=[...document.querySelectorAll('section.tab')];
function showTab(id){tabs.forEach(b=>b.classList.toggle('on',b.dataset.tab===id));
  sections.forEach(s=>s.classList.toggle('on',s.id===id));history.replaceState(null,'','#'+id);}
tabs.forEach(b=>b.addEventListener('click',()=>showTab(b.dataset.tab)));
showTab((location.hash.slice(1)&&document.getElementById(location.hash.slice(1)))?location.hash.slice(1):sections[0].id);
sections.forEach(section=>{
  const state={split:'all',group:'all'};
  const apply=()=>section.querySelectorAll('figure.card').forEach(c=>{
    c.hidden=!((state.split==='all'||c.dataset.split===state.split)&&(state.group==='all'||c.dataset.group===state.group));});
  section.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{
    state[b.dataset.key]=b.dataset.value;
    section.querySelectorAll(`.filters button[data-key="${b.dataset.key}"]`).forEach(x=>x.classList.toggle('on',x===b));apply();}));
});
const viewer=document.getElementById('viewer'), stage=document.getElementById('stage'), img=document.getElementById('vimg');
const zoomText=document.getElementById('vzoom'), posText=document.getElementById('vpos'), cap=document.getElementById('vcap'),
  openLink=document.getElementById('vopen');
const embed=document.body.dataset.embed==='1';
let list=[], idx=0, s=1, tx=0, ty=0, iw=1, ih=1, fitScale=1;
const ptrs=new Map(); let pinch=null;
function apply(){img.style.transform=`translate(${tx}px,${ty}px) scale(${s})`;zoomText.textContent=Math.round(s*100)+'%';}
function fit(){const r=stage.getBoundingClientRect();fitScale=Math.min(r.width/iw,r.height/ih);s=fitScale;
  tx=(r.width-iw*s)/2;ty=(r.height-ih*s)/2;apply();}
function zoomAt(x,y,f){const ns=Math.min(Math.max(s*f,fitScale*0.5),8);tx=x-(x-tx)*ns/s;ty=y-(y-ty)*ns/s;s=ns;apply();}
function center(){const r=stage.getBoundingClientRect();return [r.width/2,r.height/2];}
function show(i){idx=(i+list.length)%list.length;const item=list[idx];posText.textContent=`${idx+1} / ${list.length}`;
  cap.textContent=item.cap;cap.title=item.cap;openLink.href=item.full;
  if(embed)openLink.download=item.name;img.style.visibility='hidden';
  img.onload=()=>{iw=img.naturalWidth;ih=img.naturalHeight;img.style.width=iw+'px';img.style.height=ih+'px';fit();
    img.style.visibility='visible';};img.src=item.full;}
function open(section,anchor){const anchors=[...section.querySelectorAll('figure.card:not([hidden]) a.zoom')];
  list=anchors.map(a=>({full:a.dataset.full,cap:a.dataset.cap,name:a.dataset.name}));
  viewer.classList.add('on');document.body.style.overflow='hidden';show(Math.max(0,anchors.indexOf(anchor)));}
function close(){viewer.classList.remove('on');document.body.style.overflow='';img.removeAttribute('src');}
document.querySelectorAll('a.zoom').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();open(a.closest('section.tab'),a);}));
stage.addEventListener('wheel',e=>{e.preventDefault();const r=stage.getBoundingClientRect();
  zoomAt(e.clientX-r.left,e.clientY-r.top,Math.exp(-e.deltaY*(e.deltaMode===1?0.05:0.0015)));},{passive:false});
stage.addEventListener('pointerdown',e=>{stage.setPointerCapture(e.pointerId);ptrs.set(e.pointerId,[e.clientX,e.clientY]);
  stage.classList.add('drag');if(ptrs.size===2){const [p,q]=[...ptrs.values()];pinch=Math.hypot(p[0]-q[0],p[1]-q[1]);}});
stage.addEventListener('pointermove',e=>{if(!ptrs.has(e.pointerId))return;const prev=ptrs.get(e.pointerId);
  ptrs.set(e.pointerId,[e.clientX,e.clientY]);
  if(ptrs.size===1){tx+=e.clientX-prev[0];ty+=e.clientY-prev[1];apply();}
  else if(ptrs.size===2&&pinch){const [p,q]=[...ptrs.values()];const d=Math.hypot(p[0]-q[0],p[1]-q[1]);
    const r=stage.getBoundingClientRect();zoomAt((p[0]+q[0])/2-r.left,(p[1]+q[1])/2-r.top,d/pinch);pinch=d;}});
const up=e=>{ptrs.delete(e.pointerId);if(ptrs.size<2)pinch=null;if(!ptrs.size)stage.classList.remove('drag');};
stage.addEventListener('pointerup',up);stage.addEventListener('pointercancel',up);
stage.addEventListener('dblclick',e=>{const r=stage.getBoundingClientRect();
  if(s>fitScale*1.05)fit();else zoomAt(e.clientX-r.left,e.clientY-r.top,1/s);});
document.getElementById('vbar').addEventListener('click',e=>{const act=e.target.dataset&&e.target.dataset.act;if(!act)return;
  const [cx,cy]=center();({prev:()=>show(idx-1),next:()=>show(idx+1),in:()=>zoomAt(cx,cy,1.25),out:()=>zoomAt(cx,cy,0.8),
    fit:fit,one:()=>zoomAt(cx,cy,1/s),close:close})[act]();});
document.addEventListener('keydown',e=>{if(!viewer.classList.contains('on'))return;const [cx,cy]=center();
  const k={Escape:close,ArrowLeft:()=>show(idx-1),ArrowRight:()=>show(idx+1),'+':()=>zoomAt(cx,cy,1.25),'=':()=>zoomAt(cx,cy,1.25),
    '-':()=>zoomAt(cx,cy,0.8),'0':fit,'1':()=>zoomAt(cx,cy,1/s)}[e.key];if(k){e.preventDefault();k();}});
window.addEventListener('resize',()=>{if(viewer.classList.contains('on'))fit();});
})();
"""


def main():
    args = parse_args()
    out_dir = args.output.parent
    thumbs = out_dir / f"{args.output.stem}_thumbs"
    if not args.embed:
        thumbs.mkdir(parents=True, exist_ok=True)
    tab_buttons, sections = [], []
    model_lists = [list(json.loads((Path(run) / "models.json").read_text(encoding="utf-8"))["titles"])
                   if (Path(run) / "models.json").exists() else [] for _, run, _ in args.section]
    shared = bool(args.pairs) and len(args.section) > 1 and all(m == model_lists[0] and m for m in model_lists)
    shared_table = pair_table(pair_rows(Path(args.section[0][1]).resolve(), args.pairs)) if shared else ""
    for number, (title, run, intro) in enumerate(args.section):
        run = Path(run).resolve()
        meta = json.loads((run / "models.json").read_text(encoding="utf-8")) if (run / "models.json").exists() else {}
        families = meta.get("families", [])
        entries = read_entries(run)
        cards, groups = [], []
        for image, caption in entries:
            source = run / image
            split, group = group_of(image)
            if group not in groups:
                groups.append(group)
            name = f"{run.name}_{source.parent.name}_{source.stem}" if group != "overview" else f"{run.name}_{source.stem}"
            if args.embed:
                small, full = embedded(source, args.thumb_width, args.full_quality)
                href = "#"
            else:
                thumb = thumbs / f"{number}_{source.parent.name if group != 'overview' else 'x'}_{source.stem}.jpg"
                thumbnail(source, thumb, args.thumb_width)
                full = href = os.path.relpath(source, out_dir)
                small = os.path.relpath(thumb, out_dir)
            label = group_label(group, families)
            cards.append(
                f"<figure class='card{' wide' if group == 'overview' else ''}' data-split='{split}' data-group='{group}'>"
                f"<a class=zoom href='{html.escape(href)}' data-full='{html.escape(full)}' "
                f"data-name='{html.escape(name)}.webp' data-cap='{html.escape(caption)}'>"
                f"<img src='{html.escape(small)}' loading=lazy alt=''></a><figcaption><span class=chip>{split}</span>"
                f"<span class=chip>{html.escape(label)}</span>{html.escape(caption)}</figcaption></figure>")
        filters = ("<div class=filters><span class=label>数据：</span>"
                   + "".join(f"<button data-key=split data-value='{v}' class='{'on' if v == 'all' else ''}'>{t}</button>"
                             for v, t in (("all", "全部"), ("test", "测试集"), ("train", "训练集")))
                   + "<span class=label>场景组：</span><button data-key=group data-value=all class=on>全部</button>"
                   + "".join(f"<button data-key=group data-value='{g}'>{html.escape(group_label(g, families))}</button>"
                             for g in groups) + "</div>")
        if args.embed:
            summary = (run / "summary.md").read_text(encoding="utf-8") if (run / "summary.md").exists() else ""
            links = (f"<details class=summary><summary>summary.md（池内 80 / 160 个随机场景的统计）</summary>"
                     f"<pre>{html.escape(summary)}</pre></details>") if summary else ""
        else:
            links = (f"<p class=links><a href='{html.escape(os.path.relpath(run / 'summary.md', out_dir))}'>summary.md</a>"
                     f"<a href='{html.escape(os.path.relpath(run / 'index.html', out_dir))}'>原 index.html</a></p>")
        table = pair_table(pair_rows(run, args.pairs)) if args.pairs and meta and not shared else ""
        tab_id = f"s{number + 1}"
        tab_buttons.append(f"<button data-tab='{tab_id}'>{html.escape(title)}</button>")
        sections.append(f"<section class=tab id='{tab_id}'><h2>{html.escape(title)}</h2><p>{html.escape(intro)}</p>{links}"
                        f"{table}{filters}<div class=grid>{''.join(cards)}</div></section>")
        print(f"[{tab_id}] {title}: {len(cards)} figures from {run}", flush=True)
    page = (f"<!doctype html><html lang=zh-CN><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(args.title)}</title><style>{CSS}</style><body data-embed='{int(args.embed)}'>"
            f"<header><h1>{html.escape(args.title)}</h1>"
            f"<p class=intro>{html.escape(args.intro)}</p><nav class=tabs>{''.join(tab_buttons)}</nav></header>"
            f"<main>{shared_table}{''.join(sections)}</main>"
            "<div id=viewer><div id=stage><img id=vimg alt=''></div><div id=vbar>"
            "<button data-act=prev title='上一张 (←)'>‹</button><span id=vpos></span><button data-act=next title='下一张 (→)'>›</button>"
            "<button data-act=out title='缩小 (-)'>−</button><span id=vzoom></span><button data-act=in title='放大 (+)'>+</button>"
            "<button data-act=fit title='适应窗口 (0)'>适应</button><button data-act=one title='原始大小 (1)'>1:1</button>"
            "<span id=vcap></span><span id=vhelp>滚轮/双指缩放 · 拖动平移 · 双击 适应/1:1 · ←→ 切换</span>"
            + ("<a id=vopen>下载原图</a>" if args.embed else "<a id=vopen target=_blank>原图</a>")
            + "<button data-act=close title='关闭 (Esc)'>✕</button></div></div>"
            f"<script>{JS}</script></body></html>\n")
    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)"
          + ("" if args.embed else f", thumbnails in {thumbs}"), flush=True)


if __name__ == "__main__":
    main()
