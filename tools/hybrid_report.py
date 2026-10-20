"""Summarize a frozen pilot; never promotes structural checks to acceptance."""
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image,ImageDraw
from image2ppt_vision.storage import digest,read_json,write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True)
    args=parser.parse_args();root=Path(args.pilot).resolve();result=read_json(root/'result.json')
    preview=Path(result['preview']);baseline=read_json(root/'baseline.json')
    old_deck=Path(next(s['source'] for s in baseline['sources'] if s['copy']=='baseline/qwen-baseline.pptx'))
    canvas=Image.new('RGB',(1920,1140),'white');draw=ImageDraw.Draw(canvas);rows=[]
    for i,(stem,oldnum) in enumerate([('03',2),('05',4),('09',8)]):
        paths=[root/'inputs/source'/f'{stem}.png',old_deck.parent/'preview'/f'page-{oldnum:03}.png',preview/f'page-{i+1:03}.png']
        for j,path in enumerate(paths):
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((640,350));canvas.paste(im,(j*640,i*380+25))
            draw.text((j*640+8,i*380+5),stem+' / '+['source','baseline Qwen','local fix'][j],fill='black')
        before=read_json(root/'baseline'/stem/'reconstruction.json')
        after=read_json(root/'local-fix/pages'/stem/'reconstruction.json')
        rows.append({'source':stem+'.png','native_before':before['native_text'],'native_after':after['native_text'],
                     'groups_before':before['groups'],'groups_after':after['groups'],
                     'new_cleanup_blocks':[t['id'] for t in after['typography'] if t.get('reason')=='cleanup_outside_text_bounds'],
                     'status':'REVIEW_REQUIRED','visual_acceptance':False})
    canvas.save(root/'comparison.png')
    shutil.copyfile(result['pptx'],root/'三页本地修复试点.pptx')
    write_json(root/'pilot-report.json',{'status':'REVIEW_REQUIRED','pages':rows,'source_snapshots_verified':
        all(digest(root/s['copy'])==s['sha256'] for s in baseline['sources']),
        'pptx_sha256':digest(root/'三页本地修复试点.pptx'),'preview_origin':result['preview_origin'],
        'real_layer_model':'NOT_RUN','reason':'No VISION_LAYER_API_KEY configured',
        'expanded_to_14':False,'human_acceptance':False,'powerpoint_checked':False,'wps_checked':False})
    lines=['# 三页本地修复试点','',
           '这是本地文字归属、遮挡和清理保护的修复版本，尚未使用真实拆层模型。原图、OCR、计划和两个对照 PPT 已快照保存。',
           '', '| 原图 | 原生文本：原版 → 修复版 | 状态 |','|---|---:|---|']
    lines += [f"| {r['source']} | {r['native_before']} → {r['native_after']} | 待审核 |" for r in rows]
    lines += ['', '交付：`三页本地修复试点.pptx`；`comparison.png` 为原图/千问基线/修复版对照，预览来自实际 PPTX 回读。',
              '', '新增保护导致 03 页两条、09 页一条文字保留为图片；不能以外观改善认定可编辑性已通过。',
              '', '真实 Ming 拆层、背景补全、真实页模块移动后的背景验收尚未完成，因此未扩展到 14 页。PowerPoint/WPS 实际编辑和人工验收均未执行。',
              '', '模型候选导入和生成命令、LayerManifest 审核方式见 services/vision/HYBRID.md。']
    (root/'试点说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__': main()
