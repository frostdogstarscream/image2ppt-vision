"""Measure model-proposed typography, fills, text ink and box extents from the original pixels.

This is source-only calibration, not semantic correction or acceptance. Refused
measurements leave the model proposal unchanged and remain explicit in reports.
"""
from copy import deepcopy
from pathlib import Path
import math,shutil
import cv2
import numpy as np
from PIL import Image
from .storage import read_json,write_json,digest
from .typography import estimate_text, _background, _component_colors
from .renderer_calibration import origin_correction
from .environment import collect_environment, policy_flags

def _contains(a,b):
    return a[0]<=b[0] and a[1]<=b[1] and a[0]+a[2]>=b[0]+b[2] and a[1]+a[3]>=b[1]+b[3]

def _source_contrast(pixels,node):
    """Diagnostic only: proposed text against a reliable local source surface."""
    x,y,w,h=node.get('source_text_bbox',node['bbox'])
    l,t=max(0,math.floor(x)-3),max(0,math.floor(y)-3)
    r,b=min(pixels.shape[1],math.ceil(x+w)+3),min(pixels.shape[0],math.ceil(y+h)+3)
    if min(r-l,b-t)<4:return None
    surface,diagnostic=_background(pixels[t:b,l:r])
    if not diagnostic['reliable']:return None
    background=np.median(surface.reshape(-1,3),axis=0)
    colors=[run.get('color',node['color']) for run in node.get('runs',[])] or [node['color']]
    distances=[float(np.max(np.abs(np.array([int(c[i:i+2],16) for i in (0,2,4)])-background))) for c in colors]
    return {'minimum_rgb_distance':round(min(distances),3),'low_contrast_risk':min(distances)<18,
            'basis':'source_background_estimate_not_actual_pptx_render'}

def _rgb(value):
    if isinstance(value, str):
        v = value.lstrip("#")
        if len(v) == 6:
            try:
                return np.array([int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)], float)
            except ValueError:
                return None
    return None


def _hx(v):
    return ''.join('%02X' % int(round(c)) for c in np.clip(v, 0, 255))


def _ink_legacy(pixels, box):
    """Old median-based ink reading, retained as an explicit fallback switch."""
    x, y, w, h = box
    l, t = max(0, math.floor(x)), max(0, math.floor(y))
    r, b = min(pixels.shape[1], math.ceil(x + w)), min(pixels.shape[0], math.ceil(y + h))
    if min(r - l, b - t) < 4:
        return None
    flat = pixels[t:b, l:r].reshape(-1, 3).astype(float)
    if flat.shape[0] < 40:
        return None
    background = np.median(flat, axis=0)
    distance = np.linalg.norm(flat - background, axis=1)
    keep = distance >= np.percentile(distance, 92)
    if keep.sum() < 20:
        return None
    separation = float(distance[keep].mean())
    if separation < 25:
        return None
    return {'color': _hx(flat[keep].mean(axis=0)), 'separation': round(separation, 1), 'sample_pixels': int(keep.sum())}


def _ink_measured(pixels, box, *, text=None):
    """Glyph colour with evidence that the selected pixels are text.

    The local linear surface is used as a reference even when its ``reliable``
    flag is false: a gradient bar defeats the smooth-background assumption but
    its glyphs are still visible, and page05 depends on measuring them. What the
    old median-only helper lacked was evidence that the extreme pixels form
    glyph-like components rather than a band of background texture.
    """
    diagnostics = {'method': 'surface_delta_components', 'reason': None,
                   'background': None, 'accepted_components': 0,
                   'glyph_area_fraction': None, 'coverage': None}
    x, y, w, h = box
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w < 3 or h < 3:
        diagnostics['reason'] = 'invalid_bbox'
        return None, diagnostics
    pad = max(2, min(8, int(round(min(w, h) * .25))))
    l = max(0, math.floor(x) - pad); t = max(0, math.floor(y) - pad)
    r = min(pixels.shape[1], math.ceil(x + w) + pad)
    b = min(pixels.shape[0], math.ceil(y + h) + pad)
    if r - l < 3 or b - t < 3 or (r - l) * (b - t) < 64:
        diagnostics['reason'] = 'crop_too_small'
        return None, diagnostics
    crop = pixels[t:b, l:r].astype(float)
    # A tight OCR box plus padding can put neighbouring lines on the border
    # ring, where the default 2px sample would fit their glyphs as background.
    # Fit on colour-like pixels when enough of the crop is one surface.
    median = np.median(crop.reshape(-1, 3), axis=0)
    sample = np.max(np.abs(crop - median), axis=2) < 32
    surface, background = _background(crop, sample if sample.mean() >= .25 else None)
    diagnostics['background'] = background
    delta = np.max(np.abs(crop - surface), axis=2)
    threshold = max(10.0, float(background.get('border_p80_error', 0)) * 2.5 + 3.0)
    raw = delta > threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    x0, y0, x1, y1 = x - l, y - t, x + w - l, y + h - t
    accepted = np.zeros_like(raw)
    accepted_count = 0
    saw_separator = False
    reason = None
    for index in range(1, count):
        left, top, width, height, area = stats[index]
        # A separator or a long texture band is not a glyph. Checked before the
        # glyph-size floors, otherwise a 3px rule never reaches this branch.
        # A glyph line can carry a rule above or below it; only the rule is
        # skipped, and the separator reason is used when nothing else remains.
        if ((width >= max(12, .9 * w) and height <= max(3, .08 * h)) or
                (height >= max(12, .9 * h) and width <= max(3, .08 * w))):
            saw_separator = True
            continue
        if area < 4 or width < 2 or height < max(3, .12 * h) or height > .95 * h:
            continue
        overlap_x = max(0, min(left + width, x1) - max(left, x0))
        overlap_y = max(0, min(top + height, y1) - max(top, y0))
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        if (overlap_x * overlap_y) / float(width * height) < .35 or overlap_y / float(height) < .5:
            continue
        accepted[labels == index] = True
        accepted_count += 1
    diagnostics['accepted_components'] = accepted_count
    if reason is None:
        accepted_pixels = int(accepted.sum())
        area_fraction = accepted_pixels / float(w * h)
        diagnostics['glyph_area_fraction'] = round(area_fraction, 4)
        if accepted_pixels < max(8, int(w * h * .005)):
            reason = 'separator_line' if saw_separator else 'no_glyph_evidence'
        elif area_fraction > .60:
            reason = 'foreground_not_sparse_text'
        else:
            ys, xs = np.where(accepted)
            coverage = (float(xs.max() - xs.min() + 1) / w) if len(xs) else 0.0
            diagnostics['coverage'] = round(coverage, 4)
            if coverage < .12:
                reason = 'glyph_coverage_too_low'
            elif accepted_count == 1:
                width = int(xs.max() - xs.min() + 1); height = int(ys.max() - ys.min() + 1)
                if not .25 <= height / float(width) <= 4 or height < .15 * h:
                    reason = 'not_text_like'
            else:
                widest = max(stats[index][2] for index in range(1, count) if (labels == index)[accepted].any())
                if widest >= .9 * w:
                    reason = 'not_text_like'
    if reason is not None:
        diagnostics['reason'] = reason
        return None, diagnostics
    groups = _component_colors(crop, delta, accepted)
    if not groups:
        diagnostics['reason'] = 'foreground_color_uncertain'
        return None, diagnostics
    total = sum(group['weight'] for group in groups)
    if (len(groups) > 1 and groups[1]['weight'] / float(total) >= .25 and
            float(np.max(np.abs(groups[0]['color'] - groups[1]['color']))) > 40):
        diagnostics['reason'] = 'mixed_colour_glyphs'
        return None, diagnostics
    core = accepted & (delta >= np.percentile(delta[accepted], 70))
    if core.sum() < max(3, int(accepted.sum() * .10)):
        diagnostics['reason'] = 'insufficient_core_pixels'
        return None, diagnostics
    separation = float(delta[core].mean())
    if separation < 25:
        diagnostics['reason'] = 'low_ink_separation'
        return None, diagnostics
    return {'color': _hx(groups[0]['color']), 'separation': round(separation, 1),
            'sample_pixels': int(core.sum()), 'component_count': accepted_count,
            'coverage': diagnostics['coverage'],
            'glyph_area_fraction': diagnostics['glyph_area_fraction']}, diagnostics


def _ink(pixels, box):
    if policy_flags()['glyph_ink_evidence']:
        return _ink_measured(pixels, box)[0]
    return _ink_legacy(pixels, box)


def _component_fraction(mask):
    """Largest dominant-colour component share, used to reject speckle fills."""
    if not mask.any():
        return 0.0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / mask.sum())


def _surface_measurement(pixels, node, nodes):
    """Collect the pixels a shape covers, excluding whatever is drawn on top."""
    if node['geometry'] not in ('rect','roundRect','ellipse') or node['fill']=='none':
        return None
    h,w=pixels.shape[:2];x,y,bw,bh=node['bbox'];l,t,r,b=math.floor(x),math.floor(y),math.ceil(x+bw),math.ceil(y+bh)
    l,t,r,b=max(0,l),max(0,t),min(w,r),min(h,b)
    if r-l<1 or b-t<1:return None
    crop=pixels[t:b,l:r];mask=np.ones(crop.shape[:2],bool)
    margin=max(2,min(5,int(min(bw,bh)*.08)))
    mask[:margin]=False;mask[-margin:]=False;mask[:,:margin]=False;mask[:,-margin:]=False
    if node['geometry']=='ellipse':
        yy,xx=np.mgrid[:b-t,:r-l];mask &= ((xx-(r-l)/2)/max(1,(r-l)/2-margin))**2+((yy-(b-t)/2)/max(1,(b-t)/2-margin))**2<1
    for other in nodes:
        if other['id']==node['id']:continue
        box=other.get('source_text_bbox',other['bbox']) if other['type']=='text' else other['bbox']
        if other['type']=='shape' and (not _contains(node['bbox'],box) or box[2]*box[3]>=bw*bh*.94):continue
        a,c=max(l,math.floor(box[0])-2),min(r,math.ceil(box[0]+box[2])+2);d,e=max(t,math.floor(box[1])-2),min(b,math.ceil(box[1]+box[3])+2)
        if c>a and e>d:mask[d-t:e-t,a-l:c-l]=False
    values=crop[mask]
    if len(values)<max(30,bw*bh*.08):return None
    yy,xx=np.mgrid[:b-t,:r-l]
    return {'crop':crop,'mask':mask,'values':values,
            'u':(xx[mask]/max(1,(r-l)-1)).astype(float),
            'v':(yy[mask]/max(1,(b-t)-1)).astype(float),
            'bbox':[l,t,r-l,b-t]}


def _fill_measured(pixels,node,nodes):
    """Return the fitted surface and the measurement it was fitted on."""
    measurement=_surface_measurement(pixels,node,nodes)
    if measurement is None:return None,None
    crop=measurement['crop'];mask=measurement['mask'];values=measurement['values']
    declared_gradient=node.get('fill_gradient') is not None
    # Flat reading: one dominant colour filling the region.
    q=(values//16).astype(int);bins=q[:,0]*256+q[:,1]*16+q[:,2];mode=np.bincount(bins,minlength=4096).argmax();selected=bins==mode
    flat=None;weak=None
    if selected.mean()>=.55:
        color=np.median(values[selected],axis=0);near=np.max(abs(values.astype(float)-color),axis=1)<12
        # A bar with a soft gradient or a border keeps one colour dominant over
        # most of its area without filling 75% of a tight band. That is still
        # more honest than a strong declaration the pixels contradict, but it is
        # only used when no plane fits.
        if near.mean()>=.55:
            support=np.zeros(mask.shape,bool);support[mask]=near
            entry={'color':_hx(np.median(values[near],axis=0)),'sample_pixels':len(values),
                   'dominant_fraction':round(float(near.mean()),4),
                   'component_fraction':round(_component_fraction(support),4)}
            if near.mean()>=.75:flat=entry
            else:weak=entry
    if flat is not None and not declared_gradient:
        return flat,measurement
    # Gradient reading: fit a plane on the masked grid, keep the stronger axis.
    yy,xx=np.mgrid[:measurement['bbox'][3],:measurement['bbox'][2]]
    design=np.stack([np.ones_like(xx),xx/max(1,measurement['bbox'][2]-1),yy/max(1,measurement['bbox'][3]-1)],-1)[mask]
    series=values.astype(float)
    coef=np.linalg.lstsq(design,series,rcond=None)[0]
    for _ in range(3):
        selected=np.max(np.abs(series-design@coef),axis=1)<10
        if selected.sum()<len(series)*.2:break
        coef=np.linalg.lstsq(design[selected],series[selected],rcond=None)[0]
    else:
        error=np.max(np.abs(series-design@coef),axis=1)
        if np.percentile(error,95)<=24:
            axis=0 if np.linalg.norm(coef[1])>=np.linalg.norm(coef[2]) else 1
            if axis==0:first,last,angle=coef[0]+coef[2]*.5,coef[0]+coef[1]+coef[2]*.5,0
            else:first,last,angle=coef[0]+coef[1]*.5,coef[0]+coef[2]+coef[1]*.5,90
            # Projection is an extra approximation: measure it before claiming it.
            axis_position=measurement['u'] if axis==0 else measurement['v']
            projected=first+(last-first)*axis_position[:,None]
            residual=np.max(np.abs(series-projected),axis=1)
            projection={'mean':round(float(residual.mean()),3),
                        'p95':round(float(np.percentile(residual,95)),3),
                        'fraction_over_12':round(float((residual>12).mean()),4)}
            measurement['projection']=projection
            if (policy_flags()['fill_projection_guard'] and
                    not (projection['mean']<=12 and projection['p95']<=24 and projection['fraction_over_12']<=.35)):
                measurement['projection_rejected']=projection
                # A failed projection is not evidence for the weak flat reading;
                # only a strong flat survives as an honest fallback.
                return flat,measurement
            fit={'color':_hx(first),'sample_pixels':len(values),'dominant_fraction':round(float(selected.mean()),4),'angle_deg':angle,
                 'projection':projection}
            if np.max(np.abs(first-last))>4:
                fit['gradient']={'angle_deg':angle,'stops':[{'offset':0,'color':_hx(first)},{'offset':100000,'color':_hx(last)}]}
            elif flat is not None:
                return flat,measurement
            return fit,measurement
    # A declared gradient is a claim, not a constraint. When no plane fits,
    # the flat reading is the honest fallback for a bar that is nearly one
    # colour with texture the plane cannot capture.
    return (flat or weak),measurement


def _fill(pixels,node,nodes):
    """Fit the surface a shape paints, excluding whatever is drawn on top of it."""
    return _fill_measured(pixels,node,nodes)[0]


def _gradient_stops(gradient):
    """Validate a declared gradient before sampling it. None means unmeasurable."""
    if not isinstance(gradient,dict):return None
    stops=gradient.get('stops');angle=gradient.get('angle_deg')
    if not isinstance(stops,list) or not 2<=len(stops)<=8:return None
    if isinstance(angle,bool) or not isinstance(angle,(int,float)) or not math.isfinite(angle):return None
    offsets=[];colors=[]
    for stop in stops:
        if not isinstance(stop,dict):return None
        offset=stop.get('offset');color=_rgb(stop.get('color'))
        if color is None or isinstance(offset,bool) or not isinstance(offset,(int,float)) or not math.isfinite(offset):
            return None
        offsets.append(float(offset));colors.append(color)
    if offsets[0]!=0 or offsets[-1]!=100000:return None
    if any(offsets[index]>=offsets[index+1] for index in range(len(offsets)-1)):return None
    return np.array(offsets)/100000.0,np.stack(colors),float(angle)


def _gradient_position(u,v,angle,interpretation):
    radians=math.radians(angle);cos,sin=math.cos(radians),math.sin(radians)
    if interpretation=='projected':
        span=abs(cos)+abs(sin)
        t=.5+((u-.5)*cos+(v-.5)*sin)/max(span,1e-9)
    else:
        t=.5+(u-.5)*cos+(v-.5)*sin
    return np.clip(t,0,1)


def _paint_samples(node,u,v,*,interpretation='projected'):
    """Evaluate a declared colour or gradient at normalised positions."""
    gradient=node.get('fill_gradient')
    if gradient is not None:
        parsed=_gradient_stops(gradient)
        if parsed is None:return None
        offsets,colors,angle=parsed
        t=_gradient_position(u,v,angle,interpretation)
        painted=np.stack([np.interp(t,offsets,colors[:,channel]) for channel in range(3)],axis=1)
        return painted,{'kind':'gradient','angle_deg':angle,'stops':int(len(offsets)),
                        'interpretation':interpretation}
    fill=_rgb(node.get('fill'))
    if fill is None:return None
    return np.tile(fill,(len(u),1)),{'kind':'solid','interpretation':None}


def _paint_error(values,painted):
    if values is None or painted is None:return None
    error=np.max(np.abs(values-painted),axis=1)
    return {'sample_pixels':int(len(error)),'mean':round(float(error.mean()),3),
            'p95':round(float(np.percentile(error,95)),3),
            'fraction_over_12':round(float((error>12).mean()),4)}


def _declared_error(node,measurement):
    """Compare the complete declaration with the source on the same pixels."""
    gradient=node.get('fill_gradient') is not None
    if not gradient and _rgb(node.get('fill')) is None:
        return {'measurable':False,'reason':'declared_missing','consistent':False,
                'all_disagree':False,'interpretations':[]}
    interpretations=('projected','centered') if gradient else ('projected',)
    results=[]
    for interpretation in interpretations:
        painted=_paint_samples(node,measurement['u'],measurement['v'],interpretation=interpretation)
        if painted is None:
            return {'measurable':False,'reason':'declared_unmeasurable','consistent':False,
                    'all_disagree':False,'interpretations':[]}
        error=_paint_error(measurement['values'],painted[0])
        results.append({'interpretation':interpretation,'error':error,
                        'matches':error['mean']<=12 and error['p95']<=24})
    matches=[item['matches'] for item in results]
    angle_verified=(not gradient or _gradient_stops(node['fill_gradient'])[2] % 90 == 0)
    return {'measurable':True,'consistent':all(matches),'all_disagree':not any(matches),
            'best':min(results,key=lambda item:item['error']['mean']),
            'interpretations':results,
            'angle_semantics':'not_applicable' if not gradient else
                              ('axis_verified' if angle_verified else 'non_axis_assumed')}


def _candidate_error(fitted,measurement):
    node={'fill':fitted['color'],'fill_gradient':fitted.get('gradient')}
    painted=_paint_samples(node,measurement['u'],measurement['v'])
    return None if painted is None else _paint_error(measurement['values'],painted[0])


def _fill_decision(node,fitted,measurement):
    """Decide whether the measured surface may replace the declaration."""
    decision={'apply':False,'decision':'candidate_not_supported','reason':None,
              'candidate_kind':None,'declared_error':None,'candidate_error':None,
              'legacy_drift':None,'angle_semantics':None}
    declared=_declared_error(node,measurement)
    decision['angle_semantics']=declared.get('angle_semantics')
    if not declared['measurable'] and declared.get('reason')=='declared_missing':
        decision.update(apply=True,decision='declared_missing')
        return decision
    if not declared['measurable']:
        decision.update(decision='declared_unmeasurable',reason=declared.get('reason'))
        return decision
    decision['declared_error']=declared['best']['error']
    if declared['consistent']:
        decision.update(decision='declared_matches_source')
        return decision
    if len(declared['interpretations'])>1 and not declared['all_disagree']:
        decision.update(decision='angle_semantics_ambiguous')
        return decision
    candidate=_candidate_error(fitted,measurement)
    if candidate is None:
        decision.update(reason='candidate_unmeasurable')
        return decision
    decision['candidate_error']=candidate
    kind=('gradient' if 'gradient' in fitted else
          'flat_strong' if fitted.get('dominant_fraction',0)>=.75 else 'flat_weak')
    decision['candidate_kind']=kind
    declared_color=_rgb(node.get('fill'))
    if declared_color is not None:
        decision['legacy_drift']=round(float(np.abs(declared_color-np.array(
            [int(fitted['color'][index:index+2],16) for index in (0,2,4)])).mean()),3)
    improvement=declared['best']['error']['mean']-candidate['mean']
    if kind=='flat_weak':
        apply=(declared['best']['error']['mean']>=40 and
               candidate['mean']<=.75*declared['best']['error']['mean'] and
               fitted.get('component_fraction',0)>=.7)
        reason=None if apply else 'weak_candidate_not_coherent_with_declaration_error'
    else:
        apply=((candidate['mean']<=12 and candidate['p95']<=24) or
               (candidate['mean']<=24 and candidate['mean']<=.6*declared['best']['error']['mean']))
        apply=apply and improvement>=4
        reason=None if apply else 'candidate_not_a_clear_improvement'
    decision.update(apply=apply,decision='applied' if apply else 'candidate_not_supported',reason=reason)
    return decision

def _box_inside(outer,inner):
    ox,oy,ow,oh=outer;ix,iy,iw,ih=inner
    if iw<=0 or ih<=0:return False
    return (ox-4<=ix and oy-4<=iy and ox+ow+4>=ix+iw and oy+oh+4>=iy+ih
            and (iw*ih)/float(ow*oh)>=.75)


def _edge_coverage(crop,colour,tol):
    mask=np.max(np.abs(crop-colour),axis=2)<=tol
    frame=np.zeros_like(mask);frame[:3]=True;frame[-3:]=True;frame[:,:3]=True;frame[:,-3:]=True
    return float(mask[frame].mean())


def _tighten(pixels,node,nodes):
    """Follow a declared border to the element's real extent, or return None.

    The proposal is kept unless the source shows the border somewhere else and
    that somewhere is a compact, filled region. A box whose border already runs
    along its own edge is where the shape is; there is nothing to tighten.
    """
    colour=_rgb((node.get('line') or {}).get('color'))
    if colour is None or node.get('geometry') not in ('rect','roundRect'):return None
    h,w=pixels.shape[:2];x,y,bw,bh=node['bbox'];l,t,r,b=math.floor(x),math.floor(y),math.ceil(x+bw),math.ceil(y+bh)
    l,t,r,b=max(0,l),max(0,t),min(w,r),min(h,b)
    if r-l<20 or b-t<12:return None
    crop=pixels[t:b,l:r].astype(float)
    if max(_edge_coverage(crop,colour,tol) for tol in (12,18,24))>=.18:return None
    fill=_rgb(node.get('fill'))
    for tol in (12,18,24):
        mask=np.max(np.abs(crop-colour),axis=2)<=tol
        if int(mask.sum())<120:continue
        ys,xs=np.where(mask);nw,nh=int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1)
        ratio=(nw*nh)/float((r-l)*(b-t))
        if not .03<=ratio<=.55 or nw<40 or nh<16:continue
        # A frame, not a fill: text antialiasing can pass through the border colour.
        inner=mask[ys.min()+4:ys.max()-3,xs.min()+4:xs.max()-3]
        if inner.size and inner.mean()>.15:continue
        box=[l+int(xs.min()),t+int(ys.min()),nw,nh]
        # If the found box is a text node's box, the "border" was glyph antialiasing.
        if any(o['type']=='text' and _box_inside(box,o.get('source_text_bbox',o['bbox'])) for o in nodes if o['id']!=node['id']):continue
        # The interior must actually carry the declared fill.
        if fill is not None:
            inside=crop[ys.min()+3:ys.max()-2,xs.min()+3:xs.max()-2].reshape(-1,3)
            if inside.size and (np.max(np.abs(inside-fill),axis=1)<24).mean()<.35:continue
        return box
    return None


TYPOGRAPHY_THRESHOLDS = {'shape_overlap': .58, 'dimension_error': .12}
POLICY_LEVELS = ('diagnostic', 'color', 'typography')
GEOMETRY_POLICIES = ('diagnostic', 'measured')


def resolve_typography_policy(source_typography, typography_policy):
    """Map the legacy boolean onto the explicit policy scale.

    ``off`` is internal only: it preserves the pre-policy behaviour where text
    nodes were not measured at all.
    """
    if typography_policy is None:
        return 'typography' if source_typography else 'off'
    if typography_policy not in POLICY_LEVELS:
        raise ValueError('invalid typography_policy')
    return typography_policy


def _script_kind(text):
    cjk = any('㐀' <= character <= '鿿' or '豈' <= character <= '﫿'
              for character in text)
    latin = any(character.isascii() and (character.isalpha() or character.isdigit())
                for character in text)
    return 'mixed' if cjk and latin else 'cjk' if cjk else 'latin'


def _runs_colours(node):
    return {str(run.get('color', '')).lstrip('#').upper()
            for run in node.get('runs') or [] if run.get('color')}


def _set_source_colour(node, colour, row, basis):
    node['color'] = colour.lstrip('#')
    for run in node.get('runs') or []:
        run['color'] = node['color']
    row['source_color_applied'] = True
    row['source_color_basis'] = basis
    row.setdefault('applied_fields', []).append('color')


def _colour_drift(node, colour):
    declared = _rgb(node.get('color'))
    if declared is None or colour is None:
        return None
    colour = colour.lstrip('#')
    return float(np.abs(declared - np.array(
        [int(colour[index:index + 2], 16) for index in (0, 2, 4)])).mean())


def _colour_measurement(estimate, pixels, box, policies):
    """Prefer the typography estimate, fall back to glyph-only evidence."""
    diagnostics = estimate.get('diagnostics') or {}
    background = diagnostics.get('background') or {}
    if (estimate.get('color') and diagnostics.get('foreground_color_count') == 1 and
            (estimate.get('native_eligible') or background.get('reliable'))):
        return estimate['color'], 'typography_estimate', None
    if policies['glyph_ink_evidence']:
        ink, ink_diagnostics = _ink_measured(pixels, box)
    else:
        ink, ink_diagnostics = _ink_legacy(pixels, box), {'method': 'legacy_median'}
    return (ink or {}).get('color'), ('glyph_ink' if ink else None), ink_diagnostics


def _apply_typography(node, estimate, row, *, geometry_policy, source_size, target_size,
                      scale, renderer_version):
    """Apply measured typography with per-field guards and a preserved runs list."""
    runs = node.get('runs') or []
    if len(runs) > 1 and len(_runs_colours(node)) > 1:
        row['reason'] = 'mixed_runs_protected'
        row['guards'] = {'runs': 'protected_mixed_colours'}
        return False
    fontdiag = estimate['diagnostics']['font']
    x, y, bw, bh = estimate['bbox']
    size = estimate['font_size_px']
    origin = list(fontdiag['render_origin'])
    correction = origin_correction(estimate, [source_size[0], source_size[1]],
                                   [target_size[0], target_size[1]],
                                   renderer_version=renderer_version or 'unspecified')
    glyph = fontdiag.get('glyph_bbox')
    if glyph:
        # Font matching optimises local overlap and can shift a long line by
        # many pixels. Restore the measured visible left/top edges instead of
        # exporting that search translation as layout.
        origin = [x - glyph[0], y - glyph[1]]
        row['origin_policy'] = 'measured_ink_edges_minus_font_glyph_bearings'
    tx = max(0, float(origin[0]))
    ty = max(0, float(origin[1]) + correction['dy'])
    row['before'] = {key: deepcopy(node.get(key)) for key in ('bbox', 'font', 'bold', 'color', 'runs')}
    size_pt = round(size * scale * .75, 4)
    measured_family = estimate['font_family']
    if _script_kind(node.get('text', '')) == 'mixed':
        # A single fitted family cannot describe both scripts. Keep the model's
        # Latin choice and apply the measured family only to the East Asian side.
        latin = deepcopy((node.get('font') or {}).get('latin') or measured_family)
        node['font'] = {'latin': latin, 'ea': measured_family, 'size_pt': size_pt}
        row['font_scope'] = 'ea_only_mixed_script'
    else:
        node['font'] = {'latin': measured_family, 'ea': measured_family, 'size_pt': size_pt}
        row['font_scope'] = 'both_sides_single_script'
    node['bold'] = estimate['bold']
    node['color'] = estimate['color'].lstrip('#')
    applied = ['font', 'size', 'weight', 'color']
    if runs:
        # Never drop authored runs; only their colour follows a single-colour measurement.
        for run in runs:
            run['color'] = node['color']
        row['runs_policy'] = 'preserved'
    elif estimate.get('runs'):
        node['runs'] = [{**run, 'color': run['color'].lstrip('#')} for run in estimate['runs']]
        row['runs_policy'] = 'measured_single_colour_runs'
    if geometry_policy == 'measured':
        node['bbox'] = [tx, ty,
                        min(source_size[0] - tx, max(bw + 3, size * len(node['text']) * .45)),
                        min(source_size[1] - ty, max(size * 1.5, bh + 4))]
        applied.append('geometry')
        row['geometry_policy'] = 'measured_ink_edges'
    else:
        row['geometry_policy'] = 'kept_proposal'
    node['align'] = 'left'
    row.update(applied=True, applied_fields=applied, renderer_origin_calibration=correction)
    return True


COLOUR_FALLBACK_REASONS = ('font_or_ocr_shape_mismatch', 'strict_source_font_fit_refused',
                           'no_usable_installed_font', 'non_smooth_or_crossing_background',
                           'font_environment_unavailable')


def _apply_colour_only(node, estimate, row, pixels, box, policies):
    """Colour repair independent of font/geometry eligibility."""
    if len(_runs_colours(node)) > 1:
        row['source_color_refused_reason'] = 'mixed_runs_protected'
        return False
    reason = (estimate.get('diagnostics') or {}).get('reason')
    if not estimate.get('native_eligible') and reason not in COLOUR_FALLBACK_REASONS:
        row['source_color_refused_reason'] = reason or 'source_measurement_refused'
        return False
    colour, basis, ink_diagnostics = _colour_measurement(estimate, pixels, box, policies)
    row['source_color'] = estimate.get('color')
    row['source_ink_measure'] = ink_diagnostics if basis == 'glyph_ink' else None
    if colour is None:
        row['source_color_refused_reason'] = (
            (ink_diagnostics or {}).get('reason') or 'no_glyph_colour_evidence')
        return False
    drift = _colour_drift(node, colour)
    if drift is None or drift > 30:
        _set_source_colour(node, colour, row, basis)
        if drift is not None:
            row['declared_color_drift'] = round(drift, 1)
        return True
    row['source_color_refused_reason'] = 'declared_color_matches_source'
    return False


def calibrate_elements(irfile,output,*,source_typography=False,typography_policy=None,
                       geometry_policy='measured',renderer_version=None,calibrate_fills=True):
    irfile=Path(irfile);output=Path(output)
    if output.exists():raise FileExistsError('calibration output must be new')
    ir=read_json(irfile);src=irfile.parent/ir['source_image']
    authoring=read_json(irfile.parent/'authoring.json')
    if authoring.get('ir_sha256')!=digest(irfile):raise ValueError('authoring IR binding changed')
    model_typography=authoring.get('model_typography',{})
    if digest(src)!=ir['source_sha256']:raise ValueError('source binding changed')
    with Image.open(src) as im:pixels=np.array(im.convert('RGB'))
    w,h=ir['canvas']['source_px'];tw,th=ir['canvas']['target_px'];scale=tw/w
    diagnostics=[];measured=0;fills=0;tightened_count=0
    original_nodes=deepcopy(ir['nodes'])
    policies=policy_flags()
    policy=resolve_typography_policy(source_typography,typography_policy)
    if geometry_policy not in GEOMETRY_POLICIES:raise ValueError('invalid geometry_policy')
    active=policy!='off'
    try:
        environment=collect_environment(ir,renderer_version=renderer_version,
                                        texts=[node['text'] for node in ir['nodes'] if node['type']=='text'])
    except Exception:
        # Evidence collection must never turn a reconstruction into a failure.
        environment={'status':'unknown','reason':'environment_probe_failed'}
    for node in ir['nodes']:
        if node['type']=='text':
            box=node.get('source_text_bbox',node['bbox'])
            try:
                estimate=estimate_text(pixels,{'id':node['id'],'text':node['text'],'bbox':box},
                                       **({'isolate_ocr_window':True} if active else {}))
            except (OSError,ValueError) as error:
                # A broken font volume is a deployment fact, not a bad proposal.
                # Keep measuring fills and visible glyph colour where possible.
                estimate={'native_eligible':False,
                          'diagnostics':{'status':'REVIEW_REQUIRED',
                                         'reason':'font_environment_unavailable',
                                         'font_environment_error':type(error).__name__}}
            if active and estimate.get('native_eligible'):
                fontdiag=estimate['diagnostics'].get('font') or {}
                if (fontdiag.get('dimension_error',0)>TYPOGRAPHY_THRESHOLDS['dimension_error'] or
                        fontdiag.get('shape_overlap',1)<TYPOGRAPHY_THRESHOLDS['shape_overlap']):
                    estimate['native_eligible']=False
                    estimate['diagnostics'].update(status='REVIEW_REQUIRED',reason='strict_source_font_fit_refused')
            row={'id':node['id'],'kind':'typography','policy':policy,'applied':False,
                 'applied_fields':[],'diagnostics':estimate['diagnostics']}
            if active:
                row['proposal']={k:deepcopy(node.get(k)) for k in ('bbox','font','bold','color','runs')}
                row['proposal_contrast']=_source_contrast(pixels,node)
            if node['id'] in model_typography and not active:
                proposal=model_typography[node['id']]
                if node['font']!=proposal['font'] or node['bold']!=proposal['bold']:raise ValueError('model typography binding changed')
                row.update(reason='model_typography_preserved',suggestion={k:estimate.get(k) for k in ('font_family','font_size_px','bold','color','bbox')})
            elif active and policy=='typography' and estimate.get('native_eligible'):
                if _apply_typography(node,estimate,row,geometry_policy=geometry_policy,
                                     source_size=(w,h),target_size=(tw,th),scale=scale,
                                     renderer_version=renderer_version):
                    measured+=1
            if active and not row['applied'] and policy in ('color','typography'):
                _apply_colour_only(node,estimate,row,pixels,box,policies)
            if active:
                row['source_ink_bbox']=estimate.get('bbox')
                row['source_color']=estimate.get('color')
                if not row['applied']:
                    row['reason']=row.get('reason') or estimate['diagnostics'].get('reason','source_measurement_refused')
                row['needs_rendered_review']=True
                row['output_contrast']=_source_contrast(pixels,node)
                row['output']={k:deepcopy(node.get(k)) for k in ('bbox','font','bold','color','runs')}
            diagnostics.append(row)
        elif node['type']=='shape' and calibrate_fills:
            tightened=_tighten(pixels,node,original_nodes)
            if tightened:
                diagnostics.append({'id':node['id'],'kind':'geometry','applied':True,
                    'before':list(node['bbox']),'after':tightened,'basis':'declared_border_extent'});node['bbox']=tightened;tightened_count+=1
            fitted,measurement=_fill_measured(pixels,node,original_nodes)
            before={'fill':node.get('fill'),'fill_gradient':deepcopy(node.get('fill_gradient'))}
            if fitted:
                if policies['fill_surface']:
                    decision=_fill_decision(node,fitted,measurement)
                    basis='source_pixel_surface_comparison'
                else:
                    # Explicit rollback switch: the old start-colour comparison.
                    declared=_rgb(node.get('fill'))
                    drift=None if declared is None else float(np.abs(declared-np.array([int(fitted['color'][i:i+2],16) for i in (0,2,4)])).mean())
                    apply=declared is None or drift>12
                    decision={'apply':apply,'decision':'applied' if apply else 'declared_matches_source',
                              'reason':None,'candidate_kind':'legacy','declared_error':None,
                              'candidate_error':None,'legacy_drift':None if drift is None else round(drift,1),
                              'angle_semantics':None}
                    basis='legacy_declared_drift'
                candidate={'color':fitted['color'],'gradient':deepcopy(fitted.get('gradient')),
                           'dominant_fraction':fitted.get('dominant_fraction'),
                           'component_fraction':fitted.get('component_fraction'),
                           'sample_pixels':fitted.get('sample_pixels')}
                row={'id':node['id'],'kind':'fill','applied':bool(decision['apply']),
                     'basis':basis,'before':before,'candidate':candidate,
                     'projection':measurement.get('projection'),**decision}
                if decision['apply']:
                    node['fill']=fitted['color']
                    if 'gradient' in fitted:node['fill_gradient']=deepcopy(fitted['gradient'])
                    else:node.pop('fill_gradient',None)
                    row['after']={'fill':node.get('fill'),'fill_gradient':deepcopy(node.get('fill_gradient'))}
                    fills+=1
                diagnostics.append(row)
            elif measurement and measurement.get('projection_rejected'):
                diagnostics.append({'id':node['id'],'kind':'fill','applied':False,
                    'decision':'projection_rejected','basis':'single_axis_projection',
                    'before':before,'projection_rejected':measurement['projection_rejected']})
    output.mkdir(parents=True);(output/'source').mkdir();(output/'assets').mkdir();shutil.copyfile(src,output/'source/page.png');ir['source_image']='source/page.png'
    for asset in ir['assets']:
        old=irfile.parent/asset['path'];relative='assets/'+asset['id']+old.suffix;shutil.copyfile(old,output/relative);asset['path']=relative
    ir['review_notes'].append('Source-pixel typography measurements applied where eligible; refused measurements remain proposals. Font and origin estimates require actual-render review. Solid fills changed: '+str(fills)+'.')
    write_json(output/'slideir.json',ir)
    typography_rows={r['id'] for r in diagnostics if r.get('kind')=='typography' and r.get('applied')}
    colour_rows={r['id'] for r in diagnostics if r.get('kind')=='typography' and r.get('source_color_applied')}
    if active:
        authoring['original_model_typography']=model_typography
        # Keep the model font for nodes where the guarded path did not apply it.
        authoring['model_typography']={key:value for key,value in model_typography.items()
                                       if key not in typography_rows}
        authoring['applied_source_typography']={n['id']:{k:deepcopy(n[k]) for k in ('font','bold','color','bbox')} for n in ir['nodes'] if n['type']=='text' and n['id'] in typography_rows}
        authoring['applied_source_color']={n['id']:deepcopy(n.get('color')) for n in ir['nodes'] if n['type']=='text' and n['id'] in colour_rows}
        authoring['typography_policy']=policy
        authoring['geometry_policy']=geometry_policy
    authoring['tool']='supervised-elements source-pixel calibration';authoring['ir_sha256']=digest(output/'slideir.json');authoring['review_notes']=ir['review_notes'];write_json(output/'authoring.json',authoring)
    report={'schema':'ElementCalibration/1','source_sha256':ir['source_sha256'],'input_ir_sha256':digest(irfile),'output_ir_sha256':digest(output/'slideir.json'),'typography_policy':policy,'geometry_policy':geometry_policy,'typography_applied':measured,'source_color_only_applied':sum(bool(r.get('source_color_applied')) for r in diagnostics),'typography_total':sum(n['type']=='text' for n in ir['nodes']),'solid_fills_applied':fills,'geometry_tightened':tightened_count,'environment':environment,'policies':policies,'diagnostics':diagnostics,'human_acceptance':False}
    write_json(output/'calibration.json',report)
    return {k:v for k,v in report.items() if k!='diagnostics'}

if __name__=='__main__':
    import json,sys
    print(json.dumps(calibrate_elements(sys.argv[1],sys.argv[2]),ensure_ascii=False))
